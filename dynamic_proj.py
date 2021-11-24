import collections.abc
from functools import partial
from itertools import repeat
from scipy.linalg import block_diag
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_channel=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3BHNC
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale # BHNN
        attn = attn.softmax(dim=-1)
        cls_attn = attn[:, :, 0, 1:].mean(axis=1).squeeze(1) # B*(N-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, cls_attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # TODO: check what is DropPath
        # self.drop_path = DropPath()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        attn_output, cls_attn = self.attn(self.norm1(x))
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x, cls_attn


class Stage(nn.Module):
    def __init__(self, embed_dim, num_block,
                 img_size=224, in_channel=3, alpha=0.5,
                 patch_size=32, split_ratio=0.5,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super().__init__()
        self.num_features = embed_dim
        if split_ratio is not None:
            self.split_proj = nn.Linear(embed_dim, embed_dim * 4)
            self.alpha = alpha
            assert split_ratio >= 0 and split_ratio <= 1, "split ratio can only be in [0, 1]"
        self.split_ratio = split_ratio
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.alpha = alpha

        self.blocks = nn.ModuleList([
            Block(
                dim=self.num_features, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=None, norm_layer=norm_layer, act_layer=act_layer
            ) for i in range(num_block)
        ])

    def forward(self, x, glb_cls_attn):
        if self.split_ratio is not None: # split tokens
            B, N, C = x.shape
            # determine which tokens to split
            cls_token = x[:, :1, :]
            tokens = x[:, 1:, :]
            B, N, C = tokens.shape
            token_importance = torch.argsort(glb_cls_attn, dim=1, descending=True).unsqueeze(2).expand(-1, -1, C)
            tokens = torch.take_along_dim(tokens, token_importance, dim=1)

            # split important tokens
            split_n = int(N * self.split_ratio)
            split_tokens = tokens[:, :split_n, :]
            split_tokens = self.split_proj(split_tokens).reshape(B, split_n * 4, C)

            # aggregate tokens
            keep_tokens = tokens[:, split_n:, :]
            x = torch.cat((cls_token, split_tokens, keep_tokens), dim=1)

        # normal update process
        B, N, C = x.shape # update N
        glb_cls_attn = torch.zeros((B, N-1)).to(x.device)
        for block in self.blocks:
            x, cls_attn = block(x)
            glb_cls_attn = self.alpha * glb_cls_attn + (1 - self.alpha) * cls_attn
        return x, glb_cls_attn


class DynamicResolutionViT(nn.Module):
    def __init__(self, img_size=224, base_patch_size=32, in_channel=3, num_classes=1000, embed_dim=192, alpha=0.5,
                 num_blocks=(2, 6, 2, 2), split_ratios=(None, 0.4, 0.4, 0.4), num_heads=3, mlp_ratio=4., qkv_bias=True, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super().__init__()
        self.num_classes = num_classes
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        assert len(num_blocks) == len(split_ratios),\
            "length of split_ratios (%d) and num_blocks (%d) must be identical" % (len(split_ratios), len(num_blocks))
        self.num_stages = len(num_blocks)
        assert base_patch_size % (2 ** (self.num_stages - 1)) == 0,\
            "the base_patch_size (%d) has not enough factor 2's" % (base_patch_size)

        self.num_features = embed_dim
        self.num_tokens = 1
        self.patch_embed = embed_layer(
            img_size=img_size, in_channel=in_channel, embed_dim=embed_dim,
            patch_size=base_patch_size
        )
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.alpha = alpha

        stages = []
        patch_size = base_patch_size * 2
        for i_stage in range(self.num_stages):
            patch_size = patch_size // 2
            stages.append(Stage(
                embed_dim=embed_dim, num_block=num_blocks[i_stage],
                img_size=img_size, in_channel=in_channel, alpha=alpha,
                patch_size=patch_size, split_ratio=split_ratios[i_stage],
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer
            ))
        self.stages = nn.ModuleList(stages)
        self.num_features = embed_dim
        self.norm = norm_layer(embed_dim)

        # representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequantial(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # embed the original image to get low-level feature
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        glb_cls_attn = None
        for stage in self.stages:
            x, glb_cls_attn = stage(x, glb_cls_attn)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def get_test_model(num_classes):
    return DynamicResolutionViT(
        num_classes=num_classes
    )
