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
    def __init__(self, img_size=224, patch_size=32, base_patch_size=32, in_channel=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        base_patch_size = to_2tuple(base_patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        grid_h, grid_w = img_size[0] // base_patch_size[0], img_size[1] // base_patch_size[1]
        hs, ws = np.meshgrid(np.arange(grid_h), np.arange(grid_w))
        hs, ws = hs.flatten(), ws.flatten()
        hws = [(h, w) for h, w in zip(hs, ws)]

        def split_hw(hw):
            h, w = hw
            return (h*2, w*2), (h*2+1, w*2), (h*2, w*2+1), (h*2+1, w*2+1)

        while grid_h < self.grid_size[0]:
            hws = [split_hw(hw) for hw in hws] # hws is a list of tuples
            hws = [hw for subtuple in hws for hw in subtuple] # flatten it to a list
            grid_h, grid_w = grid_h * 2, grid_w * 2

        token_indices = torch.tensor([hw[0]+hw[1]*grid_w for hw in hws], dtype=torch.long)
        self.token_indices = token_indices.reshape(1, self.num_patches, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        token_indices = self.token_indices.to(x.device).expand(B, -1, self.embed_dim)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # BCHW -> BNC
            x = torch.take_along_dim(x, token_indices, dim=1)
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
        cls_attn = attn[:, :, 1:, 0].mean(axis=1) # B*(N-1)
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
                 img_size=224, in_channel=3, patch_size=32, base_patch_size=32, split_ratio=0.5,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super().__init__()
        self.num_features = embed_dim
        self.num_tokens = 1
        self.split_ratio = split_ratio
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, in_channel=in_channel, embed_dim=embed_dim,
            patch_size=patch_size, base_patch_size=base_patch_size
        )
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=self.num_features, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=None, norm_layer=norm_layer, act_layer=act_layer
            ) for i in range(num_block)
        ])

    def forward(self, x, prev_downsample, prev_cls_attn, image):
        # embed the original image to get low-level feature
        embeds = self.patch_embed(image)
        cls_token = self.cls_token.expand(embeds.shape[0], -1, -1)
        embeds = torch.cat((cls_token, embeds), dim=1)
        embeds = self.pos_drop(embeds + self.pos_embed)

        # aggregate low-level feature and high-level feature from previous stage
        if x is None:
            x = embeds
            downsample = [[[1]] * x.shape[1]] * x.shape[0]
        else:
            # determine which tokens to split
            B, N, C = x.shape
            prev_cls_attn = prev_cls_attn.detach().cpu()
            batch_split_index = torch.topk(prev_cls_attn, k=int(N*self.split_ratio), dim=1).indices
            batch_split_mask = torch.zeros_like(prev_cls_attn).scatter_(1, batch_split_index, 1).numpy()

            # downsample low-level feature
            downsample = []
            downsample_matrices = []
            for split_mask, prev_downsample_blocks in zip(batch_split_mask, prev_downsample):
                downsample_blocks = []
                for split, block in zip(split_mask, prev_downsample_blocks):
                    if split == 1:
                        downsample_blocks += [block] * 4
                    else:
                        downsample_blocks.append([0.25 * element for element in block for repeat in range(4)])
                downsample.append(downsample_blocks)
                downsample_blocks = [[1]] + downsample_blocks
                downsample_matrix = block_diag(*downsample_blocks)
                downsample_matrices.append(downsample_matrix)
            downsample_matrices = np.array(downsample_matrices)
            downsample_matrices = torch.from_numpy(downsample_matrices).float().to(x.device)
            embeds = downsample_matrices @ embeds

            # upsample high-level feature
            upsample_matrices = []
            for split_mask in batch_split_mask:
                upsample_blocks = [[1]]
                for split in split_mask:
                    if split == 1:
                        upsample_blocks.append([[1], [1], [1], [1]])
                    else:
                        upsample_blocks.append([1])
                upsample_matrix = block_diag(*upsample_blocks)
                upsample_matrices.append(upsample_matrix)
            upsample_matrices = np.array(upsample_matrices)
            upsample_matrices = torch.from_numpy(upsample_matrices).float().to(x.device)
            x = upsample_matrices @ x
                
            # aggregate using concat
            x += embeds

        # process the aggregated feature
        for block in self.blocks:
            x, cls_attn = block(x)
        return x, downsample, cls_attn


class DynamicResolutionViT(nn.Module):
    def __init__(self, img_size=224, base_patch_size=32, in_channel=3, num_classes=1000, embed_dim=192,
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
        stages = []
        patch_size = base_patch_size * 2
        for i_stage in range(self.num_stages):
            patch_size = patch_size // 2
            stages.append(Stage(
                embed_dim=embed_dim, num_block=num_blocks[i_stage],
                img_size=img_size, in_channel=in_channel,
                patch_size=patch_size, base_patch_size=base_patch_size, split_ratio=split_ratios[i_stage],
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

    def forward_features(self, image):
        x = None
        downsample = None
        cls_attn = None
        for stage in self.stages:
            x, downsample, cls_attn = stage(x, downsample, cls_attn, image)
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
