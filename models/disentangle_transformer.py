#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: disentangle_transformer.py
@author:zyl
@contact:zhangyilan@buaa.edu.cn
@time:2024/3/28 21:58
'''

import torch
import torch.nn as nn


'''Disentangled Transformer'''
class Mlp(nn.Module):
    # two mlp, fc-relu-drop-fc-relu-drop
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MIAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super(MIAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, histology, pathways, global_feature,return_attn = False):
        B,N_histology,C = histology.shape
        _,N_pathways,_ = pathways.shape

        qkv_histology = self.qkv(histology).reshape(B, N_histology, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_histology, k_histology, v_histology = qkv_histology[0], qkv_histology[1], qkv_histology[2]

        qkv_pathways = self.qkv(pathways).reshape(B, N_pathways, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_pathways, k_pathways, v_pathways = qkv_pathways[0], qkv_pathways[1], qkv_pathways[2]

        qkv_global = self.qkv(global_feature).reshape(B, 1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_global, k_global, v_global = qkv_global[0], qkv_global[1], qkv_global[2]

        # self-attention for modality-specific features
        attn_histology = (q_histology @ k_histology.transpose(-2, -1)) * self.scale
        attn_pathways = (q_pathways @ k_pathways.transpose(-2, -1)) * self.scale

        # cross-attention for modality-common features
        attn_global = (q_global @ torch.cat((k_global,k_histology,k_pathways),dim = 2).transpose(-2, -1)) * self.scale

        attn_histology = attn_histology.softmax(dim=-1)
        attn_pathways = attn_pathways.softmax(dim=-1)
        attn_global = attn_global.softmax(dim=-1)

        attn_histology = self.attn_drop(attn_histology)
        attn_pathways = self.attn_drop(attn_pathways)
        attn_global = self.attn_drop(attn_global)

        attn_histology_x = (attn_histology @ v_histology).transpose(1, 2).reshape(B, N_histology, C)
        attn_pathways_x = (attn_pathways @ v_pathways).transpose(1, 2).reshape(B, N_pathways, C)
        attn_global_x = (attn_global @ torch.cat((v_global,v_histology,v_pathways),dim = 2)).transpose(1, 2).reshape(B, 1, C)

        attn_histology_x = self.proj(attn_histology_x)
        attn_pathways_x = self.proj(attn_pathways_x)
        attn_global_x = self.proj(attn_global_x)

        attn_histology_x = self.proj_drop(attn_histology_x)
        attn_pathways_x = self.proj_drop(attn_pathways_x)
        attn_global_x = self.proj_drop(attn_global_x)

        if return_attn:
            return [attn_histology_x, attn_pathways_x, attn_global_x], [attn_histology, attn_pathways]
        else:
            return [attn_histology_x, attn_pathways_x, attn_global_x]

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class MITransformerLayer(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            mlp_ratio=1.,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        ):
        super(MITransformerLayer, self).__init__()
        self.norm1 = norm_layer(dim)

        self.attn = MIAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_hidden_dim,act_layer=act_layer, drop=drop_path)

    def forward(self, histology, pathways, global_feature, return_attn = False):
        if return_attn:
            [attn_histology_x, attn_pathways_x, attn_global_x], \
            [attn_histology,attn_pathways,] = self.attn(self.norm1(histology), self.norm1(pathways), self.norm1(global_feature),return_attn)
        else:
            attn_histology_x, attn_pathways_x, attn_global_x = self.attn(self.norm1(histology), self.norm1(pathways), self.norm1(global_feature),return_attn)

        histology = histology + self.drop_path(attn_histology_x)
        pathways = pathways + self.drop_path(attn_pathways_x)
        global_feature = global_feature + self.drop_path(attn_global_x)

        histology = histology + self.drop_path(self.mlp(self.norm2(histology)))
        pathways = pathways + self.drop_path(self.mlp(self.norm2(pathways)))
        global_feature = global_feature + self.drop_path(self.mlp(self.norm2(global_feature)))

        if return_attn:
            return histology, pathways,  global_feature, [attn_histology,attn_pathways]
        else:
            return histology, pathways, global_feature