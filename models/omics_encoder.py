#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: omics_encoder.py
@author:zyl
@contact:zhangyilan@buaa.edu.cn
@time:2024/3/28 22:03
'''

import torch.nn as nn

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))