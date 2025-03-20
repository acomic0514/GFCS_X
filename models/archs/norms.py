#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from einops import rearrange


def to_3d(x):
    """Reshape from (B, C, H, W) to (B, HW, C)"""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """Reshape from (B, HW, C) to (B, C, H, W)"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


##########################################################################
# Dynamic Tanh (DyT)
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape: int, init_alpha=1.0):
        super(DynamicTanh, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)
        
##########################################################################
#Layer Norm

class BiasFreeLayerNorm(nn.Module):
    """LayerNorm without bias"""
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    """LayerNorm with bias"""
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

##########################################################################
#Norm 方式選擇(可使用DyT)
class Norm(nn.Module):
    """General LayerNorm wrapper supporting both BiasFree and WithBias variants"""
    def __init__(self, dim: int, norm_type: str = 'WithBias'):
        super().__init__()
        valid_types = {'WithBias', 'BiasFree', 'DyT'}
        if norm_type not in valid_types:
            raise ValueError("❌ Norm方法請選擇 'WithBias'、'BiasFree' 或 'DyT' 模式")
        
        self.norm_type = norm_type
        if norm_type == 'BiasFree':
            self.body = BiasFreeLayerNorm(dim)
        elif norm_type == 'DyT':
            self.body = DynamicTanh(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if self.norm_type == 'DyT':
            return to_4d(self.body(to_3d(x)), h, w)
        else :
            return to_4d(self.body(to_3d(x).float()).half(), h, w)