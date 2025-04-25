#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


##########################################################################
# LayerNorm
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, input):
        output = F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        return output
    
###########################################################################
# LayerNorm for CNN inputs [B, C, H, W]
class LayerNorm_CNN(nn.Module):
    """LayerNorm for CNN inputs [B, C, H, W] using GroupNorm equivalence"""
    def __init__(self, num_channels: int, bias: bool = True):
        super().__init__()
        self.body = nn.GroupNorm(num_groups=1, num_channels=num_channels, affine=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

##########################################################################
# Dynamic Tanh (DyT)
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape: int, init_alpha=1.0):
        super(DynamicTanh, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        # ✅ 確保所有參數都在與 `x` 相同的設備
        alpha = self.alpha.to(x.device)
        gamma = self.gamma.to(x.device)
        beta = self.beta.to(x.device)
        
        x = torch.tanh(alpha * x)
        return gamma.view(1, -1, 1, 1) * x + beta.view(1, -1, 1, 1)
   
##########################################################################
#Norm 方式選擇(可使用DyT)
class Norm(nn.Module):
    """General LayerNorm wrapper supporting both BiasFree and WithBias variants"""
    def __init__(self, dim: int, norm_type: str = 'WithBias'):
        super().__init__()
        valid_types = {'WithBias', 'BiasFree', 'WithBiasCNN', 'BiasFreeCNN', 'DyT'}
        if norm_type not in valid_types:
            raise ValueError(f"❌ Norm方法請選擇其中之一: {valid_types}")
        
        if norm_type == 'BiasFree':
            self.body = LayerNorm(dim, bias=False)
        elif norm_type == 'WithBias':
            self.body = LayerNorm(dim, bias=True)
        elif norm_type == 'BiasFreeCNN':
            self.body = LayerNorm_CNN(dim, bias=False)
        elif norm_type == 'WithBiasCNN':
            self.body = LayerNorm_CNN(dim, bias=True)
        elif norm_type == 'DyT':
            self.body = DynamicTanh(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
