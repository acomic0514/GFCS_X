import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.modules import DDRB, ERPAB

##########################################################################
# DPENet_v1 without CFIM
class DPENet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 mid_channels=32,
                 kernel=3,
                 stride=1,
                 dilation_list=[1, 2, 5],
                 bias=False):
        super(DPENet, self).__init__()

        # Initial feature transformation
        self.inconv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=bias)
        self.outconv1 = nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0, bias=bias)
        self.inconv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=bias)
        self.outconv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0, bias=bias)

        # Network Modules
        self.ddrb = nn.Sequential(*[DDRB(mid_channels, mid_channels, kernel, stride, dilation_list, bias) for _ in range(10)])

        # Shared ERPAB instance
        self.erpab = nn.Sequential(*[ERPAB(mid_channels, mid_channels, kernel, stride, dilation_list, bias) for _ in range(3)])


    def forward(self, x):
        input_ = x
        
        # Stage 1: Initial Rain Streaks Removal
        x = self.inconv1(x)
        rs1 = self.ddrb(x)
        x = self.outconv1(rs1)
        x_mid = x + input_  # Residual connection
        
        # Stage 2: Initial Detail Reconstruction
        x = self.inconv2(F.relu(x_mid))
        dr1 = self.erpab(x)
        x = self.outconv2(dr1)
        x_final = x + x_mid  # Residual connection

        return x_final