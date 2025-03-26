import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.modules import DDRB, ERPAB, CFIM

##########################################################################
# DPENet_v2 with CFIM
class DPENet_CFIM(nn.Module):
    def __init__(self,
                 in_channels=3,
                 mid_channels=32,
                 kernel=3,
                 stride=1,
                 dilation_list=[1, 2, 5],
                 bias=False):
        super(DPENet_CFIM, self).__init__()

        # Initial feature transformation
        self.inconv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=bias)
        self.outconv1 = nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0, bias=bias)
        self.inconv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=bias)
        self.outconv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0, bias=bias)
        self.inconv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=bias)
        self.outconv3 = nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0, bias=bias)

        # Network Modules
        self.ddrb1 = nn.Sequential(*[DDRB(mid_channels, mid_channels, kernel, stride, dilation_list, bias) for _ in range(5)])
        self.ddrb2 = nn.Sequential(*[DDRB(mid_channels, mid_channels, kernel, stride, dilation_list, bias) for _ in range(5)])
        
        # Shared ERPAB instance
        self.erpab1 = ERPAB(mid_channels, mid_channels, kernel, stride, dilation_list, bias)
        self.erpab2 = nn.Sequential(*[ERPAB(mid_channels, mid_channels, kernel, stride, dilation_list, bias) for _ in range(2)])
        
        self.cfim = CFIM(mid_channels)

    def forward(self, x):
        input_ = x
        
        # Stage 1: Initial Rain Streaks Removal
        x = self.inconv1(x)
        rs1 = self.ddrb1(x)
        x = self.outconv1(rs1)
        x_mid = x + input_  # Residual connection
        
        # Stage 2: Initial Detail Reconstruction
        x = self.inconv2(F.relu(x_mid))
        dr1 = self.erpab1(x)
        
        # Cross-stage Feature Interaction
        rs2, _ = self.cfim(rs1, dr1)
        
        # Stage 3: Further Rain Streaks Removal
        x = self.ddrb2(rs2)
        x = self.outconv2(x)
        x_rain_removed = x + x_mid  # Residual connection
        
        # Stage 4: Further Detail Reconstruction
        x = self.inconv3(x_rain_removed)
        dr2 = self.erpab1(x)
        
        # Cross-stage Feature Interaction
        _, dr3 = self.cfim(rs1, dr2)
        
        # Final Detail Enhancement
        x = self.erpab2(dr3)
        x = self.outconv3(x)
        x_final = x + x_rain_removed  # Residual connection
        
        return x_final #如果需要輸出除雨中間結果x_rain_removed要再調整