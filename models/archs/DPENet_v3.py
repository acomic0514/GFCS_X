import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.modules import DDRB, ERPAB

##########################################################################
#
class ERPAB_v2(nn.Module):
    """ 
    Enhanced Residual Pixel-wise Attention Block 
    Usage:
        self.erpab = ERPAB(in_channels=32, mid_channels=32, kernel=3, stride=1, d=[1, 2, 5], bias=False)
    """
    def __init__(self,
                 in_channels=32,
                 mid_channels=32,
                 kernel=3,
                 stride=1,
                 d=[1, 2, 5],
                 bias=False):
        super(ERPAB, self).__init__()
        
        self.experts = nn.ModuleList([
            nn.AvgPool2d(3, stride=1, padding=1),  # 3×3 pooling
            nn.Conv2d(in_channels, mid_channels, 1),  # 1×1 conv
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),  # 3×3 conv
            nn.Conv2d(in_channels, mid_channels, 5, padding=2),  # 5×5 conv
            nn.Conv2d(in_channels, mid_channels, 7, padding=3),  # 7×7 conv
            nn.Conv2d(in_channels, mid_channels, 3, padding=3, dilation=3),  # 3×3 dilated
            nn.Conv2d(in_channels, mid_channels, 5, padding=6, dilation=3),  # 5×5 dilated
            nn.Conv2d(in_channels, mid_channels, 7, padding=9, dilation=3)   # 7×7 dilated
        ])
        
        self.conv1 = nn.Conv2d(mid_channels*8, mid_channels, kernel_size=3, padding=1, bias=False)
        self.attn_map = nn.Sequential(
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: input feature map
        Returns:
            enhanced feature map
        Usage:
            enhanced_feature = ERPAB(input_feature)
        """
        with torch.cuda.amp.autocast():
            expert_outputs = torch.cat([expert(x) for expert in self.experts], dim=1)
            x1 = F.relu(self.conv1(expert_outputs))
            attn_map = self.attn_map(x1)

        return x + x1 * self.sigmoid(attn_map)


##########################################################################
# DPENet_v1 without CFIM and modified 
class DPENet_v3(nn.Module):
    def __init__(self,
                 in_channels=3,
                 mid_channels=32,
                 kernel=3,
                 stride=1,
                 dilation_list=[1, 2, 5],
                 bias=False):
        super(DPENet_v3, self).__init__()

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

        return x_mid, x_final