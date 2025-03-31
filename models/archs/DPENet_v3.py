import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.modules import DDRB, ERPAB

class ERPAB_v2(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, hidden_dim=16):
        super().__init__()
        
        self.experts = nn.ModuleList([
            nn.AvgPool2d(3, stride=1, padding=1),  # 3×3 pooling
            nn.Conv2d(in_channels, out_channels, 1),  # 1×1 conv
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 3×3 conv
            nn.Conv2d(in_channels, out_channels, 5, padding=2),  # 5×5 conv
            nn.Conv2d(in_channels, out_channels, 7, padding=3),  # 7×7 conv
            nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3),  # 3×3 dilated
            nn.Conv2d(in_channels, out_channels, 5, padding=6, dilation=3),  # 5×5 dilated
            nn.Conv2d(in_channels, out_channels, 7, padding=9, dilation=3)   # 7×7 dilated
        ])
        
        self.num_experts = len(self.experts)  # 動態計算專家數量
        
        # 兩層可學習權重 W1, W2
        self.W1 = nn.Linear(in_channels, hidden_dim)  # W1: (T x C)
        self.W2 = nn.Linear(hidden_dim, self.num_experts)  # W2: (O x T)
        
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)  # 1×1 conv
        
        self.attn_map = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        b, _, _, _ = x.shape
        
        with torch.amp.autocast('cuda'):  # ✅ AMP 自動管理精度
            # 通道平均池化計算 z_c
            z_c = x.mean(dim=[2, 3])  # (B, C)
            
            # 使用 W1, W2 計算專家權重
            weights = self.W2(F.relu(self.W1(z_c)))  # (B, num_experts)
            weights = torch.softmax(weights, dim=1)  
            
            # 計算專家輸出
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, num_experts, C, H, W)
            mixture = torch.sum(weights.view(b, -1, 1, 1, 1) * expert_outputs, dim=1)  # (B, C, H, W)
            mixture = F.relu(self.final_conv(mixture))
            
            attn_map = self.attn_map(mixture)
            output = x + mixture * F.sigmoid(attn_map)
            
        return output
    



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
        self.erpab = nn.Sequential(*[ERPAB_v2(in_channels=mid_channels, out_channels=mid_channels, hidden_dim=16) for _ in range(3)])


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