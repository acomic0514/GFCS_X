import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.modules import DDRB, ERPAB
import models.archs.norms as norms

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
        self.ddrb = nn.Sequential(*[DDRB(mid_channels, mid_channels, kernel, stride, dilation_list, bias) for _ in range(5)])

        # Shared ERPAB instance
        self.erpab = nn.Sequential(*[ERPAB(mid_channels, mid_channels, kernel, stride, dilation_list, bias) for _ in range(3)])

        

    def forward(self, x):
        input_ = x
        #self.check_nan_inf(x, "x")
        
        # Stage 1: Initial Rain Streaks Removal
        x = self.inconv1(x)
        # self.check_nan_inf(x, "x_inconv1")
        rs1 = self.ddrb(x)
        # self.check_nan_inf(rs1, "rs1")
        x = self.outconv1(rs1)
        # self.check_nan_inf(x, "x_outconv1")
        x_mid = x + input_  # Apply Tanh activation
        # x_mid =F.sigmoid(x + input_)  # Apply Tanh activation
        # self.check_nan_inf(x_mid, "x_mid")
        
        # Stage 2: Initial Detail Reconstruction
        x = self.inconv2(F.relu(x_mid))
        # self.check_nan_inf(x, "x_inconv2")
        dr1 = self.erpab(x)
        # self.check_nan_inf(dr1, "dr1")
        x = self.outconv2(dr1)
        # self.check_nan_inf(x, "x_outconv2")
        x_final = x + x_mid
        # x_final = F.sigmoid(x + x_mid)
        # self.check_nan_inf(x_final, "x_final")

        return x_mid, x_final
    
        # if self.check_nan_inf(x_final, "x_final"):
            # break_flag = True
        #return x_mid, x_final , break_flag
        
    def check_nan_inf(self, x, comment=""):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️ 發現 NaN 或 Inf，檢查哪張圖片有問題...")
            
            # 找出包含 NaN 或 Inf 的圖片索引
            problematic_indices = []
            for i in range(x.shape[0]):
                if torch.isnan(x[i]).any() or torch.isinf(x[i]).any():
                    problematic_indices.append(i)
            print(f"❌ comment: {comment}")
            print(f"❌ 有問題的圖片索引：{problematic_indices}")
            # print(f"❌ 有問題得圖片數值為：{x[problematic_indices]}")
            print(f"❌ 有問題的圖片最大數值為：{x[problematic_indices].max().item()}")
            print(f"❌ 有問題的圖片最小數值為：{x[problematic_indices].min().item()}")
            
            return True