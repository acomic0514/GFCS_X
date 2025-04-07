import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.modules_test import DDRB_Traceable, TraceLayer, extract_all_trace_data

##########################################################################
# DPENet_Tracable
class DPENet_Traceable(nn.Module):
    def __init__(self,
                 in_channels=3,
                 mid_channels=4,
                 kernel=3,
                 stride=1,
                 dilation_list=[1, 2, 5],
                 bias=False):
        super(DPENet_Traceable, self).__init__()

        # Initial feature transformation
        self.inconv1 = nn.Sequential(
            TraceLayer(nn.Conv2d(in_channels, mid_channels, 
                      kernel_size=1, padding=0, bias=bias)),
            TraceLayer(nn.ReLU(inplace=False))
        )
        self.outconv1 = nn.Sequential(
            TraceLayer(nn.Conv2d(mid_channels, in_channels, 
                      kernel_size=1, padding=0, bias=bias)),
            TraceLayer(nn.ReLU(inplace=False)),
        )

        # Network Modules
        self.ddrb = DDRB_Traceable(mid_channels, mid_channels, kernel, stride, dilation_list, bias)

    def forward(self, x, trace=False):
        input_ = x       

        # trace inconv
        x = self.inconv1(x)
        features_inconv = extract_all_trace_data(self.inconv1, "inconv1")  # 修改：提取 inconv1 中所有輸出與權重

        # trace DDRB
        x, features_ddrb = self.ddrb(x)  # 修改：從 DDRB 傳回 features 字典（已內建 TraceLayer）

        # trace outconv
        x = self.outconv1(x)
        features_outconv = extract_all_trace_data(self.outconv1, "outconv1")  # 修改：提取 outconv1 中所有輸出與權重

        # residual add
        x_mid = x + input_

        # 整合所有 trace 結果
        all_features = {
            **features_inconv,
            **features_ddrb,
            **features_outconv,
            "residual_output": x_mid.detach().cpu().numpy(),
            "final_output": x.detach().cpu().numpy()
        }  # 修改：彙總所有層的 trace 結果
        
        

        if trace:
            return x_mid, all_features
        else:
            return x_mid
    

