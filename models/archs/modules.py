import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import models.archs.norms as norms


"""Restomer: Efficient Transformer for High-Resolution Image Restoration"""
##########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA)
class MultiDconvHeadTransposedSA(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        """
        Multi-DConv Head Transposed Self-Attention (MDTA)
        :param dim: 通道維度
        :param num_heads: 注意力頭數
        :param bias: 是否使用偏置
        """
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))  # 確保形狀與注意力權重匹配

        # Query, Key, Value 計算
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 
                                    kernel_size=3, stride=1, 
                                    padding=1, groups=dim * 3, 
                                    bias=bias)

        # 輸出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播：
        1. 使用 1x1 卷積得到 Q, K, V
        2. 使用 Depthwise 3x3 卷積加強局部特徵
        3. 計算自注意力
        4. 應用權重到 V 並輸出
        """
        _, _, h, w = x.shape
        # x = x.half()  # Convert to float16

        # 計算 Q, K, V
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)  # 拆分為 Q, K, V

        # 重新排列形狀以適應多頭注意力
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 對 Q, K 進行 L2 正規化，防止數值過大
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 計算注意力分數 (使用愛因斯坦求和 `einsum`)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.float().softmax(dim=-1) #.half()  # Softmax in float32 then convert back to float16

        # 計算加權輸出
        out = (attn @ v)

        # 恢復輸出形狀
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 最終輸出
        out = self.project_out(out)
        return out



##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class GatedDconvFFNetwork(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False):
        """
        GDFN - Gated-DConv Feed-Forward Network
        :param dim: 輸入通道數
        :param ffn_expansion_factor: FFN 擴展倍數
        :param bias: 是否使用偏置
        """
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # 1x1 卷積擴展通道數，將維度擴展為 2 倍
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 3x3 深度可分離卷積
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        # 1x1 卷積壓縮通道數
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播：
        1. `1x1 Conv` 提高維度
        2. `3x3 Depthwise Conv` 提取局部特徵
        3. `Gating Mechanism` 控制信息流
        4. `1x1 Conv` 降低維度
        """
        # x = x.half()  # Convert to float16
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 拆分通道
        x1 = nn.GELU(approximate='tanh')(x1.float())  # GELU in float32 then convert back to float16
        x = x1 * x2  # 閘控機制
        x = self.project_out(x)
        return x



##########################################################################
# TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, Norm_type, **kwargs):
        super(TransformerBlock, self).__init__()

        self.norm1 = norms.Norm(dim, Norm_type)
        self.MDTA = MultiDconvHeadTransposedSA(dim, num_heads, bias)
        self.norm2 = norms.Norm(dim, Norm_type)
        self.GDFN = GatedDconvFFNetwork(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.MDTA(self.norm1(x))
        x = x + self.GDFN(self.norm2(x))

        return x
    
    
"""Image De-Raining Transformer (WTM+STM)"""
##########################################################################
# 窗口切割
def window_partition(x, window_size):
    """
    x: (B, H, W, C)
    window_size: 窗口大小
    return: (B*num_windows, window_size*window_size, C), (pad_h, pad_w)
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # 補零
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # 只對 H 和 W 補零
    H_pad, W_pad = x.shape[1], x.shape[2]  # 更新補零後的 H, W

    # 執行 window_partition
    x = x.view(B, H_pad // window_size, window_size, W_pad // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    
    return  x

##########################################################################
# 窗口token合併
def window_to_token(x, window_size):
    """
    x: (B*num_windows, window_size*window_size, C)
    return: (1, all_window_tokens, C)
    """
    B = x.shape[0] // (x.shape[1] // window_size ** 2)
    num_windows = x.shape[0] // B
    x = x.view(B, num_windows, -1, x.shape[-1]).mean(dim=2)  # (B, num_windows, C)
    x = x.view(1, -1, x.shape[-1])  # 合併所有窗口為單一序列
    return x


##########################################################################
# REMSA
class REMSA(nn.Module):
    """ REMSA for WTM (Window-based Transformer Module) """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Self-Attention Components
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        # Depth-wise Convolution Path
        self.conv_proj = nn.Linear(dim, dim, bias=False)
        self.pointwise_conv1 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)  # 1x1 卷積
        self.depth_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)  # 深度卷積
        self.conv_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, H, W):
        B, N, C = x.shape  # N: token count
        H = self.num_heads

        # Multi-Head Self-Attention
        q = self.q_proj(x).reshape(B, N, H, -1).permute(0, 2, 1, 3)  
        k = self.k_proj(x).reshape(B, N, H, -1).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, H, -1).permute(0, 2, 1, 3)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, C)
        attn_output = self.attn_out(attn_output)

        # Depth-wise Convolution Path
        conv_input = self.conv_proj(x).permute(0, 2, 1)
        conv_output = self.depth_conv(conv_input).permute(0, 2, 1)
        conv_output = self.conv_out(conv_output)

        # Combine Attention & Convolution results
        output = attn_output + conv_output
        
        # Reshape to (B, H, W, C) to maintain image format
        output = output.view(B, H, W, C)
        
        return output

    
##########################################################################
# LeFF
class LocalEnhancedFFNetwork(nn.Module):
    """ Local-enhanced Feed-Forward Network (LeFF) """
    def __init__(self, dim):
        super().__init__()
        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)  # 1x1 卷積
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # 深度可分離卷積
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.pointwise_conv(x)  # 1x1 卷積
        x = self.depthwise_conv(x)  # 深度可分離 3x3 卷積
        x = x.permute(0, 2, 3, 1).view(B, N, C)  # 還原回 (B, N, C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.view(B, H, W, C)  # 恢復回影像格式 (B, H, W, C)
        return x
##########################################################################
# Window-based Transformer Module (WTM)
class WindowBasedTransformer(nn.Module):
    """ Window-based Transformer Module (WTM) """
    def __init__(self, dim, num_heads, window_size, norm_type='WithBias'):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norms.Norm(dim, norm_type)
        self.remsa = REMSA(dim, num_heads)
        self.norm2 = norms.Norm(dim, norm_type)
        self.leff = LocalEnhancedFFNetwork(dim)
        
    def forward(self, x):
        """
        x: (B, H, W, C)
        return: (B, H, W, C)
        """
        B, H, W, C = x.shape
        x_norm = self.norm1(x)
        x_windows, _ = window_partition(x_norm, self.window_size)
        
        # 直接對窗口內像素執行自注意力
        remsa_output = self.remsa(x_windows)
        remsa_output = remsa_output.view(B, H, W, C)
        
        # 殘差連接
        x = x + remsa_output
        
        # LeFF 計算
        x = x + self.leff(self.norm2(x))
        
        return x