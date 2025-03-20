import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import models.archs.Norms as Norms

"""
組件目錄
Restromer
    - MDTA (Multi-DConv Head Transposed Self-Attention)
    - GDFN (Gated-Dconv Feed-Forward Network)
    - TransformerBlock 
Image De-Raining Transformer
    - window_partition(窗口切割)
    - window_to_token(窗口token合併)
    - window_merge(WTM窗口合併)
    - token_to_window(STM Token合併)
    - REMSA (relative position enhanced self-attention)
    - LeFF (Local-enhanced Feed-Forward Network)
    - WTM (Window-based Transformer Module)
    - STM (Space-based Transformer Module)
    - IDT (Image De-Raining Transformer) (WTM>>STM)
通用小工具堆放區
    - auto_num_heads(自動計算 num_heads)
"""


"""Restromer: Efficient Transformer for High-Resolution Image Restoration"""
##########################################################################
# Multi-DConv Head Transposed Self-Attention (MDTA)
class MDTA(nn.Module):
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
        x = x.half()  # ✅ 轉 float16

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
        attn = attn.float().softmax(dim=-1).half()  # Softmax 運算仍然使用 float32 再轉回 float16

        # 計算加權輸出
        out = (attn @ v)

        # 恢復輸出形狀
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 最終輸出
        out = self.project_out(out)
        return out



##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class GDFN(nn.Module):
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
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias,)

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
        x = x.half()  # ✅ 轉 float16
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 拆分通道
        x1 = nn.GELU(x1.float()).half() # GELU 在 float32 下計算
        x = torch.mul(x1, x2)  # 閘控機制
        x = self.project_out(x)
        return x



##########################################################################
# TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, Norm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = Norms.Norm(dim, Norm_type)
        self.MDTA = MDTA(dim, num_heads, bias)
        self.norm2 = Norms.Norm(dim, Norm_type)
        self.GDFN = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.MDTA(self.norm1(x))
        x = x + self.GDFN(self.norm2(x))

        return x
    
    
"""Image De-Raining Transformer (WTM+STM)"""
##########################################################################
# 窗口切割
def window_partition(x, window_size):
    """
    x: (B, C, H, W)
    window_size: 窗口大小
    return: (B*num_windows, window_size*window_size, C), (pad_h, pad_w)
    """
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # 補零
    x = F.pad(x, (0, pad_w, 0, pad_h))  # 只對 H 和 W 補零
    H_pad, W_pad = x.shape[2], x.shape[3]  # 更新補零後的 H, W

    # 執行 window_partition
    x = x.view(B, C, H_pad // window_size, window_size, W_pad // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    
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
# WTM窗口合併
def window_merge(windows, H, W, window_size):
    """
        將窗口 Token 還原回原始圖像形狀    
        windows: (B*num_windows, window_size*window_size, C)
        H, W: 原圖高寬
        window_size: 窗口大小
        return: (B, C, H, W)
    """
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    C = windows.shape[-1]

    # 重新排列回去
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)

    return x
##########################################################################
# STM Token合併
def token_to_window(tokens, H, W, window_size):
    """
    將窗口 Token 還原回原始圖像形狀 (適用於 STM)
    
    tokens: (1, num_windows, C)
    H, W: 原圖高寬
    window_size: 窗口大小
    return: (B, C, H, W)
    """
    B, num_windows, C = tokens.shape

    # 計算窗口數量
    H_windows = H // window_size
    W_windows = W // window_size

    assert num_windows == H_windows * W_windows, \
        f"窗口數量錯誤: num_windows={num_windows}, 計算得到={H_windows * W_windows}"

    # 讓 token 變成窗口
    x = tokens.view(B, H_windows, W_windows, C)  # (B, H//window_size, W//window_size, C)
    x = x.permute(0, 3, 1, 2)  # 變成 (B, C, H//Win, W//Win)

    # 重新調整成 (B, H//Win, W//Win, Win, Win, C)
    x = x.reshape(B, C, H_windows, 1, W_windows, 1).expand(-1, -1, -1, window_size, -1, window_size)

    # 調整維度，還原成 (B, C, H, W)
    x = x.reshape(B, C, H, W)

    return x



##########################################################################
# REMSA (relative position enhanced self-attention)
class REMSA(nn.Module):
    def __init__(self, dim, num_heads):
        super(REMSA, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(dim, dim * 3, dtype=torch.float16)
        self.out_proj = nn.Linear(dim, dim, dtype=torch.float16)
        
        # 位置關聯的偏置矩陣，使用可學習的相對位置編碼
        self.position_bias = nn.Parameter(torch.randn(1, num_heads, 1, 1, dtype=torch.float16) * 0.01)  # 允許學習且非對稱
        
        # 額外的深度卷積分支
        self.conv_proj = nn.Linear(dim, dim, dtype=torch.float16)
        self.pointwise_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, groups=1).to(torch.float16)
        self.depthwise_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim).to(torch.float16)
        self.token_wise_proj = nn.Linear(dim, dim, dtype=torch.float16)
    
    def scaled_dot_product_attention(self, q, k, v):
        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16) # 確保計算在 float16 上執行 
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        B, num_heads, seq_len, _ = attn_scores.shape # 確保 position_bias 與 attn_scores 形狀匹配
        if self.position_bias.shape[-2:] != (seq_len, seq_len):
            self.position_bias = nn.Parameter(torch.randn(1, num_heads, seq_len, seq_len, dtype=torch.float16) * 0.01, 
                                              requires_grad=True).to(q.device)
            
        attn_scores = attn_scores + self.position_bias  # 加上學習到的位置偏置矩陣
        attn_weights = F.softmax(attn_scores.to(torch.float32), dim=-1).to(torch.float16)  # Softmax 運算在 float32 上執行
        
        return torch.matmul(attn_weights, v)
    
    def convolutional_branch(self, x):
        x_conv = self.conv_proj(x.to(torch.float16))  # 線性投影 運算在 float16 上執行
        x_conv = x_conv.transpose(1, 2)  # 調整維度以適應 1D 卷積
        x_conv = self.pointwise_conv(x_conv)  # 深度可分卷積
        x_conv = self.depthwise_conv(x_conv)  # 深度可分卷積
        x_conv = x_conv.transpose(1, 2)  # 恢復維度順序
        return self.token_wise_proj(x_conv)  # Token-wise 線性投影
    
    def forward(self, x):
        
        #self-attntion 分支
        x = x.to(torch.float16)  # 確保計算在 float16 上執行
        batch_size, seq_length, dim = x.shape
        
        qkv = self.qkv_proj(x).reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        
        attn_output = self.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.reshape(batch_size, seq_length, dim)
        attn_output = self.out_proj(attn_output)
        
        # 卷積分支
        conv_output = self.convolutional_branch(x)
        
        return (attn_output + conv_output).to(torch.float16)  


##########################################################################
# Local-enhanced Feed-Forward Network (LeFF)
class LeFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False).to(torch.float16)  # 1x1 卷積
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim).to(torch.float16)  # 深度可分離卷積
        self.fc1 = nn.Linear(dim, dim * 4, dtype=torch.float16)
        self.fc2 = nn.Linear(dim * 4, dim, dtype=torch.float16)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.to(torch.float16)  # 確保運算在 float16
        
        x = self.pointwise_conv(x)  # 1x1 卷積
        x = self.depthwise_conv(x)  # 深度可分離 3x3 卷積
        
        x = x.permute(0, 2, 3, 1)  # 轉換為 (B, H, W, C)
        x = x.view(B, H * W, C)  # 轉換為 (B, N, C) 以符合 fc1
        
        x = self.fc1(x)
        x = self.act(x.to(torch.float32)).to(torch.float16)  # GELU 運算在 float32 上
        x = self.fc2(x)
        
        x = x.view(B, H, W, C) # 轉換為 (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # 轉回 (B, C, H, W)
        
        return x

##########################################################################
# Window-based Transformer Module (WTM)
class WTM(nn.Module):
    def __init__(self, dim, window_size, norm_type='WithBias'):
        super().__init__()
        self.num_heads = auto_num_heads(dim)  # 自動計算 num_heads
        self.window_size = window_size
        self.norm1 = Norms.Norm(dim, norm_type)
        self.remsa = REMSA(dim, self.num_heads)
        self.norm2 = Norms.Norm(dim, norm_type)
        self.leff = LeFF(dim)
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        x = x.to(torch.float16)  # 確保整個過程使用 float16
        _, _, H, W = x.shape
        
        x_norm = self.norm1(x.to(torch.float32)).to(torch.float16)  # 正規化
        x_windows = window_partition(x_norm, self.window_size) # 窗口切割
        
        # 直接對窗口內像素執行自注意力
        remsa_output = self.remsa(x_windows) # REMSA 計算
        remsa_output = window_merge(remsa_output, H, W, self.window_size)  # 正確還原形狀
        # 殘差連接
        x = x + remsa_output
        
        # LeFF 計算 + 殘差連接
        x = x + self.leff(self.norm2(x.to(torch.float32)).to(torch.float16))
        
        return x
    
##########################################################################
# Space-based Transformer Module (STM)
class STM(nn.Module):
    def __init__(self, dim, window_size, norm_type='WithBias'):
        super().__init__()
        self.num_heads = auto_num_heads(dim)  # 自動計算 num_heads
        self.window_size = window_size
        self.norm1 = Norms.Norm(dim, norm_type)
        self.remsa = REMSA(dim, self.num_heads)
        self.norm2 = Norms.Norm(dim, norm_type)
        self.leff = LeFF(dim)
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        x = x.to(torch.float16)  # 確保整個過程使用 float16
        _, _, H, W = x.shape
        
        x_norm = self.norm1(x.to(torch.float32)).to(torch.float16) # 正規化
        x_windows = window_partition(x_norm, self.window_size) # 窗口切割
        x_windows = window_to_token(x_windows, self.window_size) # Token 合併
        
        # 直接對窗口內像素執行自注意力
        remsa_output = self.remsa(x_windows) # REMSA 計算
        remsa_output = token_to_window(remsa_output, H, W, self.window_size) # 正確還原形狀
        
        # 殘差連接
        x = x + remsa_output
        
        # LeFF 計算 + 殘差連接
        x = x + self.leff(self.norm2(x.to(torch.float32)).to(torch.float16))
        
        return x  
    
##########################################################################
# Image De-Raining Transformer (WTM >> STM) 名字怎麼都好可以再想想
class IDT(nn.Module):
    """ 
    Image De-Raining Transformer (WTM >> STM)
    """
    def __init__(self, dim, window_size, norm_type='WithBias'):
        super().__init__()
        self.wtm = WTM(dim, window_size, norm_type)
        self.stm = STM(dim, window_size, norm_type)

    def forward(self, x):
        """
        x: (B, C, H, W) - 影像特徵圖
        return: (B, C, H, W) - 經過 WTM 和 STM 處理的影像特徵圖
        """
        x = self.wtm(x)  # 先經過 WTM
        x = self.stm(x)  # 再經過 STM
        return x
    
    """通用小工具堆放區"""
##########################################################################
# 自動計算 num_heads
def auto_num_heads(dim):
    """標準 Transformer 風格的 num_heads 設定"""
    if dim >= 128:
        return 8
    elif dim >= 64:
        return 4
    elif dim >= 32:
        return 2
    else:
        return 1