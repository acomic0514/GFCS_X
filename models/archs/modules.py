import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import models.archs.norms as norms

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
Hybrid CNN-Transformer Feature Fusion
    - DegradationAwareMoE (Degradation-aware mixture of experts)
Token statistics transformer
    - CausalSelfAttention_TSSA (ToST 版本的自注意力)
    - ToSTBlock (ToST 版本的 Transformer 塊)
A novel dual-stage progressive enhancement network for single image deraining
    - dilated dense residual block (DDRB)
    - enhanced residual pixel-wise attention block (ERPAB)
    - cross-stage feature interaction module (CFIM)    
通用小工具堆放區
    - auto_num_heads(自動計算 num_heads)
    - to_3d(影像轉token序列)
    - to_4d(token序列轉影像)
    - MLP (flaoat16)
"""


"""Restromer: Efficient Transformer for High-Resolution Image Restoration"""
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
        x: (B, C, H, W) - 輸入特徵圖
        return: (B, C, H, W) - 輸出特徵圖
        前向傳播：
        1. 使用 1x1 卷積得到 Q, K, V
        2. 使用 Depthwise 3x3 卷積加強局部特徵
        3. 計算自注意力
        4. 應用權重到 V 並輸出
        """
        input_ = x
        _, _, h, w = x.shape
        
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
        attn = (torch.matmul(q, k.transpose(-2, -1))) * self.temperature
        attn = attn.softmax(dim=-1)  

        # 計算加權輸出
        out = torch.matmul(attn, v)

        # 恢復輸出形狀
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', 
                        head=self.num_heads, h=h, w=w)

        # 最終輸出
        out = self.project_out(out) + input_  # 殘差連接
        return out



##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class GatedDconvFFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, 
                 bias: bool = False):
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
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                kernel_size=3, stride=1, padding=1, 
                                groups=hidden_features * 2, bias=bias)

        # 1x1 卷積壓縮通道數
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) - 輸入特徵圖
        return: (B, C, H, W) - 輸出特徵圖
        前向傳播：
        1. `1x1 Conv` 提高維度
        2. `3x3 Depthwise Conv` 提取局部特徵
        3. `Gating Mechanism` 控制信息流
        4. `1x1 Conv` 降低維度
        """
        input_ = x
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 拆分通道
        x = torch.mul(F.gelu(x1), x2)  # 閘控機制
        x = self.project_out(x) + input_  # 殘差連接
        return x



##########################################################################
# TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim = 32, num_heads = 4, ffn_expansion_factor = 2.66, bias = False, Norm_type = 'WithBiasCNN'):
        super(TransformerBlock, self).__init__()

        self.norm1 = norms.Norm(dim, Norm_type)
        self.MDTA = MultiDconvHeadTransposedSA(dim, num_heads, bias)
        self.norm2 = norms.Norm(dim, Norm_type)
        self.GDFN = GatedDconvFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        
        x = self.norm1(x)
        x = self.MDTA(x)
        x = self.norm2(x)
        x = self.GDFN(x)

        return x
    
    
"""Image De-Raining Transformer (WTM+STM)"""
##########################################################################
# 窗口切割
def window_partition(x, window_size):
    """
    x: (B, C, H, W)
    window_size: 窗口大小
    return: (B*num_windows, window_size*window_size, C)
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
    return: (B, all_window_tokens, C)
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
class RelativePositionEnhancedSA(nn.Module):
    def __init__(self, dim, num_heads):
        super(RelativePositionEnhancedSA, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        
        # 位置關聯的偏置矩陣，使用可學習的相對位置編碼
        self.position_bias = nn.Parameter(torch.randn((1, num_heads, 1, 1)) * 0.01)  # 允許學習且非對稱
        
        # 額外的深度卷積分支
        self.conv_proj = nn.Linear(dim, dim)
        self.pointwise_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, groups=1)
        self.depthwise_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)
        self.token_wise_proj = nn.Linear(dim, dim)
    
    def scaled_dot_product_attention(self, q, k, v):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        B, num_heads, seq_len, _ = attn_scores.shape # 確保 position_bias 與 attn_scores 形狀匹配
        if self.position_bias.shape[-2:] != (seq_len, seq_len):
            self.position_bias = nn.Parameter(torch.randn(1, num_heads, seq_len, seq_len) * 0.01, 
                                              requires_grad=True).to(q.device)
            
        attn_scores = attn_scores + self.position_bias  # 加上學習到的位置偏置矩陣
        attn_weights = F.softmax(attn_scores, dim=-1)  # Softmax 
        
        return torch.matmul(attn_weights, v)
    
    def convolutional_branch(self, x):
        x_conv = self.conv_proj(x) # 1x1 卷積
        x_conv = x_conv.transpose(1, 2)  # 調整維度以適應 1D 卷積
        x_conv = self.pointwise_conv(x_conv)  # 深度可分卷積
        x_conv = self.depthwise_conv(x_conv)  # 深度可分卷積
        x_conv = x_conv.transpose(1, 2)  # 恢復維度順序
        return self.token_wise_proj(x_conv)  # Token-wise 線性投影
    
    def forward(self, x):
        """
        x: (B, N, C) - Token 序列
        return: (B, N, C) - 輸出 Token 序列
        """
        
        #self-attntion 分支
        B, N, C = x.shape
        
        with torch.amp.autocast('cuda'):  # ✅ AMP 自動管理精度
            qkv = self.qkv_proj(x).reshape(B, N, self.num_heads, 3 * self.head_dim)
            q, k, v = qkv.chunk(3, dim=-1)
            
            attn_output = self.scaled_dot_product_attention(q, k, v)
            attn_output = attn_output.reshape(B, N, C)
            attn_output = self.out_proj(attn_output)
            
            # 卷積分支
            conv_output = self.convolutional_branch(x)
            
        return attn_output + conv_output  


##########################################################################
# Local-enhanced Feed-Forward Network (LeFF)
class LocalEnhancedFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)  # 1x1 卷積
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # 深度可分離卷積
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        """
        x: (B, C, H, W) - 輸入特徵圖
        return: (B, C, H, W) - 輸出特徵圖
        """
        B, C, H, W = x.shape
        
        with torch.amp.autocast('cuda'):  # ✅ AMP 自動管理精度
            x = self.pointwise_conv(x)  # 1x1 卷積
            x = self.depthwise_conv(x)  # 深度可分離 3x3 卷積
            
            x = x.permute(0, 2, 3, 1)  # 轉換為 (B, H, W, C)
            x = x.view(B, H * W, C)  # 轉換為 (B, N, C) 以符合 fc1
            
            x = self.fc1(x)
            x = F.gelu(x)  
            x = self.fc2(x)
            
            x = x.view(B, H, W, C) # 轉換為 (B, H, W, C)
            x = x.permute(0, 3, 1, 2)  # 轉回 (B, C, H, W)
        
        return x

##########################################################################
# Window-based Transformer Module (WTM)
class WindowBasedTM(nn.Module):
    def __init__(self, dim, window_size, norm_type='WithBias'):
        super().__init__()
        self.num_heads = auto_num_heads(dim)  # 自動計算 num_heads
        self.window_size = window_size
        self.norm1 = norms.Norm(dim, norm_type)
        self.remsa = RelativePositionEnhancedSA(dim, self.num_heads)
        self.norm2 = norms.Norm(dim, norm_type)
        self.leff = LocalEnhancedFFN(dim)
        
    def forward(self, x):
        """
        x: (B, C, H, W) - 影像特徵圖
        return: (B, C, H, W) - 經過 WTM 處理的影像特徵圖
        """
        _, _, H, W = x.shape
        
        with torch.amp.autocast('cuda'):  # ✅ AMP 自動管理精度
            x_norm = to_4d(self.norm1(to_3d(x)), H, W)  # 正規化
            x_windows = window_partition(x_norm, self.window_size) # 窗口切割
            
            # 直接對窗口內像素執行自注意力
            remsa_output = self.remsa(x_windows) # REMSA 計算
            remsa_output = window_merge(remsa_output, H, W, self.window_size)  # 正確還原形狀
            # 殘差連接
            x = x + remsa_output
            
            # LeFF 計算 + 殘差連接
            x = x + self.leff(to_4d(self.norm2(to_3d(x)), H, W))
        
        return x
    
##########################################################################
# Space-based Transformer Module (STM)
class SpaceBasedTM(nn.Module):
    def __init__(self, dim, window_size, norm_type='WithBias'):
        super().__init__()
        self.num_heads = auto_num_heads(dim)  # 自動計算 num_heads
        self.window_size = window_size
        self.norm1 = norms.Norm(dim, norm_type)
        self.remsa = RelativePositionEnhancedSA(dim, self.num_heads)
        self.norm2 = norms.Norm(dim, norm_type)
        self.leff = LocalEnhancedFFN(dim)
        
    def forward(self, x):
        """
        x: (B, C, H, W) - 影像特徵圖
        return: (B, C, H, W) - 經過 STM 處理的影像特徵圖
        """
        _, _, H, W = x.shape
        
        with torch.amp.autocast('cuda'):  # ✅ AMP 自動管理精度        
            x_norm = to_4d(self.norm1(to_3d(x)), H, W) # 正規化
            x_windows = window_partition(x_norm, self.window_size) # 窗口切割
            x_windows = window_to_token(x_windows, self.window_size) # Token 合併
            
            # 直接對窗口內像素執行自注意力
            remsa_output = self.remsa(x_windows) # REMSA 計算
            remsa_output = token_to_window(remsa_output, H, W, self.window_size) # 正確還原形狀
            
            # 殘差連接
            x = x + remsa_output
            
            # LeFF 計算 + 殘差連接
            x = x + self.leff(to_4d(self.norm2(to_3d(x)), H, W))
        
        return x  
    
##########################################################################
# Image De-Raining Transformer (WTM >> STM) 名字怎麼都好可以再想想
class ImageDerainingTransformer(nn.Module):
    """ 
    Image De-Raining Transformer (WTM >> STM)
    """
    def __init__(self, dim, window_size, norm_type='WithBias'):
        super().__init__()
        self.wtm = WindowBasedTM(dim, window_size, norm_type)
        self.stm = SpaceBasedTM(dim, window_size, norm_type)

    def forward(self, x):
        """
        x: (B, C, H, W) - 影像特徵圖
        return: (B, C, H, W) - 經過 WTM 和 STM 處理的影像特徵圖
        """
        with torch.amp.autocast('cuda'):  # ✅ AMP 自動管理精度
            x = self.wtm(x)  # 先經過 WTM
            x = self.stm(x)  # 再經過 STM
        return x

"""Hybrid CNN-Transformer Feature Fusion for Single Image Deraining"""
##########################################################################
# degradation-aware mixture of experts (DaMoE)

class DegradationAwareMoE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=16):
        super(DegradationAwareMoE, self).__init__()
        
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
    
    def forward(self, x):
        b, _, _, _ = x.shape
        
        # 通道平均池化計算 z_c
        z_c = x.mean(dim=[2, 3])  # (B, C)
        
        # 使用 W1, W2 計算專家權重
        weights = self.W2(F.relu(self.W1(z_c)))  # (B, num_experts)
        weights = torch.softmax(weights, dim=1)  
        
        # 計算專家輸出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, num_experts, C, H, W)
        mixture = torch.sum(weights.view(b, -1, 1, 1, 1) * expert_outputs, dim=1)  # (B, C, H, W)
        
        # 1×1 卷積處理後再加上殘差連接
        output = self.final_conv(mixture) + x
        return output

"""Token statistics transformer: linear-time attention via variational rate reduction"""
##########################################################################
# ToST（Token Statistics Transformer） 版本的自注意力，取代傳統的 QK 相似性計算
class CausalSelfAttention_TSSA(nn.Module):

    def __init__(self, dim, num_heads = 8, block_size = 1024, dropout = 0.1, bias=False ):
        super().__init__()
        
        # query, key, value projections
        self.c_attn = nn.Linear(dim, dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = num_heads
        self.dim = dim
        self.dropout = dropout
        self.block_size = block_size
        self.temp = nn.Parameter(torch.ones((self.n_head, 1)))
        self.denom_bias = nn.Parameter(torch.zeros((self.n_head, block_size, 1)))
        
    def forward(self, x):
        """
        x: (B, N, C) - token 序列
        return: (B, N, C) - 經過 TSSA 處理的 token 序列
        """
        B, N, C = x.shape # batch size, sequence length, embedding dimensionality (dim)

        with torch.amp.autocast('cuda'):  # ✅ AMP 自動管理精度
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            w = self.c_attn(x).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            w_sq = w ** 2
            denom = (torch.cumsum(w_sq,dim=-2)).clamp_min(torch.finfo(torch.float32).eps) # cumulative sum
            w_normed = (w_sq / denom) + self.denom_bias[:,:N,:]
        
            # calculate attention weights
            tmp = torch.sum(w_normed, dim=-1)* self.temp
            Pi = F.softmax(tmp, dim=1) # B, nh, T
        
            # calculate attention
            dots = torch.cumsum(w_sq * Pi.unsqueeze(-1), dim=-2) / (Pi.cumsum(dim=-1) + torch.finfo(torch.float32).eps).unsqueeze(-1)
            attn = 1. / (1 + dots)
            attn = self.attn_dropout(attn)
        
            # apply attention weights and combine heads
            y = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)
            y = y.transpose(1, 2).contiguous().view(B, N, C) # re-assemble all head outputs side by side
            y = self.resid_dropout(self.c_proj(y))
            
        return y

##########################################################################
# ToST（Token Statistics Transformer）塊
class TokenStatisticsTransformer(nn.Module):

    def __init__(self, dim = 1024, norm_type='WithBias'):
        super().__init__()
        self.ln_1 = norms.Norm(dim, norm_type) # LayerNorm
        self.attn = CausalSelfAttention_TSSA(dim) # TSSA
        
        self.ln_2 = norms.Norm(dim, norm_type) # LayerNorm
        self.mlp = MultiLayerPerceptron(dim)
        eta = torch.finfo(torch.float32).eps
        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
    def forward(self, x):
        """
        x: (B, C, H, W) - 影像特徵圖
        return: (B, C, H, W) - 經過 ToST 處理的影像特徵圖
        """
        _, _, H, W = x.shape
        
        with torch.amp.autocast('cuda'):  # ✅ AMP 自動管理精度        
            x = x + self.gamma1.view(1, -1, 1, 1) *to_4d(self.attn(self.ln_1(to_3d(x))), H, W)
            x = x + self.gamma2.view(1, -1, 1, 1) *to_4d(self.mlp(self.ln_2(to_3d(x))), H, W)
        return x



"""A novel dual-stage progressive enhancement network for single image deraining"""
##########################################################################
# dilated dense residual block (DDRB)
class DDRB(nn.Module):
    """
    Dilated Dense Residual Block 
    Usage:
        self.ddrb = DDRB(in_channels=32, mid_channels=32, kernel=3, stride=1, d=[1, 2, 5], bias=False)
    """
    def __init__(self,
                 in_channels=32,
                 mid_channels=32,
                 kernel=3,
                 stride=1,
                 d=[1, 2, 5],
                 bias=False):
        super(DDRB, self).__init__()                
        self.convD1 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel, 
                          stride, padding=d[0], dilation=d[0], bias=bias),
                nn.ReLU(inplace=False),
                nn.Conv2d(mid_channels, mid_channels, kernel, 
                          stride, padding=d[0], dilation=d[0], bias=bias)
            ) # dilation=1
        self.convD2 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel, 
                          stride, padding=d[1], dilation=d[1], bias=bias),
                nn.ReLU(inplace=False),
                nn.Conv2d(mid_channels, mid_channels, kernel, 
                          stride, padding=d[1], dilation=d[1], bias=bias)
            ) # dilation=2
        self.convD3 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel, 
                          stride, padding=d[2], dilation=d[2], bias=bias),
                nn.ReLU(inplace=False),
                nn.Conv2d(mid_channels, mid_channels, kernel, 
                          stride, padding=d[2], dilation=d[2], bias=bias)
            ) # dilation=5
            
    def forward(self, x):
        """
        Args:
            x: input feature map
        Returns:
            enhanced feature map
        Usage:
            enhanced_feature = DDRB(input_feature)
        """
        # with torch.amp.autocast('cuda'):
        x1 = self.convD1(x)
        x2 = self.convD2(F.relu(x+x1))
        x3 = self.convD3(F.relu(x+x1+x2))
        output = F.relu(x+x1+x2+x3)
        return output
    
##########################################################################
# enhanced residual pixel-wise attention block (ERPAB)
class ERPAB(nn.Module):
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
            nn.Conv2d(in_channels, mid_channels, kernel, stride, 
                      padding=d[0], dilation=d[0], bias=bias),  # C32D1
            nn.Conv2d(in_channels, mid_channels, kernel, stride, 
                      padding=d[1], dilation=d[1], bias=bias),  # C32D2
            nn.Conv2d(in_channels, mid_channels, kernel, stride, 
                      padding=d[2], dilation=d[2], bias=bias),  # C32D5
        ])

        # self.experts = nn.ModuleList([
        # nn.AvgPool2d(3, stride=1, padding=1),  # 3×3 pooling
        # nn.Conv2d(in_channels, mid_channels, 1, stride, bias=bias),  # 1×1 conv
        # nn.Conv2d(in_channels, mid_channels, 3, stride, padding=1, bias=bias),  # 3×3 conv
        # nn.Conv2d(in_channels, mid_channels, 5, stride, padding=2, bias=bias),  # 5×5 conv
        # nn.Conv2d(in_channels, mid_channels, 7, stride, padding=3, bias=bias),  # 7×7 conv
        # nn.Conv2d(in_channels, mid_channels, 3, stride, padding=3, dilation=3, bias=bias),  # 3×3 dilated
        # nn.Conv2d(in_channels, mid_channels, 5, stride, padding=6, dilation=3, bias=bias),  # 5×5 dilated
        # nn.Conv2d(in_channels, mid_channels, 7, stride, padding=9, dilation=3, bias=bias)   # 7×7 dilated
        # ])
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(mid_channels * 8, in_channels, kernel_size=1, stride=stride, padding=0, bias=True),
            nn.ReLU(inplace=False)
        )

        self.pa = nn.Sequential(
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(1, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: input feature map
        Returns:
            enhanced feature map
        Usage:
            enhanced_feature = ERPAB(input_feature)
        """
        # with torch.amp.autocast('cuda'):
        input_ = x
        expert_outputs = torch.cat([expert(x) for expert in self.experts], dim=1)
        x1 = self.conv1(expert_outputs)
        pa = self.pa(x1)
        output = F.relu(x1 * pa + input_)  # 殘差連接
        return output
        #     x1 = F.relu(self.conv1(expert_outputs))
        #     attn_map = self.attn_map(x1)

        # return x + x1 * self.sigmoid(attn_map)

##########################################################################
# cross-stage feature interaction module (CFIM)
class CFIM(nn.Module):
    """
    Cross-Stage Feature Interaction Module
    Usage:
        self.cfim = CFIM(in_channels=32, norm_type = 'DyT' or 'WithBias' or 'BiasFree')
    """
    def __init__(self, in_channels, norm_type='WithBiasCNN'):
        super(CFIM, self).__init__()
        self.norm1 = norms.Norm(in_channels, norm_type)
        self.norm2 = norms.Norm(in_channels, norm_type)
        self.rsconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.rsconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.drconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.drconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
    
    def forward(self, r_net, dr_net):
        """
        Args:
            rs_net: rain streaks removal network intermediate output
            dr_net: details reconstruction network intermediate output
        Returns:
            to_rs_net: updated rain streaks removal network intermediate output
            to_dr_net: updated details reconstruction network intermediate output
        Usage:
            to_rs_net, to_dr_net = CFIM(r_net, dr_net)
        """     
        B, C, H, W = r_net.shape
        N = H * W # Flattened spatial dimension
        
        rs1 = self.rsconv1(self.norm1(r_net))  # (B, C, H, W)
        dr1 = self.drconv1(self.norm2(dr_net)) # (B, C, H, W)
        
        # flatten to (B, C, N)
        rs1_flat = rs1.view(B, C, N)
        dr1_flat = dr1.view(B, C, N)
        
        # attention map A: (B, C, C)
        A = torch.matmul(rs1_flat, dr1_flat.transpose(1, 2)) # (B, C, C)
        # A = torch.matmul(rs1, dr1)
        
        # second convs
        rs2_flat = self.rsconv2(r_net).view(B, C, N)  
        dr2_flat = self.drconv2(dr_net).view(B, C, N) 
        
        # apply mutual attention
        rs_side = torch.matmul(A, rs2_flat).view(B, C, H, W)
        dr_side = torch.matmul(A, dr2_flat).view(B, C, H, W)
        
        # residual update
        to_rs_net = dr_side + r_net
        to_dr_net = rs_side + dr_net

        return to_rs_net, to_dr_net

 
"""通用組件小工具堆放區"""
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

##########################################################################
# 影像轉token序列 (B, C, H, W) to (B, HW, C)
def to_3d(x):
    """Reshape from (B, C, H, W) to (B, HW, C)"""
    return rearrange(x, 'b c h w -> b (h w) c')

##########################################################################
# token序列轉影像 (B, HW, C) to (B, C, H, W)
def to_4d(x, h, w):
    """Reshape from (B, HW, C) to (B, C, H, W)"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

##########################################################################
# MLP (flaoat16)
class MultiLayerPerceptron(nn.Module):

    def __init__(self, dim, dropout=0.1, bias=True):
        super().__init__()
        self.c_fc    = nn.Linear(dim, 4*dim, bias=bias)
        self.c_proj  = nn.Linear(4*dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        with torch.amp.autocast('cuda'):  # ✅ AMP 自動管理精度
            x = self.c_fc(x)
            x = F.gelu(x)
            x = self.c_proj(x)
            x = self.dropout(x)
        return x