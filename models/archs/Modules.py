import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import models.archs.Norms as Norms

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
        x = x1 * x2  # 閘控機制
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