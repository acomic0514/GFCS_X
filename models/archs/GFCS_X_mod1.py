#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# In[2]:


def to_3d(x):
    """Reshape from (B, C, H, W) to (B, HW, C)"""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """Reshape from (B, HW, C) to (B, C, H, W)"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# In[3]:


##########################################################################
# Dynamic Tanh (DyT)
class DyT(nn.Module):
    def __init__(self, normalized_shape: int, init_alpha=1.0):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)
        
##########################################################################
#Layer Norm

class BiasFreeLayerNorm(nn.Module):
    """LayerNorm without bias"""
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    """LayerNorm with bias"""
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

##########################################################################
#Norm 方式選擇(可使用DyT)
class Norm(nn.Module):
    """General LayerNorm wrapper supporting both BiasFree and WithBias variants"""
    def __init__(self, dim: int, norm_type: str = 'WithBias'):
        super().__init__()
        valid_types = {'WithBias', 'BiasFree', 'DyT'}
        if norm_type not in valid_types:
            raise ValueError("❌ Norm方法請選擇 'WithBias'、'BiasFree' 或 'DyT' 模式")
        
        self.norm_type = norm_type
        if norm_type == 'BiasFree':
            self.body = BiasFreeLayerNorm(dim)
        elif norm_type == 'DyT':
            self.body = DyT(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if self.norm_type == 'DyT':
            return to_4d(self.body(to_3d(x)), h, w)
        else :
            return to_4d(self.body(to_3d(x).float()).half(), h, w)



        


# In[4]:


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
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

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
        b, c, h, w = x.shape
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


# In[5]:


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
        x1 = F.gelu(x1.float()).half() # GELU 在 float32 下計算
        x = x1 * x2  # 閘控機制
        x = self.project_out(x)
        return x


# In[6]:


##########################################################################
# TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, Norm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = Norm(dim, Norm_type)
        self.MDTA = MDTA(dim, num_heads, bias)
        self.norm2 = Norm(dim, Norm_type)
        self.GDFN = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.MDTA(self.norm1(x))
        x = x + self.GDFN(self.norm2(x))

        return x


# In[7]:


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


# In[8]:


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# In[9]:


##########################################################################
##---------- Restormer改一-----------------------
class GFCS_X(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        Norm_type = 'BiasFree',   ## Option: 'BiasFree' 'WithBias' 'DyT'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(GFCS_X, self).__init__()
        

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, Norm_type=Norm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.apply(self._convert_to_fp16)  # ✅ 讓所有參數都轉為 float16
        
    def _convert_to_fp16(self, module):
        """ 遍歷所有層，將數據類型轉為 `float16`，但 LayerNorm 例外 """
        if isinstance(module, Norm):
            module.float()  # ✅ LayerNorm 仍然保持 `float32`
        else:
            module.half()  # 其他層轉為 `float16`
            
    def forward(self, inp_img):
        x = inp_img.half() # ✅ 確保輸入是 float16

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1


# In[10]:


##########################################################################
## L1 Loss
class L1Loss(nn.Module):
    """L1 Loss (Mean Absolute Error) with optional weight.
    Args:
        loss_weight (float): L1 Loss 的加權係數，默認為 1.0。
        reduction (str): 損失的縮減方式，可選 'none' | 'mean' | 'sum'，默認 'mean'。
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        assert reduction in ['none', 'mean', 'sum'], f'Invalid reduction mode: {reduction}'
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None):
        """
        Args:
            pred (Tensor): 預測圖像，形狀為 (N, C, H, W)。
            target (Tensor): 真實圖像，形狀為 (N, C, H, W)。
            weight (Tensor, optional): 權重張量，形狀為 (N, C, H, W)，默認 None。

        Returns:
            Tensor: 計算後的 L1 Loss。
        """
        pred, target = pred.half(), target.half() # 確保輸入是 float16
        loss = F.l1_loss(pred, target, reduction='none')  # 計算逐像素 L1 Loss
        
        # 如果提供了權重，則進行加權
        if weight is not None:
            assert weight.shape == loss.shape, "Weight tensor must have the same shape as loss tensor."
            loss = loss * weightt.half() # 確保權重也轉為 float16

        # 根據 reduction 進行縮減
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        return self.loss_weight * loss  # 'none' 模式下直接返回逐像素 Loss


# In[12]:


"""Loss測試腳本

# ✅ 設定測試數據尺寸
batch_size = 2
channels = 3
height = 128
width = 128

# ✅ 生成隨機 `float16` 輸入
pred = torch.randn((batch_size, channels, height, width), dtype=torch.float16)
target = torch.randn((batch_size, channels, height, width), dtype=torch.float16)

# ✅ 初始化 L1Loss
criterion = L1Loss()

# ✅ 計算 Loss
loss = criterion(pred, target)

# ✅ 打印結果
print(f"Loss Shape: {loss.shape}")
print(f"Loss Data:\n{loss.cpu().numpy() if loss.numel() <= 10 else 'Shape too large to print'}")
"""


# In[16]:


#"""網路本體測試腳本

# ✅ 設定測試數據尺寸
batch_size = 2
channels = 3
height = 64
width = 64

# ✅ 生成隨機 `float16` 輸入
input_img = torch.randn((batch_size, channels, height, width), dtype=torch.float16)
target_img = torch.randn((batch_size, channels, height, width), dtype=torch.float16)

# ✅ 初始化模型（不在 CUDA 上）
model = GFCS_X(inp_channels=3, out_channels=3, dim=48)
model.half()  # ✅ 確保模型使用 `float16`

# ✅ 初始化 L1Loss
criterion = L1Loss()

# ✅ 前向傳播
output = model(input_img)

# ✅ 計算 Loss
loss = criterion(output, target_img)

# ✅ 打印結果
print(f"Output Shape: {output.shape}")
print(f"Loss Shape: {loss.shape}")
print(f"Loss Value: {loss.item()}")

#""""


# In[ ]:




