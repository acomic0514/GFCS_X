#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import models.archs.modules as modules #導入組件
from models.archs.norms import Norm  #導入組件


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, 
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, 
                                            kernel_size=3, stride=1, 
                                            padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, 
                                            kernel_size=3, stride=1, 
                                            padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



##########################################################################
##---------- Restormer改一-----------------------
class GFCS_X(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = None, 
        num_refinement_blocks = 4,
        heads = None,
        ffn_expansion_factor = 2.66,
        bias = False,
        Norm_type = 'BiasFree',   ## Option: 'BiasFree' 'WithBias' 'DyT'
        dual_pixel_task = False   ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):
        self.num_blocks = num_blocks if num_blocks is not None else [4,6,6,8]
        self.heads = heads if heads is not None else [1,2,4,8]

        super(GFCS_X, self).__init__()
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[modules.TransformerBlock(dim=dim, num_heads=heads[0], 
                                                                       ffn_expansion_factor=ffn_expansion_factor, 
                                                                       bias=bias, Norm_type=Norm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[modules.TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], 
                                                                       ffn_expansion_factor=ffn_expansion_factor, 
                                                                       bias=bias, Norm_type=Norm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[modules.TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], 
                                                                       ffn_expansion_factor=ffn_expansion_factor, 
                                                                       bias=bias, Norm_type=Norm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[modules.TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], 
                                                               ffn_expansion_factor=ffn_expansion_factor, 
                                                               bias=bias, Norm_type=Norm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[modules.TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], 
                                                                       ffn_expansion_factor=ffn_expansion_factor, 
                                                                       bias=bias, Norm_type=Norm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[modules.TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], 
                                                                       ffn_expansion_factor=ffn_expansion_factor, 
                                                                       bias=bias, Norm_type=Norm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[modules.TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], 
                                                                       ffn_expansion_factor=ffn_expansion_factor, 
                                                                       bias=bias, Norm_type=Norm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[modules.TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], 
                                                                   ffn_expansion_factor=ffn_expansion_factor, 
                                                                   bias=bias, Norm_type=Norm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.apply(self._convert_to_fp16)  # ✅ 讓所有參數都轉為 float16
        
    def _convert_to_fp16(self, module):
        """ 遍歷所有層，將數據類型轉為 `float16`，但 LayerNorm 例外 """
        if isinstance(module, Norm):  # LayerNorm 保持 `float32`
            module.float()  # ✅ LayerNorm 仍然保持 `float32`
        else:
            module.half()  # 其他層轉為 `float16`
            
    def forward(self, inp_img):
        inp_img = inp_img.half() # ✅ 確保輸入是 float16

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




