#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
import models.archs.modules as modules #導入組件
from models.archs.norms import Norm  #導入組件


def make_transformer_stack(**kwargs):
    return nn.Sequential(*[modules.TransformerBlock(**kwargs) for _ in range(kwargs['num_blocks'])])

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
class GFCSNetwork(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [1,1,1,1], 
        num_refinement_blocks = 1,
        heads = [1,1,1,1],
        ffn_expansion_factor = 2.66,
        bias = False,
        Norm_type = 'BiasFree',   ## Option: 'BiasFree' 'WithBias' 'DyT'
        dual_pixel_task = False   ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):
        super(GFCSNetwork, self).__init__()
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        kwargs = {
            'ffn_expansion_factor': ffn_expansion_factor, 
            'bias': bias, 
            'Norm_type': Norm_type
        }
        self.num_blocks = num_blocks
        
        self.encoder_level1 = make_transformer_stack(dim=int(dim), num_blocks=num_blocks[0], num_heads=heads[0], **kwargs)
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = make_transformer_stack(dim=int(dim*2**1), num_blocks=num_blocks[1], num_heads=heads[1], **kwargs)
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = make_transformer_stack(dim=int(dim*2**2), num_blocks=num_blocks[2], num_heads=heads[2], **kwargs) 

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = make_transformer_stack(dim=int(dim*2**3), num_blocks=num_blocks[3], num_heads=heads[3], **kwargs)
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = make_transformer_stack(dim=int(dim*2**2), num_blocks=num_blocks[2], num_heads=heads[2], **kwargs)


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = make_transformer_stack(dim=int(dim*2**1), num_blocks=num_blocks[1], num_heads=heads[1], **kwargs)
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = make_transformer_stack(dim=int(dim*2**1), num_blocks=num_blocks[0], num_heads=heads[0], **kwargs)
        
        self.refinement = make_transformer_stack(dim=int(dim*2**1), num_blocks=num_refinement_blocks, num_heads=heads[0], **kwargs)
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
        # Removed manual fp16 conversion; using AMP autocast for mixed precision training
        
    def forward(self, inp_img):
        # Removed manual conversion; let autocast handle precision.

        inp_enc = self.patch_embed(inp_img) # inp_enc_level1
        out_enc_level1 = checkpoint_sequential(self.encoder_level1, self.num_blocks[0], inp_enc, use_reentrant=False)
        # out_enc_level1 = self.encoder_level1(inp_enc)
        out_enc_level2 = checkpoint_sequential(self.encoder_level2, self.num_blocks[1], self.down1_2(out_enc_level1), use_reentrant=False)
        # out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        out_enc_level3 = checkpoint_sequential(self.encoder_level3, self.num_blocks[2], self.down2_3(out_enc_level2), use_reentrant=False)
        # out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2)) 

        x = self.down3_4(out_enc_level3) # inp_enc_level4
        x = checkpoint_sequential(self.latent, self.num_blocks[3], x, use_reentrant=False)
        # x = self.latent(x) # latent
                        
        x = self.up4_3(x) # inp_dec_level3 
        x = torch.cat([x, out_enc_level3], 1)
        x = self.reduce_chan_level3(x)
        x = checkpoint_sequential(self.decoder_level3, self.num_blocks[2], x, use_reentrant=False)
        # x = self.decoder_level3(x) # out_dec_level3

        x = self.up3_2(x) # inp_dec_level2
        x = torch.cat([x, out_enc_level2], 1)
        x = self.reduce_chan_level2(x)
        x = checkpoint_sequential(self.decoder_level2, self.num_blocks[1], x, use_reentrant=False)
        # x = self.decoder_level2(x) # out_dec_level2

        x = self.up2_1(x) # inp_dec_level1
        x = torch.cat([x, out_enc_level1], 1)
        x = checkpoint_sequential(self.decoder_level1, self.num_blocks[0], x, use_reentrant=False)
        # x = self.decoder_level1(x) # out_dec_level1
        
        x = self.refinement(x)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            x = x + self.skip_conv(inp_enc)
            x = self.output(x)
        ###########################
        else:
            x = self.output(x) + inp_img


        return x




