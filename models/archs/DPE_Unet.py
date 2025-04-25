#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
import models.archs.modules as modules #導入組件

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
class DPE_Unet(nn.Module):
    def __init__(self, 
        in_channels=3, 
        mid_channel=32,
        out_channels=3,
        kernel=3,
        stride=1,
        dilation_list=[1, 2, 5], 
        bias = False,
        num_blocks = [4,6,6,8],
        Norm_type = 'BiasFree',   ## Option: 'BiasFree' 'WithBias' 'DyT'
        ):
        super(DPE_Unet, self).__init__()
        
        self.patch_embed = nn.Conv2d(in_channels, mid_channel, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.encoder_level1 = nn.Sequential(
            *[modules.DDRB(mid_channel, mid_channel, kernel, 
                           stride, dilation_list, bias) for _ in range(num_blocks[0])])
        
        self.down1_2 = Downsample(mid_channel) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[modules.DDRB(mid_channel*2, mid_channel*2, kernel, 
                           stride, dilation_list, bias) for _ in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(mid_channel*2)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[modules.DDRB(mid_channel*4, mid_channel*4, kernel, 
                           stride, dilation_list, bias) for _ in range(num_blocks[2])])

        self.down3_4 = Downsample(int(mid_channel*4)) ## From Level 3 to Level 4
        
        self.latent1 = nn.Sequential(
            *[modules.DDRB(mid_channel*8, mid_channel*8, kernel, 
                           stride, dilation_list, bias) for _ in range(num_blocks[3]//2)])
        self.latent2 = nn.Sequential(
            *[modules.DDRB(mid_channel*8, mid_channel*8, kernel, 
                           stride, dilation_list, bias) for _ in range(num_blocks[3]//2)])
        
        self.up4_3 = Upsample(int(mid_channel*8)) ## From Level 4 to Level 3
        self.reduce_channel_level3 = nn.Conv2d(int(mid_channel*8), int(mid_channel*4), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[modules.DDRB(mid_channel*4, mid_channel*4, kernel, 
                           stride, dilation_list, bias) for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(int(mid_channel*4)) ## From Level 3 to Level 2
        self.reduce_channel_level2 = nn.Conv2d(int(mid_channel*4), int(mid_channel*2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[modules.DDRB(mid_channel*2, mid_channel*2, kernel, 
                           stride, dilation_list, bias) for _ in range(num_blocks[1])])
        
        self.up2_1 = nn.Sequential(nn.Conv2d(mid_channel*2, mid_channel*4, kernel_size=3, 
                                             stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))  ## From Level 2 to Level 1  (NO reduce channels)

        self.decoder_level1 = nn.Sequential(
            *[modules.DDRB(mid_channel*2, mid_channel*2, kernel, 
                           stride, dilation_list, bias) for _ in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(
            *[modules.DDRB(mid_channel*2, mid_channel*2, kernel, 
                           stride, dilation_list, bias) for _ in range(4)])
        
        ###########################
            
        self.output = nn.Conv2d(int(mid_channel*2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
    def forward(self, inp_img):
        # Removed manual conversion; let autocast handle precision.
        input_ = inp_img # (B, 3, H, W)
        enc_level1_inp = self.patch_embed(input_) # (B, C, H, W)
        enc_level1_out = self.encoder_level1(enc_level1_inp)
        
        # Downsample to level 2
        enc_level2_inp = self.down1_2(enc_level1_out) # (B, 2C, H/2, W/2)
        enc_level2_out = self.encoder_level2(enc_level2_inp) 
        
        # Downsample to level 3
        enc_level3_inp = self.down2_3(enc_level2_out) # (B, 4C, H/4, W/4)
        enc_level3_out = self.encoder_level3(enc_level3_inp) 
        
        # Downsample to level 4
        enc_level4_inp = self.down3_4(enc_level3_out) # (B, 8C, H/8, W/8)
        enc_level4_out = self.latent1(enc_level4_inp)
        # mid_result = enc_level4_out
        enc_level4_out = self.latent2(enc_level4_out)
        
        # Upsample to level 3
        dec_level3_inp = self.up4_3(enc_level4_out) # (B, 4C, H/4, W/4)
        dec_level3_inp = torch.cat((dec_level3_inp, enc_level3_out), dim=1) # (B, 8C, H/4, W/4)
        dec_level3_inp = self.reduce_channel_level3(dec_level3_inp) # (B, 4C, H/4, W/4)
        dec_level3_out = self.decoder_level3(dec_level3_inp)
        
        # Upsample to level 2
        dec_level2_inp = self.up3_2(dec_level3_out) # (B, 2C, H/2, W/2)
        dec_level2_inp = torch.cat((dec_level2_inp, enc_level2_out), dim=1) # (B, 4C, H/2, W/2)
        dec_level2_inp = self.reduce_channel_level2(dec_level2_inp) # (B, 2C, H/2, W/2)
        dec_level2_out = self.decoder_level2(dec_level2_inp)
        
        # Upsample to level 1
        dec_level1_inp = self.up2_1(dec_level2_out) # (B, C, H, W)
        dec_level1_inp = torch.cat((dec_level1_inp, enc_level1_out), dim=1) # (B, 2C, H, W)
        dec_level1_out = self.decoder_level1(dec_level1_inp)
        refinement = self.refinement(dec_level1_out)
        output = self.output(refinement) # (B, out_channels, H, W)
        
        # Final output
        output = output + input_  # Skip connection

        return output