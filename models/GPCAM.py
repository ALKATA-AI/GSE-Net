# In the name of Allah, the most compassionate, the most merciful

###############################################################################

# Idea: GPCAM: Group Prime-kernel Channel Attention Module
# New Module in Speech Enhancement and Used in Conformer!

###############################################################################


import torch
import torch.nn.functional as F
#import numpy as np
import math
import glob
import os

from torch import nn
from einops.layers.torch import Rearrange
from utils import get_padding

###############################################################################

# Taken from "CAM: Channel Attention Module" article
# Reference: https://github.com/Peachypie98/CBAM/blob/main/cbam.py

###############################################################################

# --- CAM (Channel Attention Module) ---

class CAM1D(nn.Module):
    def __init__(self, channels, r):
        super(CAM1D, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(channels, channels//r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels//r, channels, bias=True)
        )

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        max_pool = F.adaptive_max_pool1d(x, 1)
        avg_pool = F.adaptive_avg_pool1d(x, 1)

        b, c, _ = x.size()
        max_out = self.linear(max_pool.view(b, c))
        avg_out = self.linear(avg_pool.view(b, c))

        output = max_out + avg_out
        output = output.view(b, c, 1)
        output = F.sigmoid(output) * x
        return output

            
class GPCAM(nn.Module):
    def __init__(self, dim, causal=False, expansion_factor=2, dropout=0.1):
    #def __init__(self, dim, causal=False, expansion_factor=4, dropout=0.0): 
        super().__init__()

        i_feats = expansion_factor * dim
        self.i_feats = i_feats
        self.causal = causal

        self.LN = nn.LayerNorm(dim)

        if self.causal:
            self.BN = nn.GroupNorm(num_groups=8, num_channels=i_feats)
        else:
            self.BN = nn.BatchNorm1d(i_feats)

        self.RA1 = Rearrange("b n c -> b c n")
        self.RA2 = Rearrange("b c n -> b n c")
        self.Dropout = nn.Dropout(dropout)

        self.CAM = CAM1D(channels=dim, r=16) #Idea

        # Point-wise Convolutions
        self.PWC1 = nn.Conv1d(dim, i_feats * 2, 1)
        self.PWC2 = nn.Conv1d(i_feats, dim, 1)

        # Activation Functions
        self.GLU = nn.GLU(dim=1)
        self.SiLU = nn.SiLU() #new

        # Multi-scale Dilated Convolutions

        # Depth-wise Dilated Convolution
        self.DWDC9 = nn.Conv1d(i_feats//4, i_feats//4, 31, padding=0, 
            dilation=2, groups=i_feats//4)
        #self.DWC9 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4)    

        self.DWDC7 = nn.Conv1d(i_feats//4, i_feats//4, 23, padding=0, 
            dilation=2, groups=i_feats//4)
        #self.DWC7 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4)    

        self.DWDC5 = nn.Conv1d(i_feats//4, i_feats//4, 11, padding=0, 
            dilation=3, groups=i_feats//4)
        #self.DWC5 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4)    

        self.DWDC3 = nn.Conv1d(i_feats//4, i_feats//4, 3, padding=0, 
            dilation=4, groups=i_feats//4)
        #self.DWC3 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4)

        # Point-wise Convolution
        self.PWC = nn.Conv1d(i_feats//4, i_feats//4, 1)

    @staticmethod 
    def apply_causal_conv(conv_layer, a, causal):
        kernel_size = conv_layer.kernel_size[0]
        dilation = conv_layer.dilation[0]
        padding = (kernel_size - 1) * dilation
        if causal:
            # Asymmetric padding for causal mode
            a_padded = nn.functional.pad(a, (padding, 0))
        else:
            # Symmetric padding for non-causal mode
            sym_padding = padding // 2
            a_padded = nn.functional.pad(a, (sym_padding, sym_padding))
        return conv_layer(a_padded)

    def forward(self, x):
        res = x
        x = self.LN(x)
        x = self.RA1(x)
        x = self.PWC1(x)
        x = self.GLU(x)

        #############################################################
        # GPGU: Group Prime-kernel Gated Unit (Idea!)
        #############################################################

        a1, a2, a3, a4 = torch.chunk(x, 4, dim=1)

        # DDFG: Depth-wise Dilated Fusion Gate
        d1 = self.apply_causal_conv(self.DWDC3, a1, self.causal)
        d2 = self.apply_causal_conv(self.DWDC5, a2, self.causal)
        d3 = self.apply_causal_conv(self.DWDC7, a3, self.causal)
        d4 = self.apply_causal_conv(self.DWDC9, a4, self.causal)

        #d1 = self.DWC3(a1)
        #d2 = self.DWC5(a2)
        #d3 = self.DWC7(a3)
        #d4 = self.DWC9(a4)

        x = torch.cat([
            self.PWC(d1) * d1,
            self.PWC(d2) * d2,
            self.PWC(d3) * d3,
            self.PWC(d4) * d4
        ], dim=1)

        #############################################################

        x = self.BN(x)
        x = self.SiLU(x)
        x = self.PWC2(x)
        x = self.CAM(x) #Idea!
        x = self.RA2(x)
        x = self.Dropout(x) 
        return x + res