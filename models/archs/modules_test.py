import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import models.archs.norms as norms
import numpy as np
import pandas as pd
import os

##########################################################################
# dilated dense residual block (DDRB)
class DDRB_Traceable(nn.Module):
    """
    Dilated Dense Residual Block 
    Usage:
        self.ddrb = DDRB(in_channels=32, mid_channels=32, kernel=3, stride=1, d=[1, 2, 5], bias=False)
    """
    def __init__(self,
                 in_channels=4,
                 mid_channels=4,
                 kernel=3,
                 stride=1,
                 d=[1, 2, 5],
                 bias=False):
        super(DDRB_Traceable, self).__init__()   
                
        self.convD1 = nn.Sequential(
                TraceLayer(nn.Conv2d(in_channels, mid_channels, kernel, 
                          stride, padding=d[0], dilation=d[0], bias=bias)),
                TraceLayer(nn.ReLU(inplace=False)),
                TraceLayer(nn.Conv2d(mid_channels, mid_channels, kernel, 
                          stride, padding=d[0], dilation=d[0], bias=bias))
            ) # dilation=1
        self.convD2 = nn.Sequential(
                TraceLayer(nn.Conv2d(in_channels, mid_channels, kernel, 
                          stride, padding=d[1], dilation=d[1], bias=bias)),
                TraceLayer(nn.ReLU(inplace=False)),
                TraceLayer(nn.Conv2d(mid_channels, mid_channels, kernel, 
                          stride, padding=d[1], dilation=d[1], bias=bias))
            ) # dilation=2
        self.convD3 = nn.Sequential(
                TraceLayer(nn.Conv2d(in_channels, mid_channels, kernel, 
                          stride, padding=d[2], dilation=d[2], bias=bias)),
                TraceLayer(nn.ReLU(inplace=False)),
                TraceLayer(nn.Conv2d(mid_channels, mid_channels, kernel, 
                          stride, padding=d[2], dilation=d[2], bias=bias))
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
        
        features = {
            "input": x.detach().cpu().numpy(),
            **extract_all_trace_data(self.convD1, "convD1"),
            **extract_all_trace_data(self.convD2, "convD2"),
            **extract_all_trace_data(self.convD3, "convD3"),
            "output": output.detach().cpu().numpy()
            }
        
        return output, features
    
##########################################################################
# Trace Layer
# This is a simple wrapper to trace the output of a layer
# during the forward pass. It stores the output in a numpy array.
# Usage:
#     trace_layer = TraceLayer(layer)
#     model = nn.Sequential(trace_layer, ...)
#     output = model(input)
#     print(trace_layer.output)  # Access the output after the forward pass
class TraceLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer 
        self.output = None
        self.weight = None    
        self.bias = None

    def forward(self, x):
        out = self.layer(x)
        self.output = out.detach().cpu().numpy()
        if isinstance(self.layer, nn.Conv2d):
            self.weight = self.layer.weight.detach().cpu().numpy()
            self.bias = self.layer.bias.detach().cpu().numpy() if self.layer.bias is not None else None
        
        
        return out
    
##########################################################################
# Trace Sequential
# This function extracts the outputs from a sequential model
# and returns them as a dictionary.
# Usage:
#     seq = nn.Sequential(...)
#     outputs = extract_outputs_from_sequential(seq, 'prefix')
#     print(outputs)  # Access the outputs of each layer
#     # Outputs will be in the form of {'prefix_layer0': ..., 'prefix_layer1': ...}
#     # where prefix is the prefix you provided
#     # and layer0, layer1 are the indices of the layers in the sequential model.
#     # Note: This function assumes that the layers in the sequential model
#           are instances of TraceLayer.
def extract_all_trace_data(seq, prefix):
    results = {}
    for i, layer in enumerate(seq):
        if hasattr(layer, 'output') and layer.output is not None:
            results[f"{prefix}_layer{i}_output"] = layer.output
        if hasattr(layer, 'weight') and layer.weight is not None:
            results[f"{prefix}_layer{i}_weight"] = layer.weight
        if hasattr(layer, 'bias') and layer.bias is not None:
            results[f"{prefix}_layer{i}_bias"] = layer.bias
    return results

