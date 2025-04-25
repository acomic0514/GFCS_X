import torch.nn as nn
from models.archs.modules import TransformerBlock

##########################################################################
# Trans*10
class Trans10(nn.Module):
    def __init__(self,
                 in_channels=3,
                 mid_channels=32,
                 bias=False):
        super(Trans10, self).__init__()

        # Initial feature transformation
        self.inconv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 
                      kernel_size=1, padding=0, bias=bias),
            nn.ReLU(inplace=False),
        )
        self.outconv1 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, 
                      kernel_size=1, padding=0, bias=bias),
            # nn.ReLU(inplace=False),
            # nn.Tanh(),  # Apply Tanh activation
        )
               
        # Network Modules
        self.trans10 = nn.Sequential(
            *[TransformerBlock(dim = 32, num_heads = 4, 
                               ffn_expansion_factor = 2.66, bias = False, 
                               Norm_type = 'WithBiasCNN') for _ in range(10)])        

    def forward(self, x):
        input_ = x
        
        x = self.inconv1(x)
        rs1 = self.trans10(x)
        x = self.outconv1(rs1)
        x_final = x + input_
        
        return x_final
    