import torch
from torchinfo import summary
from models.archs.GFCS_X_mod1 import GFCSNetwork

if __name__ == "__main__":
    model = GFCSNetwork(inp_channels=3, out_channels=3, dim=48)
    # model.half()  # Ensure model parameters use float16
    summary(model, input_size=(1, 3, 256, 256), device='cuda')
