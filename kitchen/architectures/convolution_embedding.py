import torch
import torch.nn as nn
import numpy as np


# -------------------------------
# Convolution  # This is exactly the block used in Unet
# -------------------------------   

class Convolution1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1, norm_group = 8, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)  
        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, stride = stride, padding = kernel_size // 2),
            # Group norming
            nn.GroupNorm(num_groups=norm_group, num_channels=out_channels),
            # Surprisignly used in a lot of recent papers
            nn.Mish(),
        )

    def forward(self, x):
        """
        The convolution block, which is the building block of Unet 
        """
        # Foearding the x
        
        return self.convolution(x)



# -------------------------------
# Sinusoidal Time Embedding # This is used in Optimal transport 
# -------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)   
        self.dim = dim

    def forward(self, time):
        """
        The time embedding as seen in the transformer architecture
        """
        device = time.device
        half_dim = self.dim // 2
        
        emb_scale = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device) * - emb_scale)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # Embedding
        return emb