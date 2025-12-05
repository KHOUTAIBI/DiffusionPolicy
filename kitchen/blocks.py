import torch
from convolution_embedding import *
import torch.nn as nn
import numpy as np

# -------------------------------
# Conditional Residual Block (fixed logic)
# -------------------------------

class ConditionalResidualBlock1D(nn.Module):
    """
    Conditional Residual Block with FiLM modulation.
    Matches the logic of the first version but keeps your naming and structure.
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 conditional_dim, 
                 kernel_size=3,  
                 num_groups=8):
        
        super().__init__()

        # Two convolutional layers with normalization + activation
        self.blocks = nn.ModuleList([
            Convolution1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm_group=num_groups),
            Convolution1D(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm_group=num_groups),
        ])

        # FiLM conditioning (scale + bias per channel)
        condition_channels = out_channels * 2
        self.out_channels = out_channels

        self.condition_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(conditional_dim, condition_channels),
            nn.Unflatten(-1, (-1, 1))  # (B, cond_channels) → (B, cond_channels, 1)
        )

        # 1x1 conv if in/out channels differ for residual path
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()


    def forward(self, x, conditioning):
        """
        x: [B, in_channels, T]
        conditioning: [B, conditional_dim]
        """
        out = self.blocks[0](x)

        embedding = self.condition_encoder(conditioning)
        embedding = embedding.view(embedding.shape[0], 2, self.out_channels, 1)


        # Here we consieder observation horizon of 2 ? 
        # TODO Check the correctness of above statement
        scale = embedding[:, 0, ...]
        bias = embedding[:, 1, ...]

        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)

        return out


# -------------------------------
# Downsample and Upsample Blocks
# -------------------------------

class DownSample1d(nn.Module):

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class UpSampleBlock1d(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convtranspose = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.convtranspose(x)


# -------------------------------
# Conditional UNet 1D (refined)
# -------------------------------

class ConditionalUnet1D(nn.Module):
    def __init__(self, 
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=5,
                 n_groups=8,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.all_dims = [input_dim] + list(down_dims)
        self.start_dim = down_dims[0]

        self.diffusion_step_embedding_dimension = diffusion_step_embed_dim
        self.global_cond_dim = global_cond_dim

        # Time embedding (sinusoidal + MLP)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.diffusion_step_embedding_dimension),
            nn.Linear(self.diffusion_step_embedding_dimension, self.diffusion_step_embedding_dimension * 4),
            nn.Mish(),
            nn.Linear(self.diffusion_step_embedding_dimension * 4, self.diffusion_step_embedding_dimension),
        )

        cond_dim = diffusion_step_embed_dim + global_cond_dim
        print(f"the condition dim is: {cond_dim}")

        in_out = list(zip(self.all_dims[:-1], self.all_dims[1:]))
        mid_dim = self.all_dims[-1]

        # Middle (bottleneck) blocks
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, conditional_dim=cond_dim, kernel_size=kernel_size, num_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, conditional_dim=cond_dim, kernel_size=kernel_size, num_groups=n_groups),
        ])

        # Downsampling path
        self.down_modules = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
        
            is_last = ind == len(in_out) - 1
        
            self.down_modules.append(nn.ModuleList([
        
                ConditionalResidualBlock1D(dim_in, dim_out, conditional_dim=cond_dim, kernel_size=kernel_size, num_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, conditional_dim=cond_dim, kernel_size=kernel_size, num_groups=n_groups),
                DownSample1d(dim_out) if not is_last else nn.Identity()
        
            ]))

        # Upsampling path (mirror)
        self.up_modules = nn.ModuleList([])
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
        
            is_last = ind == len(in_out) - 1
        
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, conditional_dim=cond_dim, kernel_size=kernel_size, num_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, conditional_dim=cond_dim, kernel_size=kernel_size, num_groups=n_groups),
                UpSampleBlock1d(dim_in) if not is_last else nn.Identity()
        
            ]))

        # Final convolution
        self.final_conv = nn.Sequential(
        
            Convolution1D(self.start_dim, self.start_dim, kernel_size=kernel_size, norm_group=n_groups),
            nn.Conv1d(self.start_dim, input_dim, kernel_size=1),
        
        )

        print(f"The number of params is: {np.sum([p.numel() for p in self.parameters()])}")

    def forward(self, sample, timesteps, global_cond=None):
        """
        sample: (B, T, input_dim)
        timesteps: (B,) or scalar
        global_cond: (B, global_cond_dim)
        """
        x = sample.moveaxis(-1, -2)  # (B, C, T)

        # Time embedding
        
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=x.device)
        
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(x.device)
        
        # time embedding and global_feat    
        timesteps = timesteps.expand(x.shape[0])
        global_features = self.time_embedding(timesteps)


        # Combine time + global conditioning
        if global_cond is not None:
            global_features = torch.cat([global_features, global_cond], dim = -1)

        skip_connections = []

        # Down path
        for resnet1, resnet2, downsample in self.down_modules: # type: ignore

            x = resnet1(x, global_features)
            x = resnet2(x, global_features)
            skip_connections.append(x)
            x = downsample(x)

        # Middle
        for mid in self.mid_modules:
            x = mid(x, global_features)

        # Up path
        for resnet1, resnet2, upsample in self.up_modules: # type: ignore
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = resnet1(x, global_features)
            x = resnet2(x, global_features)
            x = upsample(x)

        # Final conv
        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)  # (B, T, input_dim)
        return x

        








