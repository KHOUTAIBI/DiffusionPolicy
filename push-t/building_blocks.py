from convolution_embedding import *

# -------------------------------
# Convolution Unet # Unet architecture for policy diffusion 
# -------------------------------

class ConditionalRisidualBlock1D(nn.Module):
    """
    The down block in the Unet Architecture
    """
    def __init__(self, in_channels, 
                 out_channels,
                 conditional_dim, 
                 kernel_size = 3,  
                 num_groups = 8):
        super().__init__()

        # Blocks of the Unet
        self.block = nn.ModuleList([
            Convolution1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm_group=num_groups),
            Convolution1D(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm_group=num_groups),
        ])

        # Conditional channels, the ones seen in the paper
        condition_channels = out_channels * 2
        self.out_channels = out_channels

        self.condition_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(conditional_dim, condition_channels),
            nn.Unflatten(-1, (-1, 1)) # This is needed for action in 1D
        )
        
        self.residual_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, conditoning):
        """Forwarding block"""
        
        out = self.block[0](x)
        
        embedding = self.condition_encoder(conditoning)

        embedding = embedding.view(
            embedding.shape[0], 2, self.out_channels, 1
        )

        scale = embedding[:, 0, ...]
        bias = embedding[:, 1, ...]

        # Forawrding in linear
        out = scale * out + bias

        out = out + self.residual_conv(x)

        # out
        return out

# -------------------------------
# Downsample block
# -------------------------------

class DownSample1d(nn.Module):
    def __init__(self, dim,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

# -------------------------------
# Upsample block
# -------------------------------
class UpSampleBlock1d(nn.Module):
    def __init__(self, dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)   

        self.convtranspose = nn.ConvTranspose1d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.convtranspose(x)



# -------------------------------
# Unet Network
# -------------------------------

class ConditionalUnet1D(nn.Module):
    def __init__(self, 
                input_dim,
                global_cond_dim,
                diffusion_step_embed_dim=256,
                down_dims=[256,512,1024],
                kernel_size=5,
                n_groups=8,
                *args, 
                 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)   

        """
        gloabal_cond_dim : Used for conditioning, as seen in the paper of Diffusion
        down_dims: the dimensions of the down network of UNET
        input_dims : input dimension of observation space
        """

        self.all_dims = [input_dim] + list(down_dims)
        self.start_dim = self.all_dims[0]
        self.diffusion_step_embedding_dimension = diffusion_step_embed_dim
        self.global_cond_dim = global_cond_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # sinus time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.diffusion_step_embedding_dimension),
            nn.Linear(self.diffusion_step_embedding_dimension, self.diffusion_step_embedding_dimension * 4),
            nn.Mish(),
            nn.Linear(self.diffusion_step_embedding_dimension * 4, self.diffusion_step_embedding_dimension),
        ).to(device=self.device)

        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # BLOCKS OF UNET
        in_out = list(zip(self.all_dims[:-1], self.all_dims[1:]))
        print(in_out)
        

        mid_dim = self.all_dims[-1]
        
        # MID LAYER
        self.mid_modules = nn.ModuleList([
            ConditionalRisidualBlock1D(
                mid_dim, mid_dim, conditional_dim=cond_dim,
                kernel_size=kernel_size, num_groups=n_groups
            ),
            ConditionalRisidualBlock1D(
                mid_dim, mid_dim, conditional_dim=cond_dim,
                kernel_size=kernel_size, num_groups=n_groups
            ),
        ]).to(self.device)

        down_modules = []

        # DOWN LAYER
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            
            down_modules.append(nn.ModuleList([
                ConditionalRisidualBlock1D(
                    dim_in, dim_out, conditional_dim=cond_dim,
                    kernel_size=kernel_size, num_groups=n_groups),
                ConditionalRisidualBlock1D(
                    dim_out, dim_out, conditional_dim=cond_dim,
                    kernel_size=kernel_size, num_groups=n_groups),
                DownSample1d(dim_out) if not is_last else nn.Identity()
            ]).to(self.device))

        # up module of unet
        up_modules = []

        # UP BLOCK OF UNET
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalRisidualBlock1D(
                    dim_out*2, dim_in, conditional_dim=cond_dim,
                    kernel_size=kernel_size, num_groups=n_groups),
                ConditionalRisidualBlock1D(
                    dim_in, dim_in, conditional_dim=cond_dim,
                    kernel_size=kernel_size, num_groups=n_groups),
                UpSampleBlock1d(dim_in) if not is_last else nn.Identity()
            ]).to(self.device))

        self.final_conv = nn.Sequential(
            Convolution1D(self.start_dim, self.start_dim, kernel_size=kernel_size, norm_group=n_groups),
            nn.Conv1d(self.start_dim, input_dim, kernel_size=1),
        ).to(self.device)

        self.down_modules = down_modules
        self.up_modules = up_modules

        print(f"The number of params is: {np.sum([p.numel() for p in self.parameters()])}")


        # FORWARD
    def forward(self, sample, timesteps, global_cond = None):
            """
            x: (B,T,input_dim)
            timestep: (B,) or int, diffusion step
            global_cond: (B,global_cond_dim)
            output: (B,T,input_dim)
            """
            sample = sample.moveaxis(-1,-2)
           
            # time_embedding = timesteps.expand(sample.shape[0])
            timesteps = timesteps.expand(sample.shape[0])
            gloabal_features = self.time_embedding(timesteps)
            

            # Conditioning global
            if global_cond is not None:
                gloabal_features = torch.cat([
                    gloabal_features, global_cond 
                ], dim = -1) 
            
            x = sample
            
            skip = []

            for _, (resnet, resnet2, downsample) in enumerate(self.down_modules):
                
                x = resnet(x, gloabal_features)
                x = resnet2(x, gloabal_features)
                skip.append(x)
                x = downsample(x)

            for mid_module in self.mid_modules:
                x = mid_module(x, gloabal_features)

            for _, (resnet, resnet2, upsample) in enumerate(self.up_modules):
                x = torch.cat((x, skip.pop()), dim=1)
                x = resnet(x, gloabal_features)
                x = resnet2(x, gloabal_features)
                x = upsample(x)

     

            x = self.final_conv(x)
            x = x.moveaxis(-1, -2)
            # Unet output
            return x
            

        








