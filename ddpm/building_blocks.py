import torch
import torch.nn as nn
import numpy as np
import scipy as sp

# Here we build the Downblok of the UNET
class DownBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, embedding_dim, down_sample = True, num_heads = 4, num_layers = 1,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.down_sample = down_sample
        self.num_heads = num_heads # number of heads in the block
        self.num_layers = num_layers # This here is the number of layers in the Down(ing) block of Unet 

        # the first convolution later
        self.unet_first_convolution = nn.ModuleList(
            [nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels = in_channels if i == 0 else out_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
            ) for i in range(self.num_layers)]
        )

        # Embedding channel
        self.time_embedding_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=self.embedding_dim, out_features=out_channels)
            ) for _ in range(self.num_layers)
        ])

        # Second unet convolution layer
        self.unet_second_convolution = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
            ) for _ in range(self.num_layers)
        ])

        # Attention layer
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, out_channels)
             for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )


        # Final conv layer and the downsampling 
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        # Downsampling which will be put to the Middle block
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if self.down_sample else nn.Identity()



    def forward(self, x, embedding):
        """
        Forwarding in the Down block, we will get a downsampled image
        """

        output = x

        for i in range(self.num_layers):

            unet_input = output
            output = self.unet_first_convolution[i](output)
            output = output + self.time_embedding_layers[i](embedding)[:, :, None, None]
            output = self.unet_second_convolution[i](output)
            output = output + self.residual_input_conv[i](output)

            B, C, H, W = output.size()
            in_attention = output.reshape(B, C, H * W)
            in_attention = self.attention_norms[i](in_attention)
            in_attention = torch.transpose(in_attention, 1, 2)
            out_attention, _ = self.attentions[i](in_attention, in_attention, in_attention)
            out_attention = out_attention.transpose(1, 2).reshape(B, C, H, W)
            out = out + out_attention

        # Forwarding in the layer of DownBlock
        out = self.down_sample_conv(out)
        return out


# Here we are building the Middle block of the UNET
class MidBlock(nn.Module):

    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.ReLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers+1)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers + 1)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers+1)
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers+1)
            ]
        )
    
    def forward(self, x, t_emb):
        out = x
        
        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):
            
            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
            
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i+1](out)
            out = out + self.t_emb_layers[i+1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_input)
        
        return out


# Here we build the Upblock of the UNET, the final layer that upsamples
class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers)
        ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(num_layers)
            ]
        )
        
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

        return out


def time_embedding(time_steps, embedding_dim):
    """
    This function adds the time embedding conditioning featured in Transformer architecture by concatenating, see paper
    Build sinusoidal embeddings.
    """
    assert embedding_dim % 2 == 0
    
    half_dim = embedding_dim // 2
    embedding = 1000 * torch.arange(0, half_dim, dtype = torch.float32) 

    embedding = embedding[:, None].repeat(1, half_dim)
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim = - 1)

    # This here returns the embedding using cos sin method in transformers
    return embedding