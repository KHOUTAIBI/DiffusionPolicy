import torch
import torch.nn as nn
import numpy as np


# -------------------------------
# Sinusoidal Time Embedding
# -------------------------------
def time_embedding(time_steps, embedding_dim):
    """
    Sinusoidal time-step embeddings as in the DDPM paper.
    Args:
        time_steps: (B,) tensor of timesteps
        embedding_dim: int, embedding dimension
    Returns:
        (B, embedding_dim)
    """
    device = time_steps.device
    half_dim = embedding_dim // 2
    emb_scale = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
    emb = time_steps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


# -------------------------------
# DownBlock
# -------------------------------
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, down_sample=True, num_heads=4, num_layers=1):
        super().__init__()

        self.num_layers = num_layers
        self.down_sample = down_sample

        self.unet_first_convolution = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size = 3, padding=1)
            ) for i in range(num_layers)
        ])

        self.time_embedding_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(embedding_dim, out_channels)
            ) for _ in range(num_layers)
        ])

        self.unet_second_convolution = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding=1)
            ) for _ in range(num_layers)
        ])

        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
        ]) # batch first always in Attention, remember the course on RNN

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
            for i in range(num_layers)
        ])

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()

    def forward(self, x, embedding):
        
        output = x

        for i in range(self.num_layers):
            unet_input = output
            output = self.unet_first_convolution[i](output)
            output = output + self.time_embedding_layers[i](embedding)[:, :, None, None]
            output = self.unet_second_convolution[i](output)
            output = output + self.residual_input_conv[i](unet_input)

            # Attention
            attn_input = self.attention_norms[i](output)
            B, C, H, W = attn_input.shape
            attn_input = attn_input.view(B, C, H * W).transpose(1, 2)
            attn_out, _ = self.attentions[i](attn_input, attn_input, attn_input)
            attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
            output = output + attn_out

        down = self.down_sample_conv(output)
        return down # output is skip connection


# -------------------------------
# MidBlock
# -------------------------------
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.unet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ) for i in range(num_layers + 1)
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(nn.ReLU(), nn.Linear(t_emb_dim, out_channels))
            for _ in range(num_layers + 1)
        ])

        self.unet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
            ) for _ in range(num_layers + 1)
        ])

        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size = 1)
            for i in range(num_layers + 1)
        ])

    def forward(self, x, t_emb):
        
        out = x
        # First block
        unet_input = out
        out = self.unet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.unet_conv_second[0](out)
        out = out + self.residual_input_conv[0](unet_input)

        # Attention + more convs / Can also use the Conv1 !
        for i in range(self.num_layers):

            attn_input = self.attention_norms[i](out)
            B, C, H, W = attn_input.shape
            attn_input = attn_input.view(B, C, H * W).transpose(1, 2)
            attn_out, _ = self.attentions[i](attn_input, attn_input, attn_input)
            attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
            out = out + attn_out

            unet_input = out
            out = self.unet_conv_first[i + 1](out)
            out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.unet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](unet_input)

        return out


# -------------------------------
# UpBlock
# -------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1):
        
        super().__init__()

        self.num_layers = num_layers
        self.up_sample = up_sample

        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2 , kernel_size=4, stride=2, padding=1) if up_sample else nn.Identity() # this is to verofy ? kernel = 4 ?

        self.unet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ) for i in range(num_layers)
        ])

        # Embedding layer
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
            for _ in range(num_layers)
        ])

        self.unet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),  
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ) for _ in range(num_layers)
        ])

        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 1)
            for i in range(num_layers)
        ])

    def forward(self, x, skip, t_emb):
        # print(x.shape)
        x = self.up_sample_conv(x)
        # print(x.shape, skip.shape)
        x = torch.cat([x, skip], dim=1)
        
        out = x
        for i in range(self.num_layers):
            unet_input = out
            out = self.unet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.unet_conv_second[i](out)
            out = out + self.residual_input_conv[i](unet_input)

            attn_input = self.attention_norms[i](out)
            B, C, H, W = attn_input.shape
            attn_input = attn_input.view(B, C, H * W).transpose(1, 2)
            attn_out, _ = self.attentions[i](attn_input, attn_input, attn_input)
            attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
            out = out + attn_out

        return out
