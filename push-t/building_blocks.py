from convolution_embedding import *

# -------------------------------
# Convolution Unet # Unet architecture for policy diffusion 
# -------------------------------

class DownBlock(nn.Module):
    """
    The down block in the Unet Architecture
    """
    def __init__(self, in_channels, 
                 out_channels, 
                 embedding_dim,
                 kernel_size = 3, 
                 down_sample=True, 
                 num_heads=4, 
                 num_layers=1):
        super().__init__()

        self.num_layers = num_layers
        self.down_sample = down_sample

        self.unet_first_convolution = nn.ModuleList([
            nn.Sequential(
                Convolution1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
            ) for i in range(num_layers)
        ])

        self.time_embedding_layers = nn.ModuleList([
            nn.Sequential(
                nn.Mish(),
                nn.Linear(embedding_dim, out_channels)
            ) for _ in range(num_layers)
        ])

        self.unet_second_convolution = nn.ModuleList([
            nn.Sequential(
                Convolution1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
            ) for _ in range(num_layers)
        ])

        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
        ]) # batch first always in Attention, remember the course on RNN

        self.residual_input_conv = nn.ModuleList([
            Convolution1D(in_channels if i == 0 else out_channels ,out_channels, kernel_size = 3, padding = 0)
            for i in range(num_layers)
        ])

        self.down_sample_conv = Convolution1D(out_channels, out_channels, kernel_size = 4, stride=1, padding=1) if down_sample else nn.Identity()

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
    """
    The Mid (choking) block of the Unet Architecture
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.unet_conv_first = nn.ModuleList([
            nn.Sequential(
                Convolution1D(in_channels if i == 0 else out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
            ) for i in range(num_layers + 1)
        ])

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(nn.Mish(), 
                          nn.Linear(t_emb_dim, out_channels)
            )
            for _ in range(num_layers + 1)
        ])

        self.unet_conv_second = nn.ModuleList([
            nn.Sequential(
                Convolution1D(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
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
            Convolution1D(in_channels if i == 0 else out_channels, out_channels, kernel_size = 1, padding=0)
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
    """
    The Upblock of the Unet Architecture 
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1):
        
        super().__init__()

        self.num_layers = num_layers
        self.up_sample = up_sample

        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2 , kernel_size=4, stride=2, padding=1) if up_sample else nn.Identity() # this is to verofy ? kernel = 4 ?

        self.unet_conv_first = nn.ModuleList([
            nn.Sequential(
                Convolution1D(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ) for i in range(num_layers)
        ])

        # Embedding layer
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
            for _ in range(num_layers)
        ])

        self.unet_conv_second = nn.ModuleList([
            nn.Sequential(
                Convolution1D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
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
            Convolution1D(in_channels if i == 0 else out_channels, out_channels, kernel_size=1, padding=0)
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


