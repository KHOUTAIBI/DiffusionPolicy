import torch
import torch.nn as nn
from building_blocks import *   # assumes DownBlock, MidBlock, UpBlock, time_embedding are defined here


class Unet(nn.Module):
    """
    U-Net for DDPM. Expects building blocks (DownBlock, MidBlock, UpBlock, time_embedding)
    to be provided in building_blocks.py and to follow the interfaces used here.
    model_config is a dict with keys:
      - im_channels
      - down_channels (list)
      - mid_channels (list)
      - time_emb_dim
      - down_sample (list of bool length len(down_channels)-1)
      - num_down_layers, num_mid_layers, num_up_layers
    """
    def __init__(self, model_config):
        super().__init__()

        im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']

        # basic sanity checks (keep as you had them)
        assert self.mid_channels[0] == self.down_channels[-1], "mid[0] must equal down[-1]"
        assert self.mid_channels[-1] == self.down_channels[-2], "mid[-1] must equal down[-2]"
        assert len(self.down_sample) == len(self.down_channels) - 1

        # projection for time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        # reversed up-sample flags (not strictly necessary, kept for parity)
        self.up_sample = list(reversed(self.down_sample))

        # initial conv
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)

        # down blocks (len(down_channels)-1 blocks)
        self.downs = nn.ModuleList()
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    self.t_emb_dim,
                    down_sample=self.down_sample[i],
                    num_layers=self.num_down_layers
                )
            )

        # mid blocks (len(mid_channels)-1 blocks)
        self.mids = nn.ModuleList()
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    self.t_emb_dim,
                    num_layers=self.num_mid_layers
                )
            )

        # up blocks: iterate reversed indexes of down blocks (as in your original)
        self.ups = nn.ModuleList()
        # when i==0 we want out_channels to be self.down_channels[0] (no magic '16')
        for i in reversed(range(len(self.down_channels) - 1)):
            out_ch = self.down_channels[i - 1] if i > 0 else self.down_channels[0]
            in_ch = self.down_channels[i] * 2  # because x will be concatenated with skip
            self.ups.append(
                UpBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    t_emb_dim=self.t_emb_dim,
                    up_sample=self.down_sample[i],
                    num_layers=self.num_up_layers
                )
            )

        # output layers: use the first down channel as the final feature dimension
        final_ch = self.down_channels[0]
        self.norm_out = nn.GroupNorm(8, final_ch)
        self.conv_out = nn.Conv2d(final_ch, im_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        x: (B, C, H, W)
        t: either a (B,) tensor of timesteps, or a scalar/1D array-like
        """
        # initial projection
        out = self.conv_in(x)  # B x C1 x H x W

        # prepare time embedding: ensure it's a LongTensor on the same device
        if not torch.is_tensor(t):
            # allow passing scalar or array-like; create tensor on same device
            t = torch.tensor(t, device=x.device)
        t = t.to(x.device).long()
        t_emb = time_embedding(t, self.t_emb_dim)      # expected shape (B, t_emb_dim)
        t_emb = self.t_proj(t_emb)                     # (B, t_emb_dim)

        # store skip connections (store *before* applying the down block)
        down_outs = []
        for down in self.downs:
            down_outs.append(out)          # skip for corresponding up block
            out = down(out, t_emb)         # expects DownBlock to return the downsampled feature

        # middle blocks
        for mid in self.mids:
            out = mid(out, t_emb)

        # up sampling: pop skips in reverse order
        for up in self.ups:
            skip = down_outs.pop()
            out = up(out, skip, t_emb)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out
