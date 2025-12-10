# transformer.py

from typing import Optional, Union, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal embedding used for diffusion timesteps.
    Input: (B,) timesteps (float or int)
    Output: (B, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: (B,)
        """
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / (half_dim - 1)
        exponents = torch.exp(torch.arange(half_dim, device=device) * exponent)
        # (B,1) * (half_dim,) -> (B, half_dim)
        emb = timesteps[:, None].float() * exponents[None, :]
        # sin & cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            # if odd, pad one dim
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)


class TransformerForDiffusion(nn.Module):
    """
    Decoder-style Transformer denoiser for diffusion over action sequences.

    - Input sample: (B, T, input_dim)        # noisy actions
    - Timestep: scalar or (B,)               # diffusion step
    - Cond: (B, To, cond_dim) optional       # observation window

    Output:
    - (B, T, output_dim)                     # predicted noise
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int,
        cond_dim: int = 0,
        n_layer: int = 4,
        n_head: int = 4,
        n_emb: int = 128,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = True,
        time_as_cond: bool = True,
        n_cond_layers: int = 1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.cond_dim = cond_dim
        self.n_emb = n_emb
        self.time_as_cond = time_as_cond

        # number of tokens in the main sequence
        T = horizon

        # number of tokens in conditioning encoder
        T_cond = 0
        if time_as_cond:
            T_cond += 1       # time token
        if cond_dim > 0:
            T_cond += n_obs_steps   # obs tokens

        # ----- Input embeddings -----
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # ----- Time embedding -----
        self.time_emb = SinusoidalPosEmb(n_emb)

        # ----- Cond encoder -----
        self.cond_obs_emb = None
        if cond_dim > 0:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False

        if T_cond > 0:
            # positional embeddings for conditioning tokens
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers,
                )
            else:
                # simple MLP if no encoder layers
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb),
                )

            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer,
            )
        else:
            # encoder-only (BERT-style) if no cond
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer,
            )

        self.encoder_only = encoder_only
        self.T = T
        self.T_cond = T_cond

        # ----- Causal mask (for decoder) -----
        if causal_attn:
            sz = T
            mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
            # True entries will be masked (set to -inf)
            self.register_buffer(
                "tgt_mask",
                mask.float().masked_fill(mask, float("-inf"))
            )
        else:
            self.tgt_mask = None

        # No special memory mask here:
        self.memory_mask = None

        # ----- Output head -----
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # ----- Init -----
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        if self.cond_pos_emb is not None:
            nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        sample: (B, T, input_dim)   - noisy actions
        timestep: scalar or (B,)    - diffusion step
        cond: (B, To, cond_dim)     - observation window (optional)

        Returns:
        (B, T, output_dim)          - predicted noise
        """
        B, T, _ = sample.shape
        device = sample.device

        # --- handle timesteps ---
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(device)
        timesteps = timesteps.float().expand(B)

        # time embedding token
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B,1,n_emb)

        # input embedding for main sequence
        x_tokens = self.input_emb(sample)  # (B,T,n_emb)

        if self.encoder_only:
            # BERT-style: time token + action tokens into an encoder
            tokens = torch.cat([time_emb, x_tokens], dim=1)  # (B,T+1,n_emb)
            pos = self.pos_emb[:, : tokens.shape[1], :]
            x = self.drop(tokens + pos)
            x = self.encoder(x)          # (B,T+1,n_emb)
            x = x[:, 1:, :]              # drop time token
        else:
            # ----- Encoder over conditioning -----
            cond_tokens = time_emb  # always include time as first token
            if cond is not None and self.cond_dim > 0 and self.cond_obs_emb is not None:
                # cond: (B, To, cond_dim)
                cond_obs = self.cond_obs_emb(cond)  # (B,To,n_emb)
                cond_tokens = torch.cat([cond_tokens, cond_obs], dim=1)  # (B,1+To,n_emb)

            # positional embeddings for cond tokens
            pos_cond = self.cond_pos_emb[:, : cond_tokens.shape[1], :]
            cond_tokens = self.drop(cond_tokens + pos_cond)

            memory = self.encoder(cond_tokens)  # (B,T_cond,n_emb)

            # ----- Decoder over action tokens -----
            pos_main = self.pos_emb[:, :T, :]
            tgt = self.drop(x_tokens + pos_main)  # (B,T,n_emb)

            x = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=self.tgt_mask,
                memory_mask=self.memory_mask,
            )  # (B,T,n_emb)

        # ----- Head -----
        x = self.ln_f(x)
        x = self.head(x)  # (B,T,output_dim)
        return x
