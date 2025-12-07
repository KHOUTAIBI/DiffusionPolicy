# transformer_model.py
from typing import Union, Optional
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal timestep embedding:
    input: (B,) of timesteps (ints)
    output: (B, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:  # pad if odd
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb  # (B, dim)


class TransformerForDiffusion(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 horizon: int,
                 n_obs_steps: int = None,
                 cond_dim: int = 0,
                 n_layer: int = 8,
                 n_head: int = 8,
                 n_emb: int = 256,
                 p_drop_emb: float = 0.1,
                 p_drop_attn: float = 0.1,
                 causal_attn: bool = True,
                 time_as_cond: bool = True,
                 n_cond_layers: int = 2):
        """
        input_dim:     action_dim
        output_dim:    action_dim (we predict noise with same shape)
        horizon:       action horizon (T)
        n_obs_steps:   observation horizon (To)
        cond_dim:      obs_dim
        """
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1  # time token

        self.time_as_cond = time_as_cond
        self.obs_as_cond = cond_dim > 0
        self.horizon = horizon

        if not time_as_cond:
            # time token goes into main sequence instead
            T += 1
            T_cond -= 1

        if self.obs_as_cond:
            assert time_as_cond, "obs_as_cond requires time_as_cond=True"
            T_cond += n_obs_steps

        # ------------ input embedding (actions) ------------
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # ------------ condition encoder (time + obs) ------------
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        if self.obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.T = T
        self.T_cond = T_cond
        self.n_emb = n_emb

        # encoder for cond tokens
        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        self.encoder_only = False

        if T_cond > 0:
            # we have condition tokens
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                # simple MLP encoder
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )

            # decoder for main sequence
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # encoder-only BERT style
            self.encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # ------------ causal attention mask ------------
        if causal_attn:
            sz = T
            # upper-triangular mask with -inf above diagonal
            mask = torch.triu(torch.ones(sz, sz), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            self.register_buffer("tgt_mask", mask)

            if time_as_cond and self.obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T), torch.arange(S),
                    indexing='ij'
                )
                # allow attending to time + past obs only
                mask_mem = t >= (s - 1)
                mask_mem = mask_mem.float().masked_fill(mask_mem == 0, float('-inf')).masked_fill(mask_mem == 1, 0.0)
                self.register_buffer("memory_mask", mask_mem)
            else:
                self.memory_mask = None
        else:
            self.tgt_mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # init
        self._init_weights()
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        if self.cond_pos_emb is not None:
            nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

    def forward(self,
                sample: torch.Tensor,            # (B, T, input_dim)
                timestep: Union[torch.Tensor, int, float],
                cond: Optional[torch.Tensor] = None  # (B, To, cond_dim)
                ) -> torch.Tensor:
        """
        sample:  (B, T, input_dim)   - noisy actions
        timestep: scalar or (B,)     - diffusion step
        cond:    (B, To, cond_dim)   - observation seq (optional)
        returns: (B, T, output_dim)  - predicted noise
        """

        B, T, _ = sample.shape
        assert T == self.horizon, f"expected horizon {self.horizon}, got {T}"

        # --- time embedding ---
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(B)   # (B,)
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B,1,n_emb)

        # --- input embedding ---
        input_emb = self.input_emb(sample)  # (B,T,n_emb)

        if self.encoder_only:
            # BERT-style (not used in our setup, but kept for completeness)
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)  # (B,T+1,n_emb)
            pos = self.pos_emb[:, :token_embeddings.shape[1], :]
            x = self.drop(token_embeddings + pos)
            x = self.encoder(src=x, mask=self.tgt_mask)
            x = x[:, 1:, :]  # drop time token
        else:
            # --- encoder for condition tokens ---
            cond_embeddings = time_emb  # always have time token as first

            if self.obs_as_cond and cond is not None:
                # cond: (B, To, cond_dim)
                cond_obs_emb = self.cond_obs_emb(cond)  # (B,To,n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)  # (B,T_cond,n_emb)

            tc = cond_embeddings.shape[1]
            cond_pos = self.cond_pos_emb[:, :tc, :]
            x_cond = self.drop(cond_embeddings + cond_pos)
            memory = self.encoder(x_cond)  # (B,T_cond,n_emb)

            # --- decoder for main seq ---
            token_embeddings = input_emb  # (B,T,n_emb)
            pos = self.pos_emb[:, :T, :]
            x = self.drop(token_embeddings + pos)  # (B,T,n_emb)

            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.tgt_mask,
                memory_mask=self.memory_mask
            )

        x = self.ln_f(x)
        x = self.head(x)  # (B,T,output_dim)
        return x
