# train_transformer.py
from dataloader import MinariSequenceDataset
from noise_scheduler import NoiseScheduler
from transformers_model import TransformerForDiffusion

import torch
import torch.nn as nn
import torch.nn.functional as F
import minari
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Working on: {device}")

    # --- horizons ---
    observation_horizon = 2
    action_horizon = 8

    # --- progress / goal conditioning ---
    # We approximate "how many subtasks are done" from rewards.
    # Assume 0..4 tasks completed  -> 5 possible levels.
    num_progress_levels = 5   # 0,1,2,3,4

    # --- load dataset ---
    dataset = minari.load_dataset("D4RL/kitchen/complete-v2", download=True)
    dataset_torch = MinariSequenceDataset(
        dataset,
        obs_horizon=observation_horizon,
        act_horizon=action_horizon,
        normalize=True,
        device=device
    )

    observation_dim = dataset.observation_space['observation'].shape[0]
    action_dim = dataset.action_space.shape[0]
    print(f"obs_dim = {observation_dim}, act_dim = {action_dim}")

    loader = DataLoader(
        dataset_torch,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,   # avoid tiny remainder batches
    )

    # quick sanity
    for idx, batch in enumerate(loader):
        print(idx, {k: v.shape for k, v in batch.items()})
        print("obs window shape:", batch["observations"][0].shape)
        print("act window shape:", batch["actions"][0].shape)
        break

    # --- training hyperparams ---
    num_epochs = 100
    num_steps = 100            # diffusion steps
    num_warmup_steps = 1000
    max_grad_norm = 1.0

    # --- Transformer denoiser ---
    # Note: cond_dim now includes obs_dim + progress_one_hot_dim
    cond_dim = observation_dim + num_progress_levels

    denoising_model = TransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,          # predict noise with same shape as actions
        horizon=action_horizon,         # T = act_horizon
        n_obs_steps=observation_horizon,
        cond_dim=cond_dim,              # per-step conditioning dim
        n_layer=6,                      # decoder layers
        n_head=8,
        n_emb=256,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=True,
        time_as_cond=True,
        n_cond_layers=2               # encoder layers
    ).to(device)

    ema = EMAModel(parameters=denoising_model.parameters(), power=0.75)
    noise_scheduler = NoiseScheduler(num_timesteps=num_steps, device=device)

    base_lr = 1e-4
    optimizer = Adam(
        denoising_model.parameters(),
        lr=base_lr,
        weight_decay=1e-6
    )

    total_training_steps = len(loader) * num_epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )

    loss_func = nn.functional.mse_loss

    print("----------------------------------")
    print("Starting Transformer DDPM training with progress conditioning")
    print("----------------------------------")

    tglobal = tqdm(range(num_epochs), desc="epoch", leave=False)
    global_step = 0
    total_loss = list()

    for epoch_idx in tglobal:

        tepoch = tqdm(loader, desc="batch", leave=False)
        epoch_loss = []

        for batch in tepoch:
            optimizer.zero_grad(set_to_none=True)

            # shapes in batch:
            #   obs: (B, obs_horizon, obs_dim)
            #   act: (B, act_horizon, act_dim)
            #   rewards: (B, act_horizon)
            obs = batch["observations"].to(device)
            acts = batch["actions"].to(device)
            rewards = batch["rewards"].to(device)  # (B, act_horizon)

            B = obs.shape[0]

            # --- build progress / goal id per window ---
            # Heuristic: total reward in this window ~ number of completed subtasks.
            # Clamp to [0, num_progress_levels-1] and one-hot encode.
            # This gives you a "progress token": how many tasks done so far.
            last_reward = rewards[:, -1]                    # (B,)
            progress_id = last_reward.round().long()  
            progress_id = torch.clamp(
                progress_id,
                0,
                num_progress_levels - 1
            )                                              # (B,)

            progress_one_hot = F.one_hot(
                progress_id,
                num_classes=num_progress_levels
            ).float()                                      # (B, 5)

            # Make it per-time-step in the obs window: (B, obs_horizon, 5)
            progress_token = progress_one_hot.unsqueeze(1).expand(
                -1, observation_horizon, -1
            )

            # Final conditioning sequence: concat obs and progress code
            # Shape: (B, obs_horizon, obs_dim + 5)
            cond_seq = torch.cat([obs, progress_token], dim=-1)

            # --- DDPM forward process ---
            t = torch.randint(0, num_steps, size=(B,), device=device)
            noise = torch.randn_like(acts, device=device)
            noisy_actions = noise_scheduler.add_noise(acts, noise, t)

            # --- noise prediction ---
            pred_noise = denoising_model(
                sample=noisy_actions,    # (B, T, act_dim)
                timestep=t,              # (B,) or scalar
                cond=cond_seq            # (B, To, cond_dim)
            )

            loss = loss_func(pred_noise, noise)
            loss.backward()

            # gradient clipping
            # torch.nn.utils.clip_grad_norm_(denoising_model.parameters(), max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            ema.step(denoising_model.parameters())
            global_step += 1

            loss_val = float(loss.item())
            tepoch.set_postfix(loss=loss_val)
            epoch_loss.append(loss_val)

        mean_loss = float(np.mean(epoch_loss))
        tglobal.set_postfix(loss=mean_loss)
        
        total_loss.append(mean_loss)

        if (epoch_idx + 1) % 10 == 0:
            print(f"Finished epoch {epoch_idx+1}/{num_epochs} | Loss: {mean_loss:.4f}")
            os.makedirs("./saves", exist_ok=True)
            torch.save(denoising_model.state_dict(), "./saves/kitchen_transformer_chkpt.pth")
            torch.save(ema.state_dict(), "./saves/ema_transformer_chkpt.pth")

    np.save("total_loss_transformer", total_loss)
    print("Finished training Transformer with progress conditioning!")


if __name__ == "__main__":
    train()
