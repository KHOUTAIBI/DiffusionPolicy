# train_transformer.py
from dataloader import MinariSequenceDataset
from transformers_model import TransformerForDiffusion
from flow_scheduler import FlowScheduler   

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
    num_progress_levels = 5   # 0,1,2,3,4 tasks done

    # --- load dataset ---
    dataset = minari.load_dataset("D4RL/kitchen/partial-v2", download=True)
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
        drop_last=True,
    )

    # quick sanity
    for idx, batch in enumerate(loader):
        print(idx, {k: v.shape for k, v in batch.items()})
        print("obs window shape:", batch["observations"][0].shape)
        print("act window shape:", batch["actions"][0].shape)
        break

    # --- training hyperparams ---
    num_epochs = 100
    num_steps = 100            # used only for LR schedule / potential ODE later
    num_warmup_steps = 1000
    max_grad_norm = 1.0

    # --- Transformer vector field (flow model) ---
    cond_dim = observation_dim + num_progress_levels

    model = TransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,          # predict velocity v(x,t,cond)
        horizon=action_horizon,
        n_obs_steps=observation_horizon,
        cond_dim=cond_dim,
        n_layer=6,
        n_head=8,
        n_emb=256,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=True,
        time_as_cond=True,
        n_cond_layers=2
    ).to(device)

    ema = EMAModel(parameters=model.parameters(), power=0.75)

    # straight-line flow scheduler (for interpolation)
    flow_scheduler = FlowScheduler(num_timesteps=num_steps, device=device)

    base_lr = 1e-4
    optimizer = Adam(
        model.parameters(),
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
    print("Starting Transformer flow-matching training with progress conditioning")
    print("----------------------------------")

    tglobal = tqdm(range(num_epochs), desc="epoch", leave=False)
    global_step = 0

    for epoch_idx in tglobal:
        tepoch = tqdm(loader, desc="batch", leave=False)
        epoch_loss = []

        for batch in tepoch:
            optimizer.zero_grad(set_to_none=True)

            # shapes:
            #   obs:     (B, obs_horizon, obs_dim)
            #   acts:    (B, act_horizon, act_dim)  -> x1
            #   rewards: (B, act_horizon)
            obs = batch["observations"].to(device)
            acts = batch["actions"].to(device)
            rewards = batch["rewards"].to(device)

            B = obs.shape[0]

            # --- progress / goal id from last reward in window ---
            last_reward = rewards[:, -1]              # (B,)
            progress_id = last_reward.round().long()
            progress_id = torch.clamp(
                progress_id,
                0,
                num_progress_levels - 1
            )                                         # (B,)

            progress_one_hot = F.one_hot(
                progress_id,
                num_classes=num_progress_levels
            ).float()                                 # (B, 5)

            progress_token = progress_one_hot.unsqueeze(1).expand(
                -1, observation_horizon, -1
            )                                         # (B, To, 5)

            cond_seq = torch.cat([obs, progress_token], dim=-1)  # (B, To, obs_dim+5)

            # --- Flow Matching forward process ---
            # x1 = acts (data), x0 = noise
            x1 = acts
            x0 = torch.randn_like(acts, device=device)

            # sample t ~ Uniform(0,1)
            t = torch.rand(B, device=device)          # (B,)
            # interpolate: x_t = (1-t)*x0 + t*x1
            x_t = flow_scheduler.add_noise(x1, x0, t)

            # target velocity field v*(x_t,t) = x1 - x0 (constant along path)
            v_target = x1 - x0                        # (B, T, act_dim)

            # model predicts v_theta(x_t, t, cond)
            v_pred = model(
                sample=x_t,        # (B,T,act_dim)
                timestep=t,        # (B,) in [0,1]
                cond=cond_seq      # (B,To,cond_dim)
            )

            loss = loss_func(v_pred, v_target)
            loss.backward()

            # # gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            ema.step(model.parameters())
            global_step += 1

            loss_val = float(loss.item())
            tepoch.set_postfix(loss=loss_val)
            epoch_loss.append(loss_val)

        mean_loss = float(np.mean(epoch_loss))
        tglobal.set_postfix(loss=mean_loss)

        # small debug: check rewards & progress mapping
        with torch.no_grad():
            print("rewards window:", rewards[0])
            print("progress_id:", progress_id[0].item())

        if (epoch_idx + 1) % 10 == 0:
            print(f"Finished epoch {epoch_idx+1}/{num_epochs} | Loss: {mean_loss:.4f}")
            os.makedirs("./saves", exist_ok=True)
            torch.save(model.state_dict(), "./saves/kitchen_flow_transformer_chkpt.pth")
            torch.save(ema.state_dict(), "./saves/ema_flow_transformer_chkpt.pth")

    print("Finished flow-mathcing Transformer training with progress conditioning!")


if __name__ == "__main__":
    train()
