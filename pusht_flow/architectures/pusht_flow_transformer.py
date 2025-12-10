import os
import yaml
import tqdm
import argparse
import torch
import torch.nn as nn
import numpy as np

from torch.optim import AdamW
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from model.dataset import PushTStateDataset
from architectures.building_blocks import *        # if you still need anything from there
from architectures.flow_scheduler import FlowScheduler
from transformer import TransformerForDiffusion


# (kept for completeness, but not used with PushTStateDataset)
def collate_fn(batch):
    obs = torch.tensor([b['observation_state'] for b in batch], dtype=torch.float32)
    act = torch.tensor([b['action'] for b in batch], dtype=torch.float32)
    return {'observation_state': obs, 'action': act}


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training Flow-Matching Transformer on: {device}")

    # Horizons and variables
    observation_dim = 5
    observation_horizon = 2
    action_dim = 2
    pred_horizon = 16
    action_horizon = 8
    num_epochs = 100
    num_timesteps = 10   # only used for logging / consistency if needed

    # -------------------------------
    # Model: Transformer for flow matching
    # -------------------------------
    noise_prediction_model = TransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,           # predict velocity with same shape as actions
        horizon=pred_horizon,            # T = 16
        n_obs_steps=observation_horizon,
        cond_dim=observation_dim,        # we condition on obs only
        n_layer=4,
        n_head=4,
        n_emb=128,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=True,
        time_as_cond=True,
        n_cond_layers=1
    ).to(device)

    ema = EMAModel(parameters=noise_prediction_model.parameters(), power=0.75)

    # Flow scheduler (just for add_noise & possibly ODE later)
    noise_scheduler = FlowScheduler(num_timesteps=num_timesteps, device=device).to(device)

    # -------------------------------
    # Dataset & dataloader
    # -------------------------------
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"

    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=observation_horizon,
        action_horizon=action_horizon
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,   # type: ignore
        batch_size=256,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        drop_last=True,   # avoids weird last batch shape issues
    )

    # -------------------------------
    # Optimizer & LR scheduler
    # -------------------------------
    optimizer = AdamW(noise_prediction_model.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    print("----------------------------------")
    print("Starting Flow-Matching Transformer training (PushT)")
    print("----------------------------------")

    total_loss = []

    with tqdm.tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:

            epoch_loss = []

            with tqdm.tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    optimizer.zero_grad()

                    # nbatch['obs']    : (B, obs_horizon, observation_dim)
                    # nbatch['action'] : (B, pred_horizon, action_dim)
                    nobs = nbatch['obs'].to(device)       # (B, To, 5)
                    naction = nbatch['action'].to(device) # (B, T, 2)

                    B = nobs.shape[0]

                    # Conditioning sequence: (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:, :observation_horizon, :]  # (B,2,5)

                    # 1. Sample Noise x0 ~ N(0, I)
                    noise = torch.randn_like(naction, device=device)  # (B,T,2)

                    # 2. Sample continuous time t ~ U(0,1)
                    # shape: (B,)
                    t = torch.rand(B, device=device)

                    # 3. Forward process: x_t = (1 - t) * x0 + t * x1
                    noisy_actions = noise_scheduler.add_noise(
                        original_samples=naction,  # x1
                        noise=noise,               # x0
                        t=t                        # (B,)
                    )

                    # 4. Target velocity: v = x1 - x0 (constant along the path)
                    target_velocity = naction - noise

                    # 5. Predict velocity with Transformer
                    velocity_pred = noise_prediction_model(
                        sample=noisy_actions,     # (B,T,2)
                        timestep=t,               # (B,) continuous time
                        cond=obs_cond             # (B,2,5)
                    )

                    # 6. Loss: MSE between predicted velocity and target velocity
                    loss = nn.functional.mse_loss(velocity_pred, target_velocity)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    ema.step(noise_prediction_model.parameters())

                    # logging
                    loss_cpu = float(loss.item())
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            mean_epoch_loss = float(np.mean(epoch_loss))
            tglobal.set_postfix(loss=mean_epoch_loss)
            total_loss.append(mean_epoch_loss)

            if (epoch_idx + 1) % 10 == 0:
                print(f"Finished epoch {epoch_idx+1}/{num_epochs} | Loss: {mean_epoch_loss:.4f}")
                os.makedirs("./saves", exist_ok=True)
                torch.save(
                    noise_prediction_model.state_dict(),
                    './saves/pusht_flow_transformer_chkpt_final.pth'
                )
                torch.save(
                    ema.state_dict(),
                    './saves/ema_pusht_flow_transformer_chkpt_final.pth'
                )

    np.save("./loss_pusht_flow_transformer.npy", np.array(total_loss))
    print("Finished Flow-Matching Transformer training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for Flow-Matching training')
    parser.add_argument('--config', dest='config_path', default='./config.yaml', type=str)
    args = parser.parse_args()
    train(args)
