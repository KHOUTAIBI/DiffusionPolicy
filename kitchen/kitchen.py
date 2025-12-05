from dataloader import *
from blocks import *
from noise_scheduler import *
from tqdm import tqdm
from torch.optim import Adam
from diffusers.training_utils import EMAModel 
from diffusers.optimization import get_scheduler
import os
import torch
import torch.nn as nn
import numpy as np
import minari
from torch.utils.data import DataLoader

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def train():
    """
    Training the kitchen dataset using Diffusion policy
    """ 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Working on: {device}")

    observation_horizon = 2
    action_horizon = 8

    # Loading dataset
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
    print(f"The observation space dimension is: {observation_dim} and action dim is: {action_dim}")
    
    loader = DataLoader(
        dataset_torch,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # quick sanity print of shapes
    for idx, batch in enumerate(loader):
        print(idx, {k: v.shape for k, v in batch.items()})
        print(f"obs window shape: {batch['observations'][0].shape}, action window shape: {batch['actions'][0].shape}")
        print(f"Example obs window:\n{batch['observations'][idx]}")
        break

    num_epochs = 100
    num_steps = 100
    num_warmup_steps = 500

    denoising_model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=observation_dim * observation_horizon,
        n_groups=8,
        down_dims=[472, 944, 1888],
        diffusion_step_embed_dim=256,
    ).to(device)
    
    ema = EMAModel(parameters=denoising_model.parameters(), power=0.75)
    noise_scheduler = NoiseScheduler(num_timesteps=num_steps, device=device)

    optimizer = Adam(params=denoising_model.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(loader) * num_epochs
    )
    loss_func = nn.functional.mse_loss

    print("----------------------------------")
    print("Starting to train")
    print("----------------------------------")

    tglobal = tqdm(range(num_epochs), desc='epoch', leave=False)

    for epoch_indx in tglobal:

        tepoch = tqdm(loader, desc='batch', leave=False)
        epoch_loss = []

        for batch in tepoch:

            optimizer.zero_grad()
            normalized_observations = batch['observations'].to(device)   # (B, obs_horizon, obs_dim)
            normalized_actions = batch['actions'].to(device)             # (B, act_horizon, act_dim)
            B = normalized_observations.shape[0]

            normalized_observation_cond = normalized_observations[:, :observation_horizon, :].flatten(start_dim=1)

            t = torch.randint(0, num_steps, size=(B,), device=device)
            noise = torch.randn_like(normalized_actions, device=device)
                
            noisy_action = noise_scheduler.add_noise(normalized_actions, noise, t)
            predicted_noise = denoising_model(noisy_action, t, normalized_observation_cond)

            loss = loss_func(predicted_noise, noise)
            loss.backward()
                
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            epoch_loss.append(loss_cpu)

            optimizer.step()
            lr_scheduler.step()
            ema.step(denoising_model.parameters())
        
        mean_epoch_loss = float(np.mean(epoch_loss))
        tglobal.set_postfix(loss=mean_epoch_loss)

        if (epoch_indx + 1) % 10 == 0:
            print(f"Finished epoch {epoch_indx+1}/{num_epochs} | Loss: {mean_epoch_loss:.4f}")
            os.makedirs("./saves", exist_ok=True)
            torch.save(denoising_model.state_dict(), './saves/kitchen_chkpt_final.pth')
            torch.save(ema.state_dict(), './saves/ema_chkpt_final.pth')
        
    print("Finished training!")


if __name__ == "__main__":
    train()
