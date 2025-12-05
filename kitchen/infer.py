import torch
import minari
import torch.nn as nn
import gymnasium_robotics
import gymnasium as gym
from tqdm import tqdm
import imageio                     
from diffusers.training_utils import EMAModel 
from blocks import *
from noise_scheduler import *
from dataloader import MinariSequenceDataset
from torch.utils.data import DataLoader
import collections
import numpy as np

def infer():
    """
    Inferring Franka Kitchen results
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gym.register_envs(gymnasium_robotics)
    env = gym.make(
        'FrankaKitchen-v1',
        tasks_to_complete=['microwave', 'kettle', 'light switch', 'slide cabinet'],
        render_mode='rgb_array',
    )
    
    
    observation_full, _ = env.reset(seed=150000)
    observation = observation_full['observation']

    # Import dataset
    dataset = minari.load_dataset("D4RL/kitchen/complete-v2", download=True)

    observation_horizon = 2
    action_horizon = 8
    
    # IMPORTANT: must match training (normalize=True)
    dataset_torch = MinariSequenceDataset(
        dataset,
        device=device,
        normalize=True,
        obs_horizon=observation_horizon,
        act_horizon=action_horizon
    )

    # Dimensions
    observation_dim = dataset.observation_space['observation'].shape[0]
    action_dim = dataset.action_space.shape[0]

    num_steps = 100  # diffusion steps
    
    # Denoising model + EMA
    denoising_model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=observation_dim * observation_horizon,
        down_dims=[472, 944, 1888],
        diffusion_step_embed_dim=256
    ).eval().to(device)

    denoising_model.load_state_dict(torch.load("./saves/kitchen_chkpt_final.pth", map_location=device))
    ema = EMAModel(parameters=denoising_model.parameters(), power=0.75)
    ema.load_state_dict(torch.load("./saves/ema_chkpt_final.pth", map_location=device))
    ema.copy_to(denoising_model.parameters())

    images = [env.render()]
    observation_deque = collections.deque(
        [observation] * observation_horizon,
        maxlen=observation_horizon
    )

    rewards = []
    done = False
    step_idx = 0
    max_steps = 100
    noise_scheduler = NoiseScheduler(num_timesteps=num_steps, device=device)

    p_bar = tqdm(total=max_steps, desc="Eval Kitchen") 
    B = 1

    # Normalization stats
    obs_min = dataset_torch.obs_min.to(device)
    obs_max = dataset_torch.obs_max.to(device)
    action_min = dataset_torch.act_min.to(device)
    action_max = dataset_torch.act_max.to(device)

    while not done:
        
        # Build observation sequence as torch tensor
        observation_sequence = torch.as_tensor(
            np.stack(observation_deque),
            dtype=torch.float32,
            device=device,
        )

        # Normalize observations to [-1, 1]
        normalized_observation_sequence = dataset_torch._normalize_minmax_pm1(
            observation_sequence,
            obs_min,
            obs_max
        )

        with torch.no_grad():

            observation_conditioning = normalized_observation_sequence.unsqueeze(0).flatten(start_dim=1)
            
            # Start from Gaussian noise
            normalized_action = torch.randn(
                size=(B, action_horizon, action_dim),
                device=device
            )

            # Denoising process
            for t in reversed(range(num_steps)):
                noise_prediction = denoising_model(normalized_action, t, observation_conditioning)
                normalized_action, _ = noise_scheduler.reverse_process(
                    normalized_action,
                    noise_prediction,
                    t=t
                )

        # Remove batch dim
        normalized_action_tensor = normalized_action.squeeze(0)

        # Unnormalize actions
        action_prediction_tensor = dataset_torch._unormalize_data(
            normalized_action_tensor,
            action_min,
            action_max
        )
        action_prediction = action_prediction_tensor.detach().cpu().numpy()

        action_sequences = action_prediction[:action_horizon, :]

        for action in action_sequences:
            action = np.clip(action, env.action_space.low, env.action_space.high)

            observation, reward, done, _, _ = env.step(action)             
            observation_deque.append(observation['observation'])
            rewards.append(reward)
            images.append(env.render())
            step_idx += 1

            p_bar.update(1)
            p_bar.set_postfix(reward=float(reward))

            if step_idx > max_steps:
                done = True
                break
        
        print(f"Current best score = {max(rewards):.3f}")

    imageio.mimsave("kitchen.mp4", images, fps=30)
    print("Saved video to: kitchen.mp4")
    print("Finished inference !")


if __name__ == "__main__":
    infer()
