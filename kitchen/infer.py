import torch
import minari
import torch.nn as nn
import gymnasium_robotics
import gymnasium as gym
from tqdm import tqdm
from skvideo.io import vwrite 
from diffusers.training_utils import EMAModel 
from blocks import *
from noise_scheduler import *
from dataloader import MinariSequenceDataset
from torch.utils.data import DataLoader
import collections

def infer():
    """
    Infering the results kitchen results
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gym.register_envs(gymnasium_robotics)
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'], render_mode = 'rgb_array')
    observation, _ = env.reset(seed = 150000)
    
    # Importing data and such normalization
    dataset = minari.load_dataset("D4RL/kitchen/partial-v2", download=True)

    dataset_torch = MinariSequenceDataset(dataset, device=device, normalize=True)

    # Observations
    observation_dim = dataset.observation_space['observation'].shape[0]
    observation_horizon = 2

    action_dim = dataset.action_space.shape[0]
    action_horizon = 8

    prediction_horizon = observation_horizon * action_horizon
    num_epochs = 5
    num_steps = 100
    num_warmup_steps = 500
    
    # Denoising model with Ema weights added
    denoising_model = ConditionalUnet1D(input_dim=action_dim, global_cond_dim= observation_dim * observation_horizon).eval()
    denoising_model.load_state_dict(torch.load("./saves/model_100.pth"))
    ema = EMAModel(parameters=denoising_model.parameters(), power=0.75)
    ema.load_state_dict(torch.load("./saves/ema_100.pth"))
    ema.copy_to(denoising_model.parameters())

    done = False
    images = [env.render()]
    observation_deque = collections.deque([observation] * observation_horizon, maxlen=observation_horizon)

    rewards = list()
    done = False
    step_idx = 0
    max_steps = 200
    noise_scheduler = NoiseScheduler(num_timesteps=num_steps)

    p_bar = tqdm(total=max_steps, desc="Eval PushT") 
    B = 1
    
    while not done:
        
        # Observation unormalization
        observation_sequence = np.stack(observation_deque)
        normalized_observation_sequence = dataset_torch._normalize_minmax_pm1(observation_sequence, dataset_torch.obs_min, dataset_torch.obs_min)
        normalized_observation_sequence = torch.from_numpy(normalized_observation_sequence).to(device, dtype=torch.float32)

        with torch.no_grad():

            observation_conditioning = normalized_observation_sequence.unsqueeze(0).flatten(start_dim=1)
            normalized_action = torch.randn(size = (B, prediction_horizon, action_dim), device = device)

            # Denoising process
            for t in reversed(range(num_steps)):

                noise_prediction = denoising_model(normalized_action, t, observation_conditioning)
                normalized_action, _ = noise_scheduler.reverse_process(normalized_action, noise_prediction, t = t)
            
        normalized_action = normalized_action.squeeze(0).detach().cpu().numpy()
        action_prediction = dataset_torch._unormalize_data(normalized_action, observation=False, action=True)


        start = observation_horizon - 1
        end = start + action_horizon
        action_sequences = action_prediction[start : end, :]

        for action in action_prediction:

            observation, reward, done, _, _ = env.step(action)             
            observation_deque.append(observation)
            rewards.append(reward)
            images.append(env.render())
            step_idx +=1

            p_bar.update(1)
            p_bar.set_postfix(reward = reward)

            if step_idx > max_steps :
                done = True
                break
        
    print(f"Score = {max(rewards)}")
    vwrite('kitchen.mp4', images)
    print("Saved video to: kitchen.mp4")
    print("Finished inference !")


if __name__ == "__main__":
    infer()
