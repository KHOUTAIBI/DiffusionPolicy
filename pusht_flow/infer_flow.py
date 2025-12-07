import collections
from building_blocks import *
from dataset import *
from flow_scheduler import * 
import gymnasium as gym
import gym_pusht
from tqdm import tqdm
from diffusers.training_utils import EMAModel
import argparse
import numpy as np
import imageio

def infer():
    """
    Inference for the PushT DDPM policy
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    observation, _ = env.reset()

    # Horizons
    num_timesteps = 100
    observation_dim = 5
    observation_horizon = 2
    action_dim = 2
    pred_horizon = 16
    action_horizon = 8

    # Dataset + stats
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=observation_horizon,    
        action_horizon=action_horizon
    )

    
    stats = dataset.stats
    print("env obs shape:", np.array(observation).shape)  # should be (5,)


    # Model
    denoising_model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=observation_horizon * observation_dim,
        n_groups=8
    ).to(device).eval() 
    

    denoising_model.load_state_dict(torch.load("./saves/pusht_chkpt_final.pth"))
    

    ema = EMAModel(parameters=denoising_model.parameters(), power=0.75)
    ema.load_state_dict(torch.load("./saves/ema_chkpt_final.pth"))
    ema.copy_to(denoising_model.parameters())

    # Env loop
    imgs = [env.render()]  # type: ignore
    obs_deque = collections.deque([observation] * observation_horizon, maxlen=observation_horizon)

    rewards = [] 
    done = False
    step_idx = 0
    max_steps = 200
    noise_scheduler = FlowScheduler(num_timesteps=num_timesteps)

    with tqdm(total=max_steps, desc="Eval PushT") as pbar:
        while not done:
            
            B = 1
            obs_sequence = np.stack(obs_deque)
            nobs = normalize_data(obs_sequence, stats=stats['obs'])
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # ... inside the inference loop ...
            with torch.no_grad():
                # 1. Prepare Observations (Conditioning)
                #    Flatten (1, T_obs, Obs_dim) -> (1, T_obs * Obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)  # <--- THIS LINE WAS MISSING
                
                # 2. Start from pure noise (t=0)
                #    In Flow Matching, Noise is the starting point (x0)
                naction = torch.randn((B, pred_horizon, action_dim), device=device)
                
                # 3. Flow Matching Inference (ODE Solver)
                num_inference_steps = 10
                dt = 1.0 / num_inference_steps
                
                for i in range(num_inference_steps):
                    # Map step i to the model's expected timestep range [0, 100]
                    t_index = int((i / num_inference_steps) * 10)
                    t_tensor = torch.tensor([t_index], device=device).expand(B)
                    
                    # Predict Velocity
                    velocity_pred = denoising_model(naction, t_tensor, obs_cond)
                    
                    # Euler Step: x_{t+1} = x_t + v * dt
                    naction = naction + velocity_pred * dt

            # Unnormalize and execute actions...
            naction = naction.squeeze(0).detach().cpu().numpy()
            action_prediction = unnormalize_data(naction, stats=stats['action'])


            # Take first action_horizon steps
            start = observation_horizon - 1
            end = start + action_horizon
            action_seq = action_prediction[start : end, :]

            for a in action_seq:
                observation, reward, done, _, _ = env.step(a)
                obs_deque.append(observation)
                rewards.append(reward)
                imgs.append(env.render())
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps or done:
                    done = True
                    break

    print('Score:', max(rewards))
    imageio.mimsave('vis.mp4', imgs, fps=30)
    print("Saved rollout video as vis.mp4")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDPM inference')
    parser.add_argument('--config', dest='config_path', default='./config.yaml', type=str)
    args = parser.parse_args()
    infer()
