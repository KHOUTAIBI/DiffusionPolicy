import torch
from building_blocks import *
import gymnasium as gym
from tqdm import tqdm
import collections
import gym_pusht
from dataset import *
from noise_scheduler import * 
from skvideo.io import vwrite 
import argparse


def infer(args):
    """
    Infer the results
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make("gym_pusht/PushT-v0", render_mode = "rgb_array")
    observation, info = env.reset()
    
    # Horizons and variables
    num_timesteps = 100
    observation_dim = 5
    observation_horizon = 2
    action_dim = 2
    pred_horizon = 16
    action_horizon = 8
    

    # Getting dataset
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"

    
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=observation_horizon,
        action_horizon=action_horizon
    )

    # stats
    stats = dataset.stats
    
    # Loading state model
    denoising_model = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=action_dim * observation_dim, n_groups=2)
    denoising_model.load_state_dict(torch.load("./saves/pusht_chkpt_60.pth"))
    denoising_model.eval()

    # Rewards and states
    observation, info = env.reset()
    

    stats = dataset.stats

    imgs = [env.render()] # type: ignore
    obs_deque = collections.deque(
        [observation] * observation_horizon, maxlen=observation_horizon
    )

    rewards = list()
    done = False
    step_idx = 0
    max_steps = 200
    noise_sceduler = NoiseScheduler(num_timesteps=num_timesteps, beta_init=0.0001, beta_end=0.02)

    with tqdm(total=max_steps, desc="Eval Pusht") as pbar:
        while not done:
            
            B = 1
            obs_sequence = np.stack(obs_deque)
            nobs = torch.from_numpy(obs_sequence).to(device, dtype=torch.float32)


            with torch.no_grad():
                
                # Getting random step
                obs_cond = nobs.unsqueeze(0).flatten(start_dim = 1)
                noisy_action = torch.randn((B, pred_horizon, action_dim), device = device)
                naction = noisy_action

                # Denoising process
                for t in reversed(range(num_timesteps)):
                    
                    noise_prediction = denoising_model(naction, t, obs_cond)
                    naction, _ = noise_sceduler.reverse_process(naction, noise_prediction, t)

            # getting action
            naction = naction.detach().to('cpu').numpy()
            naction = naction[0]
           

            # Action prediction
            action_prediction = unnormalize_data(naction, stats = stats['action']) # i need to focus more, unormalize on action !
            # actions
            start = observation_horizon - 1
            end = start + action_horizon
            action = action_prediction[start:end,:]

            for i in range(len(action)):

                observation, reward, done, truncated, info = env.step(action[i])
                obs_deque.append(observation)

                rewards.append(reward)
                imgs.append(env.render())

                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward = reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break
    
    # print out the maximum target coverage
    print('Score: ', max(rewards))

    # visualize
    from IPython.display import Video
    vwrite('vis.mp4', imgs)
    Video('vis.mp4', embed=True, width=256, height=256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDPM training')
    parser.add_argument('--config', dest='config_path', default='./config.yaml', type=str)
    args = parser.parse_args()
    infer(args)