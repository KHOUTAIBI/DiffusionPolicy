import collections
import time
import argparse
import os

import torch
import numpy as np
from tqdm import tqdm
from skvideo.io import vwrite

import gymnasium as gym
import gym_pusht

from model.dataset import *
from transformer import TransformerForDiffusion
from architectures.flow_scheduler import FlowScheduler
from diffusers.training_utils import EMAModel


def infer_flow_transformer_benchmark():
    """
    Inference + benchmark for the PushT Flow-Matching Transformer policy.

    - Uses flow matching: x_t evolves according to dx/dt = v_theta(x_t, t)
    - Runs multiple seeds, times full episodes, and saves stats.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Flow-Matching Transformer inference on: {device}")

    # ------------------------
    # Env + horizons
    # ------------------------
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
    observation, _ = env.reset()
    print("Initial env obs shape:", np.array(observation).shape)  # should be (5,)

    num_timesteps = 10          # used for FlowScheduler init (not critical for FM)
    observation_dim = 5
    observation_horizon = 2
    action_dim = 2
    pred_horizon = 16           # sequence length for diffusion/flow
    action_horizon = 8
    max_steps = 200

    # ------------------------
    # Dataset + normalization stats
    # ------------------------
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=observation_horizon,
        action_horizon=action_horizon,
    )
    stats = dataset.stats

    # ------------------------
    # Flow-Matching Transformer model (must match training config)
    # ------------------------
    denoising_model = TransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,          # predict velocity v(x,t)
        horizon=pred_horizon,           # T = 16
        n_obs_steps=observation_horizon,
        cond_dim=observation_dim,       # we condition on obs only
        n_layer=4,
        n_head=4,
        n_emb=128,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=True,
        time_as_cond=True,
        n_cond_layers=1,
    ).to(device).eval()

    # Load flow-matching transformer checkpoints
    denoising_model.load_state_dict(
        torch.load("./saves/pusht_flow_transformer_chkpt_final.pth", map_location=device)
    )

    ema = EMAModel(parameters=denoising_model.parameters(), power=0.75)
    ema.load_state_dict(
        torch.load("./saves/ema_pusht_flow_transformer_chkpt_final.pth", map_location=device)
    )
    ema.copy_to(denoising_model.parameters())

    # Flow Scheduler (for add_noise & step)
    flow_scheduler = FlowScheduler(num_timesteps=num_timesteps, device=device).to(device)

    # ------------------------
    # Benchmark config
    # ------------------------
    seeds = list(range(100))   # 50 fixed seeds: 0..49
    episode_times = []
    episode_scores = []

    print(f"Benchmarking Flow-Matching Transformer on seeds: {seeds}")

    # Number of ODE steps (Euler steps from t=0 to t=1)
    num_inference_steps = 10
    dt = 1.0 / num_inference_steps

    for seed in seeds:
        print(f"\n========== Seed {seed} ==========")

        observation, _ = env.reset(seed=seed)

        # Only record video for the first seed to avoid huge files
        record_video = (seed == seeds[0])
        if record_video:
            imgs = [env.render()]  # type: ignore
        else:
            imgs = []

        obs_deque = collections.deque(
            [observation] * observation_horizon,
            maxlen=observation_horizon,
        )

        rewards = []
        done = False
        step_idx = 0

        # Start timing this episode
        t0 = time.perf_counter()

        with tqdm(total=max_steps, desc=f"Eval PushT Flow (seed {seed})") as pbar:
            while not done:
                B = 1

                # ------------------------
                # Build normalized obs window
                # ------------------------
                obs_sequence = np.stack(obs_deque)  # (obs_horizon, obs_dim)
                nobs = normalize_data(obs_sequence, stats=stats['obs'])
                nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

                with torch.no_grad():
                    # cond: (B, obs_horizon, obs_dim)
                    obs_cond = nobs.unsqueeze(0)  # (1,2,5)

                    # Start from pure Gaussian noise x0 ~ N(0,I)
                    xt = torch.randn(
                        (B, pred_horizon, action_dim),
                        device=device
                    )

                    # Flow-matching ODE integration: from t=0 -> 1
                    for i in range(num_inference_steps):
                        # Time in [0,1], we can use midpoint for stability
                        t_scalar = i / num_inference_steps
                        t = torch.full((B,), t_scalar, device=device)

                        v_pred = denoising_model(
                            sample=xt,       # (B,T,2)
                            timestep=t,      # (B,)
                            cond=obs_cond    # (B,2,5)
                        )
                        xt = flow_scheduler.step(
                            model_output=v_pred,
                            timestep=t,
                            sample=xt,
                            dt=dt
                        )

                # ------------------------
                # Unnormalize actions
                # ------------------------
                xt = xt.squeeze(0).detach().cpu().numpy()  # (T,2)
                action_prediction = unnormalize_data(
                    xt,
                    stats=stats['action']
                )

                # Take first action_horizon steps (like original DDPM code)
                start = observation_horizon - 1
                end = start + action_horizon
                action_seq = action_prediction[start:end, :]

                for a in action_seq:
                    observation, reward, done, _, _ = env.step(a)
                    obs_deque.append(observation)
                    rewards.append(reward)
                    if record_video:
                        imgs.append(env.render())
                    step_idx += 1

                    pbar.update(1)
                    pbar.set_postfix(reward=reward)

                    if step_idx > max_steps or done:
                        done = True
                        break

        # End timing
        elapsed = time.perf_counter() - t0
        episode_times.append(elapsed)
        score = max(rewards) if len(rewards) > 0 else 0.0
        episode_scores.append(score)

        print(f"Seed {seed} | Score: {score:.3f} | Episode time: {elapsed:.3f} s")

        # Save a single video for seed 0
        if record_video:
            vwrite('vis_flow_transformer_seed0.mp4', np.array(imgs))
            print("Saved rollout video for seed 0 as vis_flow_transformer_seed0.mp4")

    # ------------------------
    # Save benchmark results
    # ------------------------
    episode_times = np.array(episode_times)
    episode_scores = np.array(episode_scores)
    seeds_arr = np.array(seeds)

    np.save("flow_transformer_inference_seeds.npy", seeds_arr)
    np.save("flow_transformer_inference_times.npy", episode_times)
    np.save("flow_transformer_inference_scores.npy", episode_scores)

    print("\n==== Flow-Matching Transformer Benchmark summary ====")
    print("Seeds:", seeds)
    print("Mean episode time:", episode_times.mean())
    print("Std  episode time:", episode_times.std())
    print("Mean score:", episode_scores.mean())
    print("Std  score:", episode_scores.std())
    print("Saved:")
    print("  flow_transformer_inference_seeds.npy")
    print("  flow_transformer_inference_times.npy")
    print("  flow_transformer_inference_scores.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for Flow-Matching Transformer inference benchmark')
    parser.add_argument('--config', dest='config_path', default='./config.yaml', type=str)
    args = parser.parse_args()
    infer_flow_transformer_benchmark()
