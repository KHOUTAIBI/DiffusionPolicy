import os
import torch
import torch.nn.functional as F
import minari
import numpy as np
import gymnasium_robotics
import gymnasium as gym
from tqdm import tqdm
import imageio
import collections

from dataloader import MinariSequenceDataset
from transformers_model import TransformerForDiffusion
from flow_scheduler import FlowScheduler
from diffusers.training_utils import EMAModel


def infer():
    """
    Inference on Franka Kitchen using Flow-Matching Transformer
    with progress (tasks-done) conditioning, consistent with training.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Flow-Matching Transformer inference on: {device}")

    # --- env setup ---
    gym.register_envs(gymnasium_robotics)
    env = gym.make(
        'FrankaKitchen-v1',
        tasks_to_complete=['microwave', 'kettle', 'light switch', 'slide cabinet'],
        render_mode='rgb_array'
    )

    observation_full, _ = env.reset()
    observation = observation_full['observation']  # (obs_dim,)

    # --- dataset + normalization (must match training) ---
    dataset = minari.load_dataset("D4RL/kitchen/partial-v2", download=True)

    observation_horizon = 2
    action_horizon = 8
    num_steps = 100              # same num_timesteps as in FlowScheduler (not critical)
    num_progress_levels = 5      # 0..4 tasks done

    dataset_torch = MinariSequenceDataset(
        dataset,
        device=device,
        normalize=True,
        obs_horizon=observation_horizon,
        act_horizon=action_horizon
    )

    observation_dim = dataset.observation_space['observation'].shape[0]
    action_dim = dataset.action_space.shape[0]

    print(f"obs_dim = {observation_dim}, act_dim = {action_dim}")
    print(f"obs_horizon = {observation_horizon}, act_horizon = {action_horizon}")

    # --- model (must match training config) ---
    cond_dim = observation_dim + num_progress_levels

    model = TransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,          # predicts velocity v(x,t,cond)
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
    ).to(device).eval()

    # load flow-matching weights
    model.load_state_dict(
        torch.load("./saves/kitchen_flow_transformer_chkpt.pth", map_location=device)
    )

    # load EMA if exists
    ema_ckpt = "./saves/ema_flow_transformer_chkpt.pth"
    if os.path.exists(ema_ckpt):
        ema = EMAModel(parameters=model.parameters(), power=0.75)
        ema.load_state_dict(torch.load(ema_ckpt, map_location=device))
        ema.copy_to(model.parameters())
        print("Loaded EMA flow weights into model.")
    else:
        print("EMA flow checkpoint not found, using raw model weights.")

    # --- flow scheduler (for ODE step) ---
    flow_scheduler = FlowScheduler(num_timesteps=num_steps, device=device)

    # --- rollout buffers ---
    images = [env.render()]
    observation_deque = collections.deque(
        [observation] * observation_horizon,
        maxlen=observation_horizon
    )

    rewards = []
    done = False
    step_idx = 0
    max_steps = 200

    p_bar = tqdm(total=max_steps, desc="Eval Kitchen (Flow + progress)")

    B = 1

    # normalization stats on device
    obs_min = dataset_torch.obs_min.to(device)
    obs_max = dataset_torch.obs_max.to(device)
    act_min = dataset_torch.act_min.to(device)
    act_max = dataset_torch.act_max.to(device)

    # tracks "#tasks done" ≈ env reward at current step
    reward_level = 0.0

    # ODE integration settings
    num_ode_steps = 10
    ts = torch.linspace(0.0, 1.0, num_ode_steps + 1, device=device)  # [0,1]

    while not done:

        # --- build observation window (To, obs_dim) ---
        observation_sequence = torch.as_tensor(
            np.stack(observation_deque),
            dtype=torch.float32,
            device=device
        )  # (obs_horizon, obs_dim)

        # normalize obs to [-1, 1]
        normalized_observation_sequence = dataset_torch._normalize_minmax_pm1(
            observation_sequence,
            obs_min,
            obs_max
        )  # (obs_horizon, obs_dim)

        # --- build progress conditioning (same semantics as training) ---
        # reward_level ≈ current "#tasks done"
        progress_id = int(round(reward_level))
        progress_id = max(0, min(num_progress_levels - 1, progress_id))

        progress_one_hot = F.one_hot(
            torch.tensor([progress_id], device=device),
            num_classes=num_progress_levels
        ).float()  # (1, 5)

        # expand along obs_horizon → (1, To, 5)
        progress_token = progress_one_hot.unsqueeze(1).expand(
            -1, observation_horizon, -1
        )

        # final conditioning sequence: (1, To, obs_dim + 5)
        cond_seq = torch.cat(
            [normalized_observation_sequence.unsqueeze(0), progress_token],
            dim=-1
        )

        with torch.no_grad():
            # start from base x0 ~ N(0,I) in normalized action space
            x = torch.randn(
                size=(B, action_horizon, action_dim),
                device=device
            )

            # integrate dx/dt = v_theta(x,t,cond) from t=0 to 1
            for i in range(num_ode_steps):

                t0 = ts[i]
                t1 = ts[i + 1]
                dt = t1 - t0

                # time as (B,) to match training
                t_batch = t0.expand(B)

                v = model(
                    sample=x,        # (B,T,D)
                    timestep=t_batch,
                    cond=cond_seq
                )                  # (B,T,D) velocity field

                x = flow_scheduler.step(
                    model_output=v,
                    timestep=t0,
                    sample=x,
                    dt=dt
                )

        # x is now approximate y (normalized actions)
        normalized_action_tensor = x.squeeze(0)  # (T, act_dim)

        # unnormalize to env action scale
        action_prediction_tensor = dataset_torch._unormalize_data(
            normalized_action_tensor,
            act_min,
            act_max
        )
        action_prediction = action_prediction_tensor.detach().cpu().numpy()
        action_sequences = action_prediction[:action_horizon, :]

        # --- execute in env ---
        for action in action_sequences:
            action = np.clip(action, env.action_space.low, env.action_space.high)

            observation, reward, done, _, _ = env.step(action)
            reward_level = float(reward)   # use current reward as "#tasks done"
            rewards.append(reward_level)
            observation_deque.append(observation['observation'])
            images.append(env.render())
            step_idx += 1

            p_bar.update(1)
            p_bar.set_postfix(
                step=step_idx,
                reward=reward_level,
                max_r=float(max(rewards) if rewards else 0.0)
            )

            if step_idx >= max_steps:
                done = True
                break

        if rewards:
            print(
                f"Last reward_level = {reward_level:.3f}, "
                f"max step reward so far = {max(rewards):.3f}"
            )
        else:
            print("No rewards yet.")

    
    imageio.mimsave("kitchen.mp4", images, fps=30)
    print("Saved video to: kitchen_flow_transformer_progress.mp4")
    print("Finished Flow-Matching Transformer inference with progress conditioning!")


if __name__ == "__main__":
    infer()
