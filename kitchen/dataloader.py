import torch
import minari
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings

# ignore deprecations
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# TODO The last episode seems to be missing the some points in the observation / action space. This one to be ignored in training 
# TODO https://github.com/Farama-Foundation/minari-dataset-generation-scripts?utm_source=chatgpt.com 
# TODO Go the the link above in order to get dataset of franks kitchen

class MinariSequenceDataset(Dataset):
    
    def __init__(self, minari_dataset, obs_horizon=2, act_horizon=8, normalize=False, device = 'cuda'):

        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon

        segments = []

        for ep in minari_dataset:

            # ep.observations is usually a dict, e.g. {'observation': array, 'goal': ...}
            obs = ep.observations['observation']   # shape: (T+1, obs_dim)
            acts = ep.actions                      # shape: (T,   act_dim)
            rews = ep.rewards
            terms = ep.terminations
            truncs = ep.truncations

            T = len(acts)  # number of transitions in episode

            # room for both obs and actions windows
            max_h = max(obs_horizon, act_horizon)

            for t in range(T - max_h + 1):
                obs_window = torch.as_tensor(
                    obs[t : t + obs_horizon],   # (obs_horizon, obs_dim)
                    dtype=torch.float32
                )
                act_window = torch.as_tensor(
                    acts[t : t + act_horizon],  # (act_horizon, act_dim)
                    dtype=torch.float32
                )

                # "done" at the last action in the window
                done_idx = t + act_horizon - 1
                done = bool(terms[done_idx] or truncs[done_idx])

                segments.append({
                    "observations": obs_window,
                    "actions": act_window,
                    "rewards": torch.as_tensor(rews[t : t + act_horizon], dtype=torch.float32),
                    "done": torch.as_tensor(done, dtype=torch.float32),
                })

        self.segments = segments

        # optionally compute normalization stats over all segments
        self.normalize_flag = normalize
        if normalize:
            self._compute_stats()

    def _compute_stats(self):
        # stack all obs/actions over all time in all segments
        obs_all = torch.cat(
            [s["observations"].reshape(-1, s["observations"].shape[-1]) for s in self.segments],
            dim=0,
        )
        act_all = torch.cat(
            [s["actions"].reshape(-1, s["actions"].shape[-1]) for s in self.segments],
            dim=0,
        )

        self.obs_min = obs_all.min(dim=0).values
        self.obs_max = obs_all.max(dim=0).values
        self.act_min = act_all.min(dim=0).values
        self.act_max = act_all.max(dim=0).values

    def _normalize_minmax_pm1(self, x, x_min, x_max, eps=1e-6):
        # [0,1] then [-1,1]
        return 2 * (x - x_min) / (x_max - x_min + eps) - 1
    
    def _unormalize_data(self, x, observation = True, action = False):
        """
        Unormalize Data
        """
        x = (x + 1) / 2

        if observation : 
            unormalized_x = x * (self.obs_max - self.obs_min) + self.obs_min

        if action :
            unormalized_x = x * (self.act_max - self.act_min) + self.act_min
        
        return unormalized_x

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]

        obs = seg["observations"]
        act = seg["actions"]

        if self.normalize_flag:
            obs = self._normalize_minmax_pm1(obs, self.obs_min, self.obs_max)
            act = self._normalize_minmax_pm1(act, self.act_min, self.act_max)

        return {
            "observations": obs,      # (obs_horizon, obs_dim)
            "actions": act,           # (act_horizon, act_dim)
            "rewards": seg["rewards"],
            "done": seg["done"],
        }
