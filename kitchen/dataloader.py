import torch
import minari
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings

# ignore deprecations

warnings.filterwarnings("ignore", category=DeprecationWarning) 

class MinariTransitionDataset(Dataset):
    def __init__(self, minari_dataset):

        transitions = []

        for ep in minari_dataset:
            
            obs = ep.observations      # shape: (T+1, obs_dim)
            acts = ep.actions          # shape: (T,   act_dim)
            rews = ep.rewards          # shape: (T,)
            terms = ep.terminations    # shape: (T,)
            truncs = ep.truncations    # shape: (T,)

            T = len(acts) # timesteps of EACH EPISODE

            for t in range(T):
                done = bool(terms[t] or truncs[t])
                transitions.append({
                    "observations":      torch.as_tensor(obs['observation'][t],     dtype=torch.float32),
                    "actions":           torch.as_tensor(acts[t],    dtype=torch.float32),
                    "rewards":           torch.as_tensor(rews[t],    dtype=torch.float32),
                    "next_observations": torch.as_tensor(obs['observation'][t + 1], dtype=torch.float32),
                    "done":              torch.as_tensor(done,       dtype=torch.float32),
                })


        self.transitions = transitions
        # stats of the obs and act
        self.compute_stats()

    # Getting min and max
    def compute_stats(self):
        """
        This returns the min and max of the data
        """
        obs = np.array([t["observations"].numpy() for t in self.transitions])
        actions = np.array([t["actions"].numpy() for t in self.transitions])

        self.obs_min = obs.min(axis=0)
        self.obs_max = obs.max(axis=0)

        self.act_min = actions.min(axis=0)
        self.act_max = actions.max(axis=0)

    # methods of len and get
    def __len__(self):
    
        return len(self.transitions)
    
    def __getitem__(self, idx):

        """
        Get the NORMALIZED data ! 
        """
        tr = self.transitions[idx]

        obs = (tr["observations"] - self.obs_min) / (self.obs_max - self.obs_min)
        obs = 2 * obs - 1  # [0,1] → [-1,1]

        next_obs = (tr["next_observations"] - self.obs_min) / (self.obs_max - self.obs_min)
        next_obs = 2 * next_obs - 1

        act = (tr["actions"] - self.act_min) / (self.act_max - self.act_min)
        act = 2 * act - 1

        return {
            "observations": obs,
            "actions": act,
            "rewards": tr["rewards"],
            "next_observations": next_obs,
            "done": tr["done"],
        }


# TODO The last episode seems to be missing the some points in the observation / action space. This one to be ignored in training 
# TODO https://github.com/Farama-Foundation/minari-dataset-generation-scripts?utm_source=chatgpt.com 
# TODO Go the the link above in order to get dataset of franks kitchen 
