import json
import yaml
import tqdm
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
from noise_scheduler import NoiseScheduler
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam
import gymnasium as gym
import gym_pusht


env = gym.make("gym_pusht/PushT-v0", render_mode = 'human')
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    print(observation)
    if terminated or truncated:
        observation, info = env.reset()

env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for DDPM training')
    parser.add_argument('--config', dest='config_path', default='./config.yaml', type=str)
    args = parser.parse_args()
    