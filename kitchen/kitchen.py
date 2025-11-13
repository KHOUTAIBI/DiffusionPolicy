import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'])

# TODO https://github.com/Farama-Foundation/minari-dataset-generation-scripts?utm_source=chatgpt.com 
# TODO Go the the link above in order to get dataset of franks kitchen 