import gymnasium as gym
import gymnasium_robotics

env = gym.make('AdroitHandHammer-v1', render_mode = 'human')

obs, info = env.reset()


while True:

    action = env.action_space.sample()
    

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        
    env.render()

env.close()

