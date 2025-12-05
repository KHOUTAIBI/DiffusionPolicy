import minari
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from skvideo.io import vwrite


def make_env_from_dataset(dataset, eval_env=False):
    """
    Recover the environment associated to a Minari dataset,
    with rgb_array rendering enabled.
    """
    env = dataset.recover_environment(
        render_mode="human",
        eval_env=eval_env
    )
    print("Recovered env spec id:", env.spec.id if hasattr(env, "spec") else "no spec")
    print("Render modes:", env.metadata.get("render_modes", None))
    return env


def replay_episode(env, episode, out_path="episode.mp4"):
    """
    Replay a single episode from a Minari dataset in the given env,
    and save a video.

    env: environment from dataset.recover_environment(render_mode='rgb_array')
    episode: one element of the Minari dataset (dataset[i])
    out_path: filename of the mp4 video to save
    """
    # Reset env (this may not perfectly match the dataset's initial state,
    # but is enough to visualize behavior)
    obs, info = env.reset()
    frames = [env.render()]

    total_reward = 0.0

    actions = episode.actions
    terminations = episode.terminations
    truncations = episode.truncations

    for t, (a, term, trunc) in enumerate(zip(actions, terminations, truncations)):

        obs, rew, terminated, truncated, info = env.step(a)
        total_reward += rew
        env.render()

        # You can use either the env signals (terminated/truncated) or
        # the ones stored in the episode (term/trunc). Here we "or" them.
        if terminated or truncated or term or trunc:
            print(f"Stopped at step {t}, terminated={terminated}, truncated={truncated}")
            break

    print(f"Episode replay finished. Sum of rewards: {total_reward:.3f}")
    frames = np.array(frames)  # (T, H, W, 3)
    vwrite(out_path, frames)
    print(f"Saved video to: {out_path}")


def main():
    # 1. Load Minari dataset
    dataset_id = "D4RL/kitchen/partial-v2"
    print(f"Loading dataset: {dataset_id}")
    dataset = minari.load_dataset(dataset_id, download=True)
    print(f"Number of episodes in dataset: {len(dataset)}")

    # 2. Recover env(s)
    # register robotics envs if not already:
    gym.register_envs(gymnasium_robotics)

    env = make_env_from_dataset(dataset, eval_env=False)
    # eval_env = make_env_from_dataset(dataset, eval_env=True)  # optional, if you want

    # 3. Replay a few episodes
    num_eps_to_replay = 3
    for i in range(min(num_eps_to_replay, len(dataset))):
        print(f"\n=== Replaying episode {i} ===")
        ep = dataset[i]
        out_file = f"kitchen_partial_ep{i}.mp4"
        replay_episode(env, ep, out_path=out_file)


if __name__ == "__main__":
    main()
