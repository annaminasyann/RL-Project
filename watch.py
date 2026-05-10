import argparse
import os

# Set up headless rendering before importing gym/mujoco.
# EGL works on most cloud GPUs (including L4 on GCP).
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO, SAC, TD3

import config

ALGO_CLS = {"ppo": PPO, "sac": SAC, "td3": TD3}

VIDEOS_DIR = os.path.join(config.ROOT, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)


def record_episode(algo, seed=0, n_episodes=1, video_length=1000):
    model_path = os.path.join(config.MODELS_DIR, f"{algo}_seed{seed}_final.zip")
    if not os.path.exists(model_path):
        print(f"[skip] no model at {model_path}")
        return None

    env = gym.make(config.ENV_ID, render_mode="rgb_array")
    model = ALGO_CLS[algo].load(model_path)

    out_path = os.path.join(VIDEOS_DIR, f"{algo}_seed{seed}.mp4")
    frames = []
    total_reward = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 5000 + ep)
        ep_reward = 0.0
        for step in range(video_length):
            frame = env.render()
            frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total_reward += ep_reward
        print(f"  {algo.upper()} episode {ep+1}: return = {ep_reward:.1f}")

    env.close()

    imageio.mimsave(out_path, frames, fps=30, codec="libx264", quality=8)
    print(f"  saved → {out_path} ({len(frames)} frames, mean return {total_reward/n_episodes:.1f})")
    return out_path


def make_comparison_grid(seed=0):
    """Stack PPO/SAC/TD3 episodes side by side into one video for easy comparison."""
    import imageio.v3 as iio

    paths = {a: os.path.join(VIDEOS_DIR, f"{a}_seed{seed}.mp4") for a in config.ALGORITHMS}
    available = {a: p for a, p in paths.items() if os.path.exists(p)}
    if len(available) < 2:
        print("[skip] need at least 2 algorithm videos for comparison")
        return None

    print(f"  Building comparison grid from {list(available.keys())}...")
    videos = {a: iio.imread(p) for a, p in available.items()}
    min_frames = min(v.shape[0] for v in videos.values())

    stacked_frames = []
    for i in range(min_frames):
        row = np.concatenate([videos[a][i] for a in available.keys()], axis=1)
        stacked_frames.append(row)

    out_path = os.path.join(VIDEOS_DIR, "comparison.mp4")
    imageio.mimsave(out_path, stacked_frames, fps=30, codec="libx264", quality=8)
    print(f"  saved → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=config.ALGORITHMS + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--length", type=int, default=1000, help="max steps per episode")
    parser.add_argument("--no-comparison", action="store_true")
    args = parser.parse_args()

    print("Recording videos...")
    algos = config.ALGORITHMS if args.algo == "all" else [args.algo]
    for a in algos:
        record_episode(a, seed=args.seed, n_episodes=args.episodes, video_length=args.length)

    if args.algo == "all" and not args.no_comparison:
        print("\nBuilding side-by-side comparison...")
        make_comparison_grid(seed=args.seed)

    print("\nDone. Videos saved in videos/")


if __name__ == "__main__":
    main()
