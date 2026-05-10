import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy

import config

ALGO_CLS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def evaluate_run(algo, seed, episodes=10, render=False):
    model_path = os.path.join(config.MODELS_DIR, f"{algo}_seed{seed}_final.zip")
    if not os.path.exists(model_path):
        print(f"[skip] no model at {model_path}")
        return None

    render_mode = "human" if render else None
    env = gym.make(config.ENV_ID, render_mode=render_mode)
    env.reset(seed=seed + 5000)

    model = ALGO_CLS[algo].load(model_path)
    mean, std = evaluate_policy(model, env, n_eval_episodes=episodes, deterministic=True)
    print(f"{algo.upper():4s} seed={seed} → {mean:.2f} ± {std:.2f}  ({episodes} episodes)")
    env.close()
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=config.ALGORITHMS + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    algos = config.ALGORITHMS if args.algo == "all" else [args.algo]
    for a in algos:
        evaluate_run(a, args.seed, episodes=args.episodes, render=args.render)


if __name__ == "__main__":
    main()
