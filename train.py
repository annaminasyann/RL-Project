import argparse
import json
import os
import warnings

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

import config

warnings.filterwarnings("ignore")


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_env(seed=0, log_path=None):
    def _init():
        env = gym.make(config.ENV_ID)
        if log_path:
            env = Monitor(env, log_path)
        env.reset(seed=seed)
        return env
    return _init


def load_best_params(algo):
    path = os.path.join(config.TUNING_DIR, f"{algo}_best.json")
    if not os.path.exists(path):
        print(f"  [warn] no tuned params at {path}, using defaults")
        return {}
    with open(path) as f:
        return json.load(f)["best_params"]


def build_model(algo, env, params, device, seed=0):
    if algo == "ppo":
        return PPO("MlpPolicy", env, verbose=0, device=device, seed=seed,
                   tensorboard_log=config.LOG_DIR, **params)
    if algo == "sac":
        return SAC("MlpPolicy", env, verbose=0, device=device, seed=seed,
                   buffer_size=100_000, learning_starts=1_000,
                   tensorboard_log=config.LOG_DIR, **params)
    if algo == "td3":
        n_actions = env.envs[0].action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        return TD3("MlpPolicy", env, verbose=0, device=device, seed=seed,
                   buffer_size=100_000, learning_starts=1_000,
                   action_noise=action_noise,
                   tensorboard_log=config.LOG_DIR, **params)
    raise ValueError(f"unknown algo: {algo}")


def train_one(algo, seed):
    device = get_device()
    print(f"\n{'='*60}\nTraining {algo.upper()} | seed={seed} | device={device} | steps={config.FINAL_TIMESTEPS:,}\n{'='*60}")

    params = load_best_params(algo)
    print(f"  Using tuned params: {params}")

    log_path = os.path.join(config.LOG_DIR, f"{algo}_seed{seed}")
    os.makedirs(log_path, exist_ok=True)

    train_env = DummyVecEnv([make_env(seed=seed, log_path=log_path)])
    eval_env = DummyVecEnv([make_env(seed=seed + 1000)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config.MODELS_DIR, f"{algo}_seed{seed}"),
        log_path=log_path,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.EVAL_EPISODES,
        deterministic=True,
        verbose=0,
    )

    model = build_model(algo, train_env, params, device, seed=seed)
    model.learn(
        total_timesteps=config.FINAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=False,
        tb_log_name=f"{algo}_seed{seed}",
    )

    final_model_path = os.path.join(config.MODELS_DIR, f"{algo}_seed{seed}_final")
    model.save(final_model_path)

    final_mean, final_std = evaluate_policy(
        model, eval_env, n_eval_episodes=config.EVAL_EPISODES, deterministic=True
    )
    print(f"  Final return: {final_mean:.2f} ± {final_std:.2f}")

    train_env.close()
    eval_env.close()

    return {
        "algo": algo,
        "seed": seed,
        "final_mean": float(final_mean),
        "final_std": float(final_std),
        "model_path": final_model_path,
        "params": params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=config.ALGORITHMS + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    algos = config.ALGORITHMS if args.algo == "all" else [args.algo]
    seeds = [args.seed] if args.seed is not None else config.FINAL_SEEDS

    results = []
    for a in algos:
        for s in seeds:
            r = train_one(a, s)
            results.append(r)

    out_path = os.path.join(config.RESULTS_DIR, "final_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll done. Results → {out_path}")


if __name__ == "__main__":
    main()
