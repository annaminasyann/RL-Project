import argparse
import json
import os
import warnings

import gymnasium as gym
import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import config

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_env(seed=0):
    def _init():
        env = gym.make(config.ENV_ID)
        env.reset(seed=seed)
        return env
    return _init


def sample_ppo_params(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
    }


def sample_sac_params(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
    }


def sample_td3_params(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "policy_delay": trial.suggest_categorical("policy_delay", [1, 2, 4]),
    }


def build_model(algo, env, params, device, seed=0):
    if algo == "ppo":
        return PPO("MlpPolicy", env, verbose=0, device=device, seed=seed, **params)
    if algo == "sac":
        return SAC("MlpPolicy", env, verbose=0, device=device, seed=seed,
                   buffer_size=100_000, learning_starts=1_000, **params)
    if algo == "td3":
        n_actions = env.action_space.shape[-1] if hasattr(env, "action_space") else env.envs[0].action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        return TD3("MlpPolicy", env, verbose=0, device=device, seed=seed,
                   buffer_size=100_000, learning_starts=1_000,
                   action_noise=action_noise, **params)
    raise ValueError(f"unknown algo: {algo}")


def objective_factory(algo, device):
    sampler_fn = {"ppo": sample_ppo_params, "sac": sample_sac_params, "td3": sample_td3_params}[algo]

    def objective(trial):
        params = sampler_fn(trial)
        try:
            train_env = DummyVecEnv([make_env(seed=trial.number)])
            eval_env = gym.make(config.ENV_ID)
            eval_env.reset(seed=trial.number + 1000)

            model = build_model(algo, train_env, params, device, seed=trial.number)
            model.learn(total_timesteps=config.TUNING_TIMESTEPS, progress_bar=False)

            mean_reward, _ = evaluate_policy(
                model, eval_env, n_eval_episodes=config.TUNING_EVAL_EPISODES, deterministic=True
            )
            train_env.close()
            eval_env.close()
            return float(mean_reward)
        except Exception as e:
            print(f"  [trial {trial.number}] failed: {e}")
            return -10000.0

    return objective


def tune(algo, n_trials=None):
    n_trials = n_trials or config.TUNING_TRIALS
    device = get_device()
    print(f"\n{'='*60}\nTuning {algo.upper()} | device={device} | trials={n_trials} | steps/trial={config.TUNING_TIMESTEPS:,}\n{'='*60}")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=2),
        study_name=f"{algo}_tuning",
    )
    study.optimize(objective_factory(algo, device), n_trials=n_trials, show_progress_bar=False)

    print(f"\n{algo.upper()} tuning complete.")
    print(f"  Best return: {study.best_value:.2f}")
    print(f"  Best params: {study.best_params}")

    out = {
        "algo": algo,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "trials": [
            {"number": t.number, "value": float(t.value) if t.value is not None else None, "params": t.params}
            for t in study.trials
        ],
    }
    out_path = os.path.join(config.TUNING_DIR, f"{algo}_best.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=config.ALGORITHMS + ["all"], default="all")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    algos = config.ALGORITHMS if args.algo == "all" else [args.algo]
    for a in algos:
        tune(a, n_trials=args.trials)


if __name__ == "__main__":
    main()
