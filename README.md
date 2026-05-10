# HalfCheetah-v5 Deep RL Pipeline

End-to-end reinforcement learning pipeline comparing **PPO**, **SAC**, and **TD3** on the HalfCheetah-v5 MuJoCo benchmark, with **Bayesian hyperparameter optimization** via Optuna.

## What this does

1. **Tunes** each algorithm's hyperparameters using Bayesian optimization (Optuna's TPE sampler).
2. **Trains** each algorithm using its best-found hyperparameters.
3. **Evaluates** the final trained agents.
4. **Plots** all results: learning curves, tuning history, final comparison.

## Quick start (one command)

```bash
bash run_all.sh
```

That's it. The script handles everything: virtual environment, dependencies, tuning, training, evaluation, and plotting.

## Manual steps (if you prefer)

```bash
# 1. Create environment and install deps
python3 -m venv venv
source venv/bin/activate
pip install -r code/requirements.txt

# 2. Bayesian hyperparameter tuning
python code/tune.py --algo all

# 3. Final training using best hyperparameters
python code/train.py --algo all

# 4. Generate figures
python code/plot_results.py

# 5. Evaluate (and optionally render)
python code/evaluate.py --algo sac --render
```

## Time budget

Configured for a roughly 2-hour total runtime on CPU/MPS (Apple Silicon). Adjust `code/config.py` if you have more compute available:

- `TUNING_TRIALS` — number of Bayesian optimization trials per algorithm (default 6)
- `TUNING_TIMESTEPS` — steps per tuning trial (default 20,000)
- `FINAL_TIMESTEPS` — steps for the final training run (default 50,000)
- `FINAL_SEEDS` — seeds for the final run (default `[0]`)

For a full-quality study, increase to `TUNING_TRIALS=30`, `TUNING_TIMESTEPS=100_000`, `FINAL_TIMESTEPS=1_000_000`, `FINAL_SEEDS=[0,1,2]`. That run takes 24 to 48 hours on a GPU.

## Outputs

After `run_all.sh` finishes:

- `tuning/{ppo,sac,td3}_best.json` — best hyperparameters and full trial history
- `models/{algo}_seed0_final.zip` — saved trained models (loadable via SB3)
- `logs/` — TensorBoard logs and Monitor CSVs
- `figures/` — all plots
- `results/final_results.json` — summary metrics

## Watch training live

```bash
tensorboard --logdir logs
```

Open http://localhost:6006 in your browser.

## Watch a trained agent run

```bash
python code/evaluate.py --algo sac --seed 0 --render
```

Opens a MuJoCo window showing the cheetah running. Requires MuJoCo display libraries.
