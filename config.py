import os

ENV_ID = "HalfCheetah-v5"

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
TUNING_DIR = os.path.join(ROOT, "tuning")
MODELS_DIR = os.path.join(ROOT, "models")

for d in (LOG_DIR, RESULTS_DIR, FIGURES_DIR, TUNING_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

TUNING_TRIALS = 15
TUNING_TIMESTEPS = 30_000
TUNING_EVAL_EPISODES = 5

FINAL_TIMESTEPS = 50_000
FINAL_SEEDS = [0]
EVAL_EPISODES = 10
EVAL_FREQ = 5_000

ALGORITHMS = ["ppo", "sac", "td3"]
