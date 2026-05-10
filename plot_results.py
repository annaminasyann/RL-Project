import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

COLORS = {"ppo": "#3B5998", "sac": "#0D9488", "td3": "#B85042"}


def load_monitor(algo, seed):
    log_path = os.path.join(config.LOG_DIR, f"{algo}_seed{seed}")
    if not os.path.exists(log_path):
        return None
    files = [f for f in os.listdir(log_path) if "monitor" in f and f.endswith(".csv")]
    if not files:
        return None
    df = pd.read_csv(os.path.join(log_path, files[0]), skiprows=1)
    if len(df) == 0:
        return None
    df["timesteps"] = df["l"].cumsum()
    return df


def load_eval(algo, seed):
    eval_path = os.path.join(config.LOG_DIR, f"{algo}_seed{seed}", "evaluations.npz")
    if not os.path.exists(eval_path):
        return None, None
    data = np.load(eval_path)
    return data["timesteps"], data["results"].mean(axis=1)


def smooth(x, window=10):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_learning_curves():
    fig, ax = plt.subplots(figsize=(11, 6))

    for algo in config.ALGORITHMS:
        all_curves = []
        for seed in config.FINAL_SEEDS:
            df = load_monitor(algo, seed)
            if df is None or len(df) == 0:
                continue
            steps = df["timesteps"].values
            rewards = smooth(df["r"].values, window=min(20, len(df) // 2 or 1))
            steps_smoothed = steps[-len(rewards):]
            all_curves.append((steps_smoothed, rewards))

        if not all_curves:
            print(f"  [skip] no data for {algo}")
            continue

        # interpolate to common grid
        max_step = max(c[0][-1] for c in all_curves)
        grid = np.linspace(0, max_step, 100)
        interpolated = []
        for steps, rewards in all_curves:
            interp = np.interp(grid, steps, rewards)
            interpolated.append(interp)
        arr = np.stack(interpolated)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) if len(arr) > 1 else np.zeros_like(mean)

        ax.plot(grid, mean, color=COLORS[algo], lw=2.4, label=algo.upper())
        if len(arr) > 1:
            ax.fill_between(grid, mean - std, mean + std, color=COLORS[algo], alpha=0.18)

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode return")
    ax.set_title("HalfCheetah-v5: Learning curves (with tuned hyperparameters)",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "learning_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved → {out}")


def plot_eval_curves():
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False

    for algo in config.ALGORITHMS:
        for seed in config.FINAL_SEEDS:
            steps, rewards = load_eval(algo, seed)
            if steps is None:
                continue
            ax.plot(steps, rewards, color=COLORS[algo], lw=2.4,
                    label=f"{algo.upper()} (seed {seed})", marker="o", markersize=5)
            plotted = True

    if not plotted:
        print("  [skip] no evaluation data")
        plt.close()
        return

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean evaluation return")
    ax.set_title("Deterministic evaluation curves",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "eval_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved → {out}")


def plot_final_comparison():
    results_path = os.path.join(config.RESULTS_DIR, "final_results.json")
    if not os.path.exists(results_path):
        print("  [skip] no final_results.json")
        return

    with open(results_path) as f:
        results = json.load(f)

    by_algo = {}
    for r in results:
        by_algo.setdefault(r["algo"], []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    algos = list(by_algo.keys())
    means = [np.mean([r["final_mean"] for r in by_algo[a]]) for a in algos]
    stds = [np.mean([r["final_std"] for r in by_algo[a]]) for a in algos]
    colors = [COLORS[a] for a in algos]

    bars = ax.bar([a.upper() for a in algos], means, yerr=stds, capsize=8,
                  color=colors, alpha=0.85, edgecolor="black", linewidth=1)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{m:.0f}", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Final evaluation return")
    ax.set_title("Final performance comparison (after tuning)",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "final_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved → {out}")


def plot_tuning_history():
    for algo in config.ALGORITHMS:
        path = os.path.join(config.TUNING_DIR, f"{algo}_best.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        trials = data["trials"]
        values = [t["value"] for t in trials if t["value"] is not None]
        if not values:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(range(len(values)), values, marker="o", lw=2, color=COLORS[algo], markersize=8)
        running_best = np.maximum.accumulate(values)
        ax.plot(range(len(values)), running_best, "--", lw=1.5, color="#64748B", label="best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Eval return")
        ax.set_title(f"{algo.upper()} Bayesian optimization (Optuna TPE)",
                     fontsize=13, fontweight="bold", loc="left", pad=12)
        ax.legend(loc="lower right", frameon=False)
        plt.tight_layout()
        out = os.path.join(config.FIGURES_DIR, f"tuning_{algo}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  saved → {out}")


def main():
    print("Generating figures...")
    plot_learning_curves()
    plot_eval_curves()
    plot_final_comparison()
    plot_tuning_history()
    print("Done.")


if __name__ == "__main__":
    main()
