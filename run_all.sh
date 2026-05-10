#!/usr/bin/env bash
set -e

# ============================================================
# HalfCheetah-v5 Deep RL Pipeline
# Bayesian Hyperparameter Tuning + Final Training + Plotting
# ============================================================

cd "$(dirname "$0")"

echo ""
echo "=========================================="
echo " HalfCheetah-v5 RL Pipeline"
echo "=========================================="
echo ""

# ---------- Step 1: Environment setup ----------
if [ ! -d "venv" ]; then
    echo ">> Creating virtual environment..."
    python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo ">> Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "   Done."
echo ""

START_TIME=$(date +%s)

# ---------- Step 2: Bayesian hyperparameter tuning ----------
echo "=========================================="
echo " STAGE 1/3: Hyperparameter tuning (Optuna)"
echo "=========================================="
python tune.py --algo all

# ---------- Step 3: Final training with best params ----------
echo ""
echo "=========================================="
echo " STAGE 2/3: Final training with best params"
echo "=========================================="
python train.py --algo all

# ---------- Step 4: Generate figures ----------
echo ""
echo "=========================================="
echo " STAGE 3/3: Generating figures"
echo "=========================================="
python plot_results.py

# ---------- Step 5: Final evaluation ----------
echo ""
echo "=========================================="
echo " Final evaluation of trained agents"
echo "=========================================="
python evaluate.py --algo all --seed 0 --episodes 10

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo " Pipeline complete in ${MINUTES}m ${SECONDS}s"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  tuning/   - best hyperparameters per algorithm (JSON)"
echo "  models/   - saved trained models (.zip)"
echo "  logs/     - TensorBoard logs and Monitor CSVs"
echo "  figures/  - all plots (learning curves, comparisons, tuning history)"
echo "  results/  - final_results.json with summary metrics"
echo ""
echo "View training in TensorBoard:"
echo "  tensorboard --logdir logs"
echo ""
echo "Watch a trained agent run:"
echo "  python evaluate.py --algo sac --seed 0 --render"
echo ""
