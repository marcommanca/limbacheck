#!/usr/bin/env bash
#
# Runs the entire Limbacheck experimental pipeline end to end, in the order
# described in the paper: data prep -> MLM pre-training -> detector fine-tuning
# -> BERT pipeline evaluation -> LLM baseline fine-tuning/evaluation (10
# checkpoints) -> CROS comparison -> final report.
#
# Prerequisites:
#   - Run from the repository root (where this script lives).
#   - Python 3.10+ with `pip install -r requirements.txt` already done.
#   - A CUDA GPU for every stage except clean-corpus/split-corpus/
#     build-vocabulary/prepare-llm-data/evaluate-cros/report (see
#     `python run_experiments.py --list-stages` for the [GPU] tags).
#   - If bitsandbytes can't find your CUDA libs, export LIMBACHECK_CUDA_LIB_PATH
#     before running this script (see src/llm_baselines/train_llm.py).
#
# Usage:
#   ./run_all_experiments.sh
#
# This is a thin wrapper around `python run_experiments.py --stage all`; run
# that directly if you'd rather not use a shell script (e.g. on Windows).
# Re-running this script is safe: train-llm-all skips checkpoints that already
# exist (pass --force via `python run_experiments.py --stage train-llm-all --force`
# to retrain from scratch).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Limbacheck: full experimental pipeline ==="
python run_experiments.py --stage all

echo ""
echo "=== Done. Final comparison table: runs/comparison.{csv,md,tex} ==="
