#!/usr/bin/env bash
# start_reward_to_choice_decoding.sh
# Run bandit_behaviour_decoding.py on both reward_to_choice run folders.
# Call this AFTER the training tmux session completes.
#
# Usage:
#   bash scripts/start_reward_to_choice_decoding.sh \
#       results/grid_search_20260521_150124_same_partition_reward_to_choice_hybrid_alpha_050/run_20260521_150127 \
#       results/grid_search_20260521_150124_same_partition_reward_to_choice_discrete_only/run_20260521_150127

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

PY=".venv/bin/python"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <hybrid_run_dir> <discrete_run_dir>"
  echo "Example:"
  echo "  $0 results/grid_search_..._hybrid_alpha_050/run_... results/grid_search_..._discrete_only/run_..."
  exit 1
fi

HYBRID_RUN="$1"
DISC_RUN="$2"

for RUN in "$HYBRID_RUN" "$DISC_RUN"; do
  if [ ! -d "$RUN" ]; then
    echo "ERROR: run directory not found: $RUN"
    exit 1
  fi
  if [ ! -f "$RUN/data/latent_trajectories_train.npy" ]; then
    echo "ERROR: latent_trajectories_train.npy missing in $RUN/data/"
    echo "  (training may not have completed)"
    exit 1
  fi
done

echo "Starting decoding on:"
echo "  Hybrid:   $HYBRID_RUN"
echo "  Discrete: $DISC_RUN"
echo ""

for RUN in "$HYBRID_RUN" "$DISC_RUN"; do
  SESSION="decoding_rtc_$(basename "$RUN")_${TIMESTAMP}"
  LOG="$RUN/data/decoding_${TIMESTAMP}.log"
  mkdir -p "$RUN/data/decoding"
  tmux new-session -d -s "$SESSION" \
    bash -lc "'$PY' scripts/bandit_behaviour_decoding.py '$RUN' 2>&1 | tee '$LOG'"
  echo "Started: $SESSION  ->  $LOG"
done

echo ""
echo "Monitor with:  tmux list-sessions"
