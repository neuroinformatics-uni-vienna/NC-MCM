#!/usr/bin/env bash
set -euo pipefail

BASE="/home/kerim/Projects/Neural Algorithms/NC-MCM"
RUN_PARENT="$BASE/results/grid_search_20260510_222915"
PY="$BASE/.venv/bin/python"

i=0
for run in "$RUN_PARENT"/run_*; do
  [ -d "$run" ] || continue
  session="decoding_${i}_$(date +%s)"
  mkdir -p "$run/data/decoding"
  log="$run/data/decoding/${session}.log"
  # Quote python path and run path to be robust to spaces
  cmd="'$PY' scripts/bandit_behaviour_decoding.py \"$run\""
  tmux new-session -d -s "$session" bash -lc "$cmd 2>&1 | tee \"$log\""
  echo "Started session: $session -> $log"
  i=$((i+1))
done
