#!/usr/bin/env bash
set -euo pipefail

# Start a detached tmux session to run the bandit gridsearch and log output.
# Quote the path because it contains spaces
cd "/home/kerim/Projects/Neural Algorithms/NC-MCM"

session="gridsearch_$(date +%Y%m%d_%H%M%S)"
log="results/${session}.log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found, please install tmux"
  exit 1
fi

cmd="'/home/kerim/Projects/Neural Algorithms/NC-MCM/.venv/bin/python' scripts/bandit_gridsearch.py --data_path '/home/kerim/Projects/Neural Algorithms/NC-MCM/datasets/raw/twoArmBandit/JPAS_0023_20230922' --downsample_fs 30 --downsample_method gaussian --good_neurons_only false --apply_hold_transitions none --normalize_method minmax_global --window 50 --latent_dim 3 --batch_size 50 --n_epochs 500 --learning_rate 0.0001 --gamma 0.75 --device cuda --lazy_loading --pca_init --choosing_state_mode side --gaussian_sigma_ms 25.0 --b_mode decision_strict --trial_based --trial_test_ratio 0.2 --hgf_model binary2 --hgf_column x_1_expected_mean --b_type hybrid --alpha 0.1 0.3 0.5 0.7 0.9 --output_dir './results'"

# Start detached tmux session and tee output to a log file under results/
# Use bash -lc to run the command; the python path is single-quoted inside $cmd
tmux new-session -d -s "$session" bash -lc "$cmd 2>&1 | tee \"$log\""

# Record session name for convenience
echo "$session" > results/last_tmux_session.txt

echo "Started tmux session: $session; log: $log"
