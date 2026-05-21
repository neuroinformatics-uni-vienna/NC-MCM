"""
Experiment folder creation and ID generation.
"""

import re
from pathlib import Path

# Canonical subfolder names — every experiment folder has exactly these.
SUBFOLDERS = [
    'logs',
    'model',
    'data',
    'training',
    'behavior_decoding',
    'time_analysis',
    'geometry',
    'mve_trial_based',
    'reports',
]


def generate_experiment_id(config: dict, run_dir_name: str) -> str:
    """Build a human-readable experiment ID from a run config and folder name.

    Format:
        {session}_{b_mode}_{b_type_and_alpha}_{seed_str}_{timestamp}

    Examples:
        JPAS_0023_20230922_reward_to_choice_hybrid_alpha_050_seed42_20260521_150127
        JPAS_0023_20230922_old_multistate_discrete_seed42_20260519_013620
    """
    # Session from data_path
    data_path = config.get('data_path', '')
    session = Path(data_path).name if data_path else 'unknown_session'

    # b_mode
    b_mode = config.get('b_mode', 'unknown_bmode')

    # b_type + alpha → loss_mode string
    b_type = config.get('b_type', 'unknown_btype')
    alpha = config.get('alpha', None)
    if b_type == 'hybrid' and alpha is not None:
        alpha_int = int(round(float(alpha) * 100))
        loss_mode = f'hybrid_alpha_{alpha_int:03d}'
    else:
        loss_mode = b_type

    # Random seed
    seed = config.get('trial_random_state', None)
    seed_str = f'seed{seed}' if seed is not None else 'seedX'

    # Timestamp from run_dir name (e.g. run_20260521_150127 → 20260521_150127)
    ts_match = re.search(r'(\d{8}_\d{6})', run_dir_name)
    ts = ts_match.group(1) if ts_match else 'ts00000000_000000'

    parts = [session, b_mode, loss_mode, seed_str, ts]
    exp_id = '_'.join(
        p.replace('/', '_').replace('\\', '_').replace(' ', '_')
        for p in parts
    )
    return exp_id


def create_experiment_folder(base_dir: Path, experiment_id: str) -> dict:
    """Create the standard subfolder tree and return a dict of named paths.

    Returns a dict where each key is a subfolder name (with hyphens replaced by
    underscores for Python attribute access) and value is an absolute Path.
    The special key ``'root'`` holds the experiment root directory.
    """
    exp_dir = Path(base_dir) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {'root': exp_dir}
    for sub in SUBFOLDERS:
        p = exp_dir / sub
        p.mkdir(exist_ok=True)
        paths[sub.replace('-', '_')] = p

    return paths
