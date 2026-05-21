"""
ncmcm.experiment_archive
========================
Utilities for creating structured experiment archives from BunDLe-Net runs.

Public API
----------
generate_experiment_id  — build a human-readable run ID from config + run name
create_experiment_folder — mkdir the standardised subfolder tree, return path dict
build_manifest          — collect all metadata into a manifest dict
save_manifest           — write manifest.json
generate_report         — write reports/experiment_report.md
"""

from .folders import generate_experiment_id, create_experiment_folder, SUBFOLDERS
from .manifest import build_manifest, save_manifest, get_git_info
from .report import generate_report

__all__ = [
    'generate_experiment_id',
    'create_experiment_folder',
    'SUBFOLDERS',
    'build_manifest',
    'save_manifest',
    'get_git_info',
    'generate_report',
]
