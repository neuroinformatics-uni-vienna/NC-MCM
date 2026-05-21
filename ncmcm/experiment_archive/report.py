"""
Markdown report generation for experiment archives.
"""

import datetime
import json
from pathlib import Path


def generate_report(
    manifest: dict,
    experiment_dir: Path,
) -> Path:
    """Write ``reports/experiment_report.md`` and return the path.

    The report is a self-contained plain-language summary of what was trained,
    the main decoding results, and which analysis components are present.
    """
    report_path = Path(experiment_dir) / 'reports' / 'experiment_report.md'
    report_path.parent.mkdir(exist_ok=True)

    model    = manifest.get('model', {})
    training = manifest.get('training', {})
    git      = manifest.get('git', {})
    data_inf = manifest.get('data', {})

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        '# Experiment Report',
        '',
        f"**Experiment ID:** `{manifest.get('experiment_id', '?')}`  ",
        f"**Created:** {manifest.get('created_at', '?')}  ",
        f"**Git commit:** `{git.get('commit_hash', '?')}`  ",
        f"**Branch:** `{git.get('branch', '?')}`  ",
        f"**Git status:** {git.get('status_summary', '?')}  ",
        '',
        '---',
        '',
    ]

    # ── What was trained ──────────────────────────────────────────────────────
    lines += [
        '## What Was Trained',
        '',
        '| Parameter | Value |',
        '|-----------|-------|',
        f"| Session | `{manifest.get('session', '?')}` |",
        f"| Data path | `{manifest.get('data_path', '?')}` |",
        f"| b_mode | `{model.get('b_mode', '?')}` |",
        f"| b_type | `{model.get('b_type', '?')}` |",
        f"| alpha (HGF weight) | `{model.get('alpha', '?')}` |",
        f"| latent_dim | `{model.get('latent_dim', '?')}` |",
        f"| context_policy | `{model.get('context_policy', '?')}` |",
        f"| trial_random_state | `{model.get('trial_random_state', '?')}` |",
        f"| n_epochs | `{model.get('n_epochs', '?')}` |",
        f"| window | `{model.get('window', '?')}` |",
        f"| normalize_method | `{model.get('normalize_method', '?')}` |",
        f"| downsample_fs | `{model.get('downsample_fs', '?')}` Hz |",
        f"| hgf_model | `{model.get('hgf_model', '—')}` |",
        f"| hgf_column | `{model.get('hgf_column', '—')}` |",
        f"| b_labels | {model.get('b_labels', [])} |",
        '',
        f"**Training duration:** {training.get('execution_time', '?')}  ",
        f"**Status:** {training.get('status', '?')}  ",
        '',
        '### Final Training Losses',
        '',
    ]
    losses = training.get('final_losses', {})
    if losses:
        lines += ['| Metric | Value |', '|--------|-------|']
        for k, v in losses.items():
            lines.append(f'| {k} | {_fmt(v)} |')
    else:
        lines.append('_No training loss data available._')

    # ── Data ─────────────────────────────────────────────────────────────────
    lines += ['', '## Data Summary', '']
    b_labels     = model.get('b_labels', [])
    class_counts = data_inf.get('class_counts', {})
    for split, counts in class_counts.items():
        lines.append(f'**{split.capitalize()} set:**')
        for cls_int, count in counts.items():
            cls_name = (
                b_labels[int(cls_int)]
                if b_labels and int(cls_int) < len(b_labels)
                else cls_int
            )
            lines.append(f'  - Class {cls_int} ({cls_name}): {count:,} windows')
    trial_splits = data_inf.get('trial_splits', {})
    if trial_splits:
        lines.append('')
        for split, info in trial_splits.items():
            lines.append(
                f"**{split} trials:** "
                f"{info.get('n_trials', '?')} trials → "
                f"{info.get('n_windows', 0):,} windows"
            )

    # ── Decoding ──────────────────────────────────────────────────────────────
    lines += ['', '## Behavior Decoding Results', '', _decoding_section(experiment_dir)]

    # ── Geometry ──────────────────────────────────────────────────────────────
    geom_dir  = Path(experiment_dir) / 'geometry'
    geom_json = geom_dir / 'pca_summary.json'
    geom_figs = sorted(geom_dir.glob('*.png')) if geom_dir.exists() else []
    lines += ['', '## Latent Space Geometry', '']
    if geom_json.exists():
        try:
            pca = json.loads(geom_json.read_text())
            ev  = pca.get('explained_variance_ratio', [])
            lines.append('**PCA explained variance:**')
            for i, v in enumerate(ev):
                lines.append(f'  - PC{i+1}: {v*100:.2f}%')
            lines.append('')
        except Exception:
            pass
    if geom_figs:
        lines.append(f'Generated {len(geom_figs)} geometry figure(s):')
        for p in geom_figs:
            lines.append(f'  - `{p.name}`')
    else:
        lines.append('_Geometry analysis not run or produced no figures._')

    # ── Time analysis ─────────────────────────────────────────────────────────
    time_dir  = Path(experiment_dir) / 'time_analysis'
    time_json = time_dir / 'time_analysis_summary.json'
    time_figs = sorted(time_dir.glob('*.png')) if time_dir.exists() else []
    lines += ['', '## Time-Resolved Analysis', '']
    if time_json.exists():
        try:
            ts = json.loads(time_json.read_text())
            lines += [
                f"**Overall accuracy:** {_fmt(ts.get('val_acc'))}  ",
                f"**Balanced accuracy:** {_fmt(ts.get('val_balanced_acc'))}  ",
                f"**Majority baseline:** {_fmt(ts.get('majority_baseline'))}  ",
                f"**Has trial IDs:** {ts.get('has_trial_ids', False)}  ",
                '',
            ]
        except Exception:
            pass
    if time_figs:
        lines.append(f'Generated {len(time_figs)} time-analysis figure(s):')
        for p in time_figs:
            lines.append(f'  - `{p.name}`')
    else:
        lines.append('_Time analysis not run or produced no figures._')

    # ── MVE ───────────────────────────────────────────────────────────────────
    mve_status_path = Path(experiment_dir) / 'mve_trial_based' / 'status.json'
    lines += ['', '## Microvariable Evaluation (MVE)', '']
    if mve_status_path.exists():
        try:
            mve = json.loads(mve_status_path.read_text())
            lines += [
                f"**Status:** {mve.get('status', '?')}  ",
                f"**Note:** {mve.get('note', '')}  ",
            ]
        except Exception:
            lines.append('_Could not read MVE status._')
    else:
        lines.append('_MVE not run._')

    # ── Missing / incomplete ──────────────────────────────────────────────────
    missing: list[str] = []
    if not geom_figs:
        missing.append('geometry analysis figures')
    if not time_figs:
        missing.append('time analysis figures')
    if not (Path(experiment_dir) / 'model' / 'bundlenet_model.pt').exists():
        missing.append('model checkpoint copy')
    if not (Path(experiment_dir) / 'behavior_decoding' / 'decoding_summary.json').exists():
        missing.append('behavior decoding results')

    lines += ['', '## Missing / Incomplete', '']
    if missing:
        for m in missing:
            lines.append(f'- {m}')
    else:
        lines.append('_Nothing missing._')

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        '',
        '---',
        f'_Report generated by `ncmcm.experiment_archive.report` at '
        f'{datetime.datetime.now().isoformat()}_',
    ]

    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt(v, digits: int = 4) -> str:
    try:
        return f'{float(v):.{digits}f}'
    except Exception:
        return str(v)


def _decoding_section(experiment_dir: Path) -> str:
    dec_path = Path(experiment_dir) / 'behavior_decoding' / 'decoding_summary.json'
    if not dec_path.exists():
        return '_Behavior decoding not run._'
    try:
        dec = json.loads(dec_path.read_text())
    except Exception:
        return '_Could not read decoding_summary.json._'

    metrics = dec.get('metrics', {})
    lines: list[str] = []

    disc = metrics.get('discrete', {})
    if disc:
        lines += [
            '**Discrete (choice left/right):**',
            f"  - Unweighted acc: {_fmt(disc.get('unweighted_val_acc'))} "
            f"± {_fmt(disc.get('unweighted_val_acc_std'))}",
            f"  - Weighted acc:   {_fmt(disc.get('weighted_val_acc'))} "
            f"± {_fmt(disc.get('weighted_val_acc_std'))}",
            f"  - Chance acc:     {_fmt(disc.get('chance_acc'))}",
            '',
        ]

    hgf = metrics.get('hgf', {})
    if hgf:
        lines += [
            '**HGF belief decoding:**',
            f"  - Val R²:     {_fmt(hgf.get('val_r2_mean'))} ± {_fmt(hgf.get('val_r2_std'))}",
            f"  - Chance R²:  {_fmt(hgf.get('chance_r2_mean'))}",
            '',
        ]

    hybrid = metrics.get('hybrid', {})
    if hybrid:
        lines += [
            '**Hybrid (discrete + continuous):**',
            f"  - Discrete UW acc:   {_fmt(hybrid.get('unweighted_val_acc'))}",
            f"  - Discrete W acc:    {_fmt(hybrid.get('weighted_val_acc'))}",
            f"  - Continuous R² (UW): {_fmt(hybrid.get('unweighted_cont_r2_mean'))}",
            '',
        ]

    return '\n'.join(lines) if lines else '_No metrics available._'
