import torch
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path
import time
import itertools
import gc

from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space, project_into_latent_space_lazy
from ncmcm.bundlenet.utils import (prep_data, timeseries_train_test_split, prep_data_lazy, timeseries_train_test_split_lazy,
                                     timeseries_train_test_split_cv, timeseries_train_test_split_cv_lazy, make_hybrid_b)
from ncmcm.bundlenet.initialisations import pca_initialisation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural_plotly
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser


def parse_args():
    parser = argparse.ArgumentParser(description='Train BunDLeNet on 2-arm bandit task data')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, nargs='+',
                        default=['/home/kerim/Projects/Neural Algorithms/NC-MCM/datasets/raw/twoArmBandit/JPAS_0023_20230922'],
                        help='Path to dataset directory (can specify multiple paths for grid search)')
    parser.add_argument('--downsample_fs', type=int, nargs='+', default=[15],
                        help='Downsampling frequency (can specify multiple values for grid search)')
    parser.add_argument('--downsample_method', type=str, nargs='+', default=['count'],
                        choices=['binary', 'count', 'rate', 'mean', 'gaussian'],
                        help='Downsampling method (can specify multiple values for grid search)')
    parser.add_argument('--good_neurons_only', type=str, nargs='+', default=['false'],
                        choices=['true', 'false'],
                        help='Use only good neurons: true or false (can specify multiple for grid search, e.g., true false)')
    parser.add_argument('--apply_hold_transitions', type=str, nargs='+', default=['none'],
                        choices=['none', 'HOLD_TO_CHOOSING_TRANSITIONS', 'CHOOSING_TO_OUTCOME_TRANSITIONS', 'CHOOSING_TO_CORRECTNESS_TRANSITIONS'],
                        help='State transition mapping to apply: none, HOLD_TO_CHOOSING_TRANSITIONS, CHOOSING_TO_OUTCOME_TRANSITIONS, or CHOOSING_TO_CORRECTNESS_TRANSITIONS (can specify multiple for grid search)')
    parser.add_argument('--normalize_method', type=str, nargs='+', default=['None'],
                        choices=['None', 'minmax', 'minmax_global'],
                        help='Normalization method: None, minmax (per-neuron), or minmax_global (can specify multiple for grid search)')
    
    parser.add_argument('--window', type=int, nargs='+', default=[50],
                        help='Window length for time delay embedding (can specify multiple values for grid search)')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, nargs='+', default=[3],
                        help='Dimensionality of latent space (can specify multiple values for grid search)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, nargs='+', default=[50],
                        help='Batch size for training (can specify multiple values for grid search)')
    parser.add_argument('--n_epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, nargs='+', default=[0.0001],
                        help='Learning rate (can specify multiple values for grid search)')
    parser.add_argument('--gamma', type=float, nargs='+', default=[0.8],
                        help='Weight for behaviour loss (can specify multiple values for grid search)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for training')
    
    # Visualization parameters
    parser.add_argument('--vis_samples', type=int, nargs=2, default=None,
                        metavar=('START', 'END'),
                        help='Range of samples to use for visualization (start end). If not specified, uses full range.')
    parser.add_argument('--recurrence_threshold', type=float, default=None,
                        help='Threshold for recurrence plot. If not specified, recurrence plots will not be generated.')
    
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, 
                        default='./results',
                        help='Base directory for output')
    parser.add_argument('--generate_gif', action='store_true',
                        help='Enable GIF generation for rotating 3D plots')
    parser.add_argument('--generate_3d_html', action='store_true',
                        help='Enable interactive 3D HTML plot generation')
    
    # Memory optimization
    parser.add_argument('--lazy_loading', action='store_true',
                        help='Use lazy loading for memory-efficient data preparation')
    
    # Cross-validation
    parser.add_argument('--cv_folds', type=int, default=None,
                        help='Number of cross-validation folds. If not specified, uses single train/test split.')

    # Train/test split (single-split mode, i.e. cv_folds=None)
    parser.add_argument('--kfold_n_splits', type=int, default=7,
                        help='Number of KFold splits for the single train/test split (default: 7)')
    parser.add_argument('--kfold_test_fold', type=int, default=4,
                        help='Which fold index to use as test set in the single train/test split (default: 4)')
    
    # Behaviour type
    parser.add_argument('--b_type', type=str, nargs='+', default=['discrete'],
                        choices=['discrete', 'continuous', 'hybrid'],
                        help='Type of behaviour variable for BunDLeNet loss (can specify multiple for grid search)')
    
    # HGF parameters (used when b_type contains hybrid)
    parser.add_argument('--hgf_model', type=str, default='binary2',
                        help='HGF model variant to use for belief computation (e.g. binary2, binary3)')
    parser.add_argument('--hgf_column', type=str, default='x_1_expected_mean',
                        help='HGF output column to use as continuous behaviour signal')
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5],
                        help='Discrete CE weight in hybrid loss: alpha*CE + (1-alpha)*MSE. Can specify multiple for grid search.')

    # Model initialisation
    parser.add_argument('--pca_init', action='store_true',
                        help='Initialise BunDLeNet tau with PCA before training (improves reproducibility)')

    # Dataset config (not grid-searchable, but should appear in config for reproducibility)
    parser.add_argument('--choosing_state_mode', type=str, default='side',
                        choices=['side', 'correctness'],
                        help='How choosing states are labelled: side (left/right) or correctness (correct/incorrect)')
    parser.add_argument('--gaussian_sigma_ms', type=float, default=25.0,
                        help='Gaussian kernel sigma in ms (only used when downsample_method=gaussian)')
    parser.add_argument('--recompute_cache', action='store_true',
                        help='Force recompute dataset cache even if one exists')

    return parser.parse_args()


def create_grid_search_directory(base_dir):
    """Create timestamped grid search directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    grid_dir = Path(base_dir) / f'grid_search_{timestamp}'
    grid_dir.mkdir(parents=True, exist_ok=True)
    return grid_dir


def create_run_directory(grid_dir, run_idx, params):
    """Create directory for a specific parameter combination"""
    # Sanitize parameters for use in directory names
    sanitized_params = {}
    for k, v in params.items():
        if k == 'data_path':
            # Extract just the last directory name from the path
            v = Path(v).name
        # Replace problematic characters
        v_str = str(v).replace('/', '_').replace('\\', '_').replace(':', '_')
        sanitized_params[k] = v_str
    
    param_str = '_'.join([f"{k}={v}" for k, v in sanitized_params.items()])
    run_dir = grid_dir / f'run_{run_idx:03d}_{param_str}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / 'figures').mkdir(exist_ok=True)
    (run_dir / 'model').mkdir(exist_ok=True)
    (run_dir / 'data').mkdir(exist_ok=True)
    
    return run_dir


def save_config(args, output_dir, extra=None):
    """Save configuration to JSON file.

    Args:
        args: parsed argument namespace
        output_dir: Path to output directory
        extra: optional dict of additional runtime fields to merge in
    """
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    if extra:
        config.update(extra)
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")


def save_comprehensive_config(args, params, output_dir, execution_time, execution_time_seconds, metrics, cv_summary=None, start_timestamp=None, error=None):
    """Save comprehensive configuration with execution results and metrics
    
    Args:
        args: Argument parser namespace with all configuration
        params: Dictionary of actual parameter values used for this run
        output_dir: Path to output directory
        execution_time: Formatted execution time string
        execution_time_seconds: Execution time in seconds
        metrics: Dictionary of performance metrics
        cv_summary: Optional CV summary dictionary (for CV runs)
        start_timestamp: Optional start timestamp (ISO format)
        error: Optional error message if run failed
    """
    # Build comprehensive config matching grid search summary structure
    comprehensive_config = {
        # Metadata
        'start_timestamp': start_timestamp if start_timestamp else datetime.now().isoformat(),
        'completed_at': datetime.now().isoformat(),
        'execution_time': execution_time,
        'execution_time_seconds': execution_time_seconds,
        'status': 'failed' if error else 'completed',
        'error': error,
        
        # Parameters used (actual values, not lists)
        'parameters': params,
        
        # Full configuration (includes all args, with lists for grid search parameters)
        'full_configuration': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        
        # Output paths
        'output_dir': str(output_dir),
        
        # Performance metrics
        'metrics': metrics
    }
    
    # Add CV summary if available
    if cv_summary is not None:
        comprehensive_config['cv_summary'] = cv_summary
    
    # Save to JSON
    config_path = output_dir / 'run_summary.json'
    with open(config_path, 'w') as f:
        json.dump(comprehensive_config, f, indent=4)
    
    print(f"Comprehensive configuration saved to {config_path}")


def load_data(data_path, downsample_fs, downsample_method, good_neurons_only, apply_hold_transitions='none', normalize_method='none',
              hgf_model=None, hgf_column=None, choosing_state_mode='side',
              gaussian_sigma_ms=25.0, recompute_cache=False):
    """Load and prepare dataset"""
    # Determine state_transitions parameter
    transition_lookup = {
        'none': None,
        'false': None,  # backward compatibility for boolean flags
        'true': BanditTaskNeuroPixelsDataset.HOLD_TO_CHOOSING_TRANSITIONS,
        'hold_to_choosing_transitions': BanditTaskNeuroPixelsDataset.HOLD_TO_CHOOSING_TRANSITIONS,
        'HOLD_TO_CHOOSING_TRANSITIONS': BanditTaskNeuroPixelsDataset.HOLD_TO_CHOOSING_TRANSITIONS,

        # Descriptive transition sets
        'choosing_to_outcome_transitions': BanditTaskNeuroPixelsDataset.CHOOSING_TO_OUTCOME_TRANSITIONS,
        'CHOOSING_TO_OUTCOME_TRANSITIONS': BanditTaskNeuroPixelsDataset.CHOOSING_TO_OUTCOME_TRANSITIONS,
        'choosing_to_correctness_transitions': BanditTaskNeuroPixelsDataset.CHOOSING_TO_CORRECTNESS_TRANSITIONS,
        'CHOOSING_TO_CORRECTNESS_TRANSITIONS': BanditTaskNeuroPixelsDataset.CHOOSING_TO_CORRECTNESS_TRANSITIONS,

        # Backward compatibility aliases for deprecated names
        'choosing_to_reward_transitions': BanditTaskNeuroPixelsDataset.CHOOSING_TO_OUTCOME_TRANSITIONS,
        'CHOOSING_TO_REWARD_TRANSITIONS': BanditTaskNeuroPixelsDataset.CHOOSING_TO_OUTCOME_TRANSITIONS,
        'choosing_to_reward_2_transitions': BanditTaskNeuroPixelsDataset.CHOOSING_TO_CORRECTNESS_TRANSITIONS,
        'CHOOSING_TO_REWARD_2_TRANSITIONS': BanditTaskNeuroPixelsDataset.CHOOSING_TO_CORRECTNESS_TRANSITIONS,
    }

    transition_key = (apply_hold_transitions or 'none')
    # Normalize key for lookup while still accepting exact class variable names
    transition_key_lower = transition_key.lower() if isinstance(transition_key, str) else 'none'
    state_transitions = transition_lookup.get(transition_key, transition_lookup.get(transition_key_lower))

    if isinstance(state_transitions, type(None)) and transition_key not in transition_lookup and transition_key_lower not in transition_lookup:
        raise ValueError(f"Unsupported apply_hold_transitions value: {apply_hold_transitions}")
    
    # Convert 'None' string to None for normalize_method
    norm_method = None if normalize_method == 'None' else normalize_method
    
    print("Loading dataset...")
    
    if state_transitions is not None:
        print(f"  Applying state transitions: {transition_key}")
    
    if norm_method is not None:
        print(f"  Applying normalization: {norm_method}")
    
    dataset = BanditTaskNeuroPixelsDataset(
        data_path=data_path, 
        downsample_fs=downsample_fs, 
        downsample_method=downsample_method,
        good_neurons_only=good_neurons_only,
        state_transitions=state_transitions,
        normalize_method=norm_method,
        choosing_state_mode=choosing_state_mode,
        gaussian_sigma_ms=gaussian_sigma_ms,
        recompute_cache=recompute_cache,
        **(dict(hgf_model=hgf_model, hgf_column=hgf_column) if hgf_model is not None else {})
    )
    # Use float32 to reduce memory usage by 50%
    x = dataset.x.T.toarray().astype(np.float32)
    b = dataset.b.toarray().flatten()
    b_labels = dataset.b_labels
    b_colors = dataset.get_color_map_for_plotting()
    b_colors_rgb = dataset.get_rgb_colors_for_visualizer()
    hgf_beliefs = getattr(dataset, 'hgf_beliefs', None)  # None if HGF was not loaded
    
    # Free memory from dataset object (contains large sparse matrices)
    del dataset
    gc.collect()
    
    print(f"Data shapes - x: {x.shape}, b: {b.shape}")
    print(f"behaviour labels: {b_labels}")
    
    return x, b, b_labels, b_colors, b_colors_rgb, hgf_beliefs


def preprocess_data(x, b, window, lazy_loading=False, cv_folds=None, kfold_n_splits=7, kfold_test_fold=4):
    """Preprocess data for BunDLeNet
    
    Args:
        x: Neural data
        b: behaviour labels
        window: Window size for time delay embedding
        lazy_loading: Use memory-efficient lazy loading
        cv_folds: Number of CV folds (None for single split)
        kfold_n_splits: Number of KFold splits for single train/test split (default: 7)
        kfold_test_fold: Which fold index to use as test set in single split (default: 4)
    
    Returns:
        If cv_folds is None:
            x_, b_, x_train, x_test, b_train, b_test
        If cv_folds is specified:
            x_, b_, splits (list of tuples)
    """
    print("Preprocessing data...")
    
    # Prepare data with time delay embedding
    if lazy_loading:
        print("Using lazy loading (memory-efficient)...")
        x_, b_ = prep_data_lazy(x, b, win=window)
        print(f"Lazy dataset created - shape: {x_.shape}, b_: {b_.shape}")
    else:
        x_, b_ = prep_data(x, b, win=window)
        print(f"Prepared data shapes - x_: {x_.shape}, b_: {b_.shape}")
    
    # Train/test split or CV splits
    if cv_folds is not None:
        print(f"Creating {cv_folds}-fold cross-validation splits...")
        if lazy_loading:
            splits = timeseries_train_test_split_cv_lazy(x_, b_, n_splits=cv_folds)
        else:
            splits = timeseries_train_test_split_cv(x_, b_, n_splits=cv_folds)
        
        print(f"Created {len(splits)} CV splits")
        for i, (x_train, x_test, b_train, b_test) in enumerate(splits):
            train_size = x_train.shape[0] if hasattr(x_train, 'shape') else len(x_train)
            test_size = x_test.shape[0] if hasattr(x_test, 'shape') else len(x_test)
            print(f"  Fold {i}: train={train_size}, test={test_size}")
        
        return x_, b_, splits
    else:
        # Single train/test split using configurable KFold
        from torch.utils.data import Subset
        total_samples = len(x_) if not hasattr(x_, 'shape') else x_.shape[0]
        indices = np.arange(total_samples)
        kf = KFold(n_splits=kfold_n_splits, shuffle=False)
        for i, (train_index, test_index) in enumerate(kf.split(indices)):
            if i == kfold_test_fold:
                if lazy_loading:
                    x_train = Subset(x_, train_index)
                    x_test = Subset(x_, test_index)
                else:
                    x_train, x_test = x_[train_index], x_[test_index]
                b_train = b_[train_index]
                b_test = b_[test_index]
                break
        
        train_size = len(x_train) if not hasattr(x_train, 'shape') else x_train.shape[0]
        test_size = len(x_test) if not hasattr(x_test, 'shape') else x_test.shape[0]
        print(f"KFold({kfold_n_splits}) fold-{kfold_test_fold} split: train={train_size}, test={test_size}")
        
        return x_, b_, x_train, x_test, b_train, b_test


def project_latent_space(x_data, model, lazy_loading=False):
    """Project data into latent space using appropriate method"""
    if lazy_loading:
        print("Using lazy projection (memory-efficient)...")
        return project_into_latent_space_lazy(x_data, model)
    else:
        return project_into_latent_space(x_data, model)


def train_bundlenet(x_train, b_train, x_test, b_test, x_shape, args, output_dir, n_classes=None):
    """Train BunDLeNet model"""
    print("Initializing BunDLeNet model...")

    b_type = getattr(args, 'b_type', 'discrete')
    alpha = getattr(args, 'alpha', 0.5)
    pca_init = getattr(args, 'pca_init', False)

    if b_type == 'hybrid':
        if n_classes is None:
            raise ValueError("n_classes must be provided for b_type='hybrid'")
        # Output head: n_classes logits + n_continuous outputs (cols 1+ of b_train)
        n_continuous = b_train.shape[1] - 1
        num_behaviour = n_classes + n_continuous
    elif b_type == 'continuous':
        num_behaviour = b_train.shape[1] if b_train.ndim > 1 else 1
    else:  # discrete
        num_behaviour = len(np.unique(b_train))
    
    model = BunDLeNet(
        latent_dim=args.latent_dim,
        num_behaviour=num_behaviour,
        input_shape=x_shape
    )

    device = torch.device(args.device)

    if pca_init:
        print("Running PCA initialisation of tau...")
        # pca_initialisation requires eager numpy array — materialise if lazy
        if hasattr(x_train, 'dataset'):  # Subset (lazy)
            from torch.utils.data import DataLoader as _DL
            _dl = _DL(x_train, batch_size=len(x_train), shuffle=False)
            x_train_np = next(iter(_dl)).numpy()
        else:
            x_train_np = x_train if isinstance(x_train, np.ndarray) else np.array(x_train)
        model.tau = pca_initialisation(x_train_np, model.tau, args.latent_dim, device)
        print("PCA initialisation complete.")

    print(f"Training on {device} for {args.n_epochs} epochs (b_type='{b_type}', pca_init={pca_init})...")
    
    loss_array, test_loss_array = train_model(
        x_train,
        b_train,
        model,
        b_type=b_type,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        device=device,
        validation_data=(x_test, b_test),
        n_classes=n_classes,
        alpha=alpha,
    )
    
    # Save model
    model_path = output_dir / 'model' / 'bundlenet_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save loss arrays
    np.save(output_dir / 'data' / 'loss_array.npy', loss_array)
    np.save(output_dir / 'data' / 'test_loss_array.npy', test_loss_array)
    if loss_array.shape[1] == 5:
        np.save(output_dir / 'data' / 'disc_loss_array.npy', loss_array[:, 3])
        np.save(output_dir / 'data' / 'cont_loss_array.npy', loss_array[:, 4])
        np.save(output_dir / 'data' / 'disc_test_loss_array.npy', test_loss_array[:, 3])
        np.save(output_dir / 'data' / 'cont_test_loss_array.npy', test_loss_array[:, 4])
    
    return model, loss_array, test_loss_array


def plot_training_loss(loss_array, test_loss_array, output_dir):
    """Plot and save training and validation loss curves"""
    print("Plotting training loss...")
    is_hybrid = loss_array.shape[1] == 5
    n_plots = 5 if is_hybrid else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    
    labels = [
        r"$\mathcal{L}_{\mathrm{Markov}}$",
        r"$\mathcal{L}_{\mathrm{Behaviour}}$",
        r"Total loss $\mathcal{L}$"
    ]
    if is_hybrid:
        labels += [
            r"$\mathcal{L}_{\mathrm{Discrete}}$ (CE component)",
            r"$\mathcal{L}_{\mathrm{Continuous}}$ (MSE component)"
        ]
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.semilogy(loss_array[:, i], label='Train', linewidth=2)
        ax.semilogy(test_loss_array[:, i], label='Test', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'figures' / 'training_loss.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training loss plot saved to {plot_path}")


def visualize_neural_behavioural(x, b, b_labels, b_colors, output_dir):
    """Visualize neural activity and behavioural choices"""
    print("Plotting neural-behavioural data...")
    fig = plotting_neuronal_behavioural_plotly(x, b, b_names=b_labels, b_colors=b_colors, show_fig=False)
    
    plot_path = output_dir / 'figures' / 'neural_behavioural_overview.html'
    fig.write_html(str(plot_path))
    
    print(f"Neural-behavioural plot saved to {plot_path}")


def visualize_latent_space(y, b, b_labels, output_dir, vis_range, data_split='train', generate_gif=False, generate_3d_html=False, colors=None, continuous_vars=None):
    """Create all latent space visualizations
    
    Args:
        continuous_vars: Optional dict of {label: array} for continuous-variable coloured
                         phase space plots (e.g. {'HGF belief': hgf_beliefs_split}).
    """
    print(f"Creating latent space visualizations for {data_split} data...")
    
    # Extract range
    if vis_range is None:
        start, end = 0, len(y)
        print(f"  Using full range: 0 to {end}")
    else:
        start, end = vis_range
        start = max(0, start)
        end = min(end, len(y))
        print(f"  Using samples from {start} to {end}")
    
    y_vis = y[start:end]
    b_vis = b[start:end]
    
    # Save latent trajectories and behaviour labels with split name
    np.save(output_dir / 'data' / f'latent_trajectories_{data_split}.npy', y)
    np.save(output_dir / 'data' / f'behaviour_labels_{data_split}.npy', b)

    # Save continuous variables (e.g. HGF belief) when present
    if continuous_vars is not None:
        for var_name, var_array in continuous_vars.items():
            safe_name = var_name.lower().replace(' ', '_')
            np.save(output_dir / 'data' / f'{safe_name}_{data_split}.npy', var_array)

    vis = LatentSpaceVisualiser(y_vis, b_vis, b_labels, show_points=True, colors=colors)
    
    # Time series plot
    print("  - Latent time series...")
    vis.plot_latent_timeseries_plotly(
        show_fig=False,
        filename=str(output_dir / 'figures' / f'latent_time_series_{data_split}.html')
    )
    
    # Phase space plots from multiple perspectives
    print("  - Phase space dynamics (multiple views)...")
    phase_space_views = [
        # ((0, 180), "back"),    # redundant with front
        # ((-90, 0), "bottom"),  # redundant with top
        ((0, 0), "front"),
        ((90, 0), "top"),
        ((0, 90), "right"),
        # ((0, -90), "left"),    # redundant with right
    ]
    
    for (elev, azim), view_name in phase_space_views:
        fig, ax = vis.plot_phase_space(
            show_fig=False,
            axis_view=(elev, azim),
            filename=str(output_dir / 'figures' / f'phase_space_dynamics_{data_split}_{view_name}.png')
        )
        plt.close(fig)
    
    # Rotating 3D plot (GIF)
    if generate_gif:
        print("  - Rotating 3D plot...")
        fig, ax = vis.rotating_plot(
            show_fig=False,
            filename=str(output_dir / 'figures' / f'rotation_3d_{data_split}.gif')
        )
        plt.close(fig)
    else:
        print("  - Skipping rotating 3D plot (GIF generation disabled)")
    
    # Interactive 3D plot (HTML)
    if generate_3d_html:
        print("  - Interactive 3D plot...")
        _ = vis.plot_interactive_3d(
            show_fig=False,
            filename=str(output_dir / 'figures' / f'interactive_3d_{data_split}.html')
        )
    else:
        print("  - Skipping interactive 3D plot (HTML generation disabled)")

    # Continuous-variable phase space plots
    if continuous_vars:
        print("  - Continuous-variable phase space plots...")
        for var_label, c_array in continuous_vars.items():
            c_vis = np.asarray(c_array)[start:end]
            safe_label = var_label.replace(' ', '_').replace('/', '_')
            for (elev, azim), view_name in phase_space_views:
                fig, ax = vis.plot_phase_space_continuous(
                    c=c_vis,
                    label=var_label,
                    show_fig=False,
                    axis_view=(elev, azim),
                    filename=str(output_dir / 'figures' / f'phase_space_continuous_{data_split}_{safe_label}_{view_name}.png')
                )
                plt.close(fig)


def plot_recurrence(y, output_dir, threshold, vis_range, data_split='train'):
    """Generate recurrence plot"""
    print(f"Creating recurrence plot for {data_split} data...")
    
    if vis_range is None:
        start, end = 0, len(y)
    else:
        start, end = vis_range
        start = max(0, start)
        end = min(end, len(y))
    
    y_subset = y[start:end]
    
    # Compute pairwise distances
    pd_y = np.linalg.norm(y_subset[:, np.newaxis] - y_subset, axis=-1) < threshold
    
    # Create a fresh figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(pd_y, cmap='Greys')
    ax.set_title(f'Recurrence Plot - {data_split.capitalize()} (threshold={threshold}, samples {start}-{end})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Time')
    
    plot_path = output_dir / 'figures' / f'recurrence_plot_{data_split}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the specific figure
    print(f"Recurrence plot saved to {plot_path}")


def format_elapsed_time(seconds):
    """Format elapsed time as HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


def print_step_time(step_name, start_time, step_start_time):
    """Print timing information for a step"""
    step_duration = time.time() - step_start_time
    total_elapsed = time.time() - start_time
    print(f"  ✓ {step_name} completed in {format_elapsed_time(step_duration)}")
    print(f"  Total elapsed time: {format_elapsed_time(total_elapsed)}\n")


def generate_param_combinations(args):
    """Generate all parameter combinations for grid search"""
    # Parameters to grid search over
    param_grid = {
        'data_path': args.data_path,
        'downsample_fs': args.downsample_fs,
        'downsample_method': args.downsample_method,
        'good_neurons_only': [x.lower() == 'true' for x in args.good_neurons_only],
        'apply_hold_transitions': args.apply_hold_transitions,
        'normalize_method': args.normalize_method,
        'window': args.window,
        'latent_dim': args.latent_dim,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'b_type': args.b_type,
        'alpha': args.alpha,
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    return combinations, param_grid


def initialize_grid_search_summary(grid_dir, combinations, param_grid):
    """Initialize grid search summary JSON file"""
    summary = {
        'total_runs': len(combinations),
        'start_timestamp': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat(),
        'completed_runs': 0,
        'failed_runs': 0,
        'grid_parameters': param_grid,
        'runs': []
    }
    
    summary_path = grid_dir / 'grid_search_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Grid search summary initialized at {summary_path}")
    return summary_path


def update_grid_search_summary(summary_path, run_idx, params, result):
    """Update grid search summary with a completed run"""
    # Load existing summary
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Add run information
    run_info = {
        'run_id': run_idx,
        'parameters': params,
        'execution_time': result.get('execution_time', None),
        'execution_time_seconds': result.get('execution_time_seconds', None),
        'output_dir': str(result.get('output_dir', '')),
        'status': 'failed' if 'error' in result else 'completed',
        'error': result.get('error', None),
        'completed_at': datetime.now().isoformat()
    }
    # Attach metrics if available
    if 'metrics' in result and isinstance(result['metrics'], dict):
        run_info['metrics'] = result['metrics']
    summary['runs'].append(run_info)
    
    # Update counters
    summary['last_updated'] = datetime.now().isoformat()
    summary['completed_runs'] = sum(1 for r in summary['runs'] if r['status'] == 'completed')
    summary['failed_runs'] = sum(1 for r in summary['runs'] if r['status'] == 'failed')
    
    # Compute best runs based on validation (test) losses if metrics exist
    def _best_by_key(key_name):
        candidates = [r for r in summary['runs'] if r.get('metrics') and r['status'] == 'completed']
        if not candidates:
            return None
        
        # Handle both CV and non-CV runs
        valid_candidates = []
        for r in candidates:
            metrics = r['metrics']
            if metrics.get('cv_mode', False):
                # CV run - use mean metric
                mean_key = key_name + '_mean'
                if mean_key in metrics:
                    valid_candidates.append((r, metrics[mean_key]))
            else:
                # Non-CV run - use direct metric
                if key_name in metrics:
                    valid_candidates.append((r, metrics[key_name]))
        
        if not valid_candidates:
            return None
        
        best_run, best_loss = min(valid_candidates, key=lambda x: x[1])
        result = {
            'run_id': best_run['run_id'],
            'loss': best_loss,
            'parameters': best_run['parameters'],
            'output_dir': best_run['output_dir'],
            'cv_mode': best_run['metrics'].get('cv_mode', False)
        }
        
        # Add epoch info for non-CV runs
        if not result['cv_mode']:
            epoch_key = key_name.replace('loss', 'epoch')
            result['epoch'] = best_run['metrics'].get(epoch_key, None)
        else:
            # Add CV-specific info
            result['std'] = best_run['metrics'].get(key_name.replace('_loss', '_loss_std'), None)
            result['n_folds'] = best_run['metrics'].get('n_folds', None)
        
        return result

    best_markov = _best_by_key('best_markovian_loss')
    best_behaviour = _best_by_key('best_behaviour_loss')

    if best_markov is not None:
        summary['best_markovian_run'] = best_markov
    if best_behaviour is not None:
        summary['best_behaviour_run'] = best_behaviour

    # Save updated summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Summary updated: {summary['completed_runs']}/{summary['total_runs']} completed, {summary['failed_runs']} failed")


def run_single_fold(fold_idx, x_train, x_test, b_train, b_test, x_shape, b_labels, b_colors_rgb, args, fold_dir, run_start_time, n_classes=None):
    """Train and evaluate model on a single fold
    
    Returns:
        dict with metrics and paths
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx}")
    print(f"{'='*70}\n")
    
    # Train model
    step_start = time.time()
    model, loss_array, test_loss_array = train_bundlenet(
        x_train, b_train, x_test, b_test, x_shape, args, fold_dir, n_classes=n_classes
    )
    print_step_time(f"Fold {fold_idx} - Model training", run_start_time, step_start)
    
    # Plot training loss
    step_start = time.time()
    plot_training_loss(loss_array, test_loss_array, fold_dir)
    print_step_time(f"Fold {fold_idx} - Training loss plotting", run_start_time, step_start)
    
    # Project training data into latent space
    step_start = time.time()
    print("Projecting training data into latent space...")
    y_train = project_latent_space(x_train, model, lazy_loading=args.lazy_loading)
    print_step_time(f"Fold {fold_idx} - Training latent space projection", run_start_time, step_start)
    
    # Visualize training latent space
    step_start = time.time()
    b_train_vis = b_train[:, 0].astype(int) if hasattr(b_train, 'ndim') and b_train.ndim > 1 else b_train
    continuous_train = {'HGF belief': b_train[:, 1]} if hasattr(b_train, 'ndim') and b_train.ndim > 1 else None
    visualize_latent_space(y_train, b_train_vis, b_labels, fold_dir, args.vis_samples, 
                          data_split='train', generate_gif=args.generate_gif, generate_3d_html=args.generate_3d_html, colors=b_colors_rgb,
                          continuous_vars=continuous_train)
    print_step_time(f"Fold {fold_idx} - Training latent space visualization", run_start_time, step_start)
    
    # Training recurrence plot
    if args.recurrence_threshold is not None:
        step_start = time.time()
        plot_recurrence(y_train, fold_dir, args.recurrence_threshold, args.vis_samples, data_split='train')
        print_step_time(f"Fold {fold_idx} - Training recurrence plot", run_start_time, step_start)
    
    # Free memory from training data
    del y_train, x_train, b_train
    gc.collect()
    
    # Project validation data into latent space
    step_start = time.time()
    print("Projecting validation data into latent space...")
    y_test = project_latent_space(x_test, model, lazy_loading=args.lazy_loading)
    print_step_time(f"Fold {fold_idx} - Validation latent space projection", run_start_time, step_start)
    
    # Visualize validation latent space
    step_start = time.time()
    b_test_vis = b_test[:, 0].astype(int) if hasattr(b_test, 'ndim') and b_test.ndim > 1 else b_test
    continuous_test = {'HGF belief': b_test[:, 1]} if hasattr(b_test, 'ndim') and b_test.ndim > 1 else None
    visualize_latent_space(y_test, b_test_vis, b_labels, fold_dir, args.vis_samples,
                          data_split='validation', generate_gif=args.generate_gif, generate_3d_html=args.generate_3d_html, colors=b_colors_rgb,
                          continuous_vars=continuous_test)
    print_step_time(f"Fold {fold_idx} - Validation latent space visualization", run_start_time, step_start)
    
    # Validation recurrence plot
    if args.recurrence_threshold is not None:
        step_start = time.time()
        plot_recurrence(y_test, fold_dir, args.recurrence_threshold, args.vis_samples, data_split='validation')
        print_step_time(f"Fold {fold_idx} - Validation recurrence plot", run_start_time, step_start)
    
    # Collect metrics
    fold_metrics = {
        'fold': fold_idx,
        'best_markovian_loss': float(np.min(test_loss_array[:, 0])),
        'best_markovian_epoch': int(np.argmin(test_loss_array[:, 0])),
        'best_behaviour_loss': float(np.min(test_loss_array[:, 1])),
        'best_behaviour_epoch': int(np.argmin(test_loss_array[:, 1])),
        'final_markovian_loss': float(test_loss_array[-1, 0]),
        'final_behaviour_loss': float(test_loss_array[-1, 1]),
        'final_total_loss': float(test_loss_array[-1, 2])
    }
    
    # Free memory
    del y_test, x_test, b_test, model, loss_array, test_loss_array
    gc.collect()
    
    return fold_metrics


def generate_cv_summary(fold_metrics_list, output_dir):
    """Generate summary statistics across CV folds"""
    print("\nGenerating cross-validation summary...")
    
    # Aggregate metrics
    metrics_names = ['best_markovian_loss', 'best_behaviour_loss', 
                    'final_markovian_loss', 'final_behaviour_loss', 'final_total_loss']
    
    cv_summary = {
        'n_folds': len(fold_metrics_list),
        'fold_metrics': fold_metrics_list,
        'aggregated': {}
    }
    
    for metric in metrics_names:
        values = [fold[metric] for fold in fold_metrics_list]
        cv_summary['aggregated'][metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values
        }
    
    # Identify best fold by validation loss
    best_fold_idx = int(np.argmin([fold['best_markovian_loss'] for fold in fold_metrics_list]))
    cv_summary['best_fold'] = {
        'fold_index': best_fold_idx,
        'metrics': fold_metrics_list[best_fold_idx]
    }
    
    # Save summary
    summary_path = output_dir / 'cv_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=4)
    
    print(f"CV Summary saved to {summary_path}")
    print(f"\nCross-Validation Results ({len(fold_metrics_list)} folds):")
    print(f"  Best Markovian Loss: {cv_summary['aggregated']['best_markovian_loss']['mean']:.6f} ± {cv_summary['aggregated']['best_markovian_loss']['std']:.6f}")
    print(f"  Best behaviour Loss:  {cv_summary['aggregated']['best_behaviour_loss']['mean']:.6f} ± {cv_summary['aggregated']['best_behaviour_loss']['std']:.6f}")
    print(f"  Best Fold: {best_fold_idx} (Markovian Loss: {fold_metrics_list[best_fold_idx]['best_markovian_loss']:.6f})")
    
    return cv_summary


def run_single_experiment(args, params, output_dir, run_idx, total_runs):
    """Run a single experiment with specific parameters"""
    run_start_time = time.time()
    start_timestamp = datetime.now().isoformat()
    
    print(f"\n{'='*80}")
    print(f"RUN {run_idx + 1}/{total_runs}")
    print(f"Parameters: {params}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Update args with current parameter combination
    for key, value in params.items():
        setattr(args, key, value)
    
    # Save initial configuration (for crash recovery)
    step_start = time.time()
    save_config(args, output_dir)
    print_step_time("Initial configuration saved", run_start_time, step_start)
    
    # Load data
    step_start = time.time()
    b_type = getattr(args, 'b_type', 'discrete')
    x, b, b_labels, b_colors, b_colors_rgb, hgf_beliefs = load_data(
        args.data_path, args.downsample_fs, args.downsample_method, args.good_neurons_only,
        args.apply_hold_transitions, args.normalize_method,
        hgf_model=args.hgf_model if b_type == 'hybrid' else None,
        hgf_column=args.hgf_column if b_type == 'hybrid' else None,
        choosing_state_mode=getattr(args, 'choosing_state_mode', 'side'),
        gaussian_sigma_ms=getattr(args, 'gaussian_sigma_ms', 25.0),
        recompute_cache=getattr(args, 'recompute_cache', False),
    )
    print_step_time("Data loading", run_start_time, step_start)
    
    # Encode behaviour labels and optionally build hybrid b array
    label_encoder = LabelEncoder()
    b_encoded = label_encoder.fit_transform(b)   # int64 class indices
    n_classes_actual = len(np.unique(b_encoded))

    if b_type == 'hybrid':
        if hgf_beliefs is None:
            raise ValueError(
                "b_type='hybrid' requires HGF beliefs but the dataset returned none. "
                "Pass --hgf_model and --hgf_column to enable HGF loading."
            )
        b_for_bundlenet = make_hybrid_b(b_encoded, hgf_beliefs)
        print(f"Hybrid b array shape: {b_for_bundlenet.shape} (col 0 = class index, cols 1+ = continuous)")
        n_classes = n_classes_actual
    else:
        b_for_bundlenet = b_encoded
        n_classes = None  # not needed for non-hybrid

    # Re-save config now that we know runtime data dimensions
    save_config(args, output_dir, extra={
        'n_timesteps': int(x.shape[1]),
        'n_neurons': int(x.shape[0]),
        'n_classes': n_classes_actual,
        'b_labels': b_labels,
    })
    
    # Visualize raw data
    step_start = time.time()
    visualize_neural_behavioural(x, b, b_labels, b_colors, output_dir)
    print_step_time("Neural-behavioural visualization", run_start_time, step_start)
    
    # Preprocess data
    step_start = time.time()
    preprocess_result = preprocess_data(x, b_for_bundlenet, args.window, lazy_loading=args.lazy_loading, cv_folds=args.cv_folds, kfold_n_splits=args.kfold_n_splits, kfold_test_fold=args.kfold_test_fold)
    
    if args.cv_folds is not None:
        x_, b_, splits = preprocess_result
        print_step_time("Data preprocessing (CV mode)", run_start_time, step_start)
    else:
        x_, b_, x_train, x_test, b_train, b_test = preprocess_result
        print_step_time("Data preprocessing", run_start_time, step_start)
    
    # Free memory from original arrays
    del x, b, b_for_bundlenet
    
    # Cross-validation or single run
    if args.cv_folds is not None:
        # === CROSS-VALIDATION MODE ===
        print(f"\nRunning {args.cv_folds}-fold cross-validation...")
        fold_metrics_list = []
        
        for fold_idx, (x_train, x_test, b_train, b_test) in enumerate(splits):
            # Create fold directory
            fold_dir = output_dir / f'fold_{fold_idx}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / 'figures').mkdir(exist_ok=True)
            (fold_dir / 'model').mkdir(exist_ok=True)
            (fold_dir / 'data').mkdir(exist_ok=True)
            
            # Run fold experiment
            fold_metrics = run_single_fold(
                fold_idx, x_train, x_test, b_train, b_test, x_.shape,
                b_labels, b_colors_rgb, args, fold_dir, run_start_time, n_classes=n_classes
            )
            fold_metrics_list.append(fold_metrics)
        
        # Free memory from preprocessed data
        del x_, b_, splits
        gc.collect()
        
        # Generate CV summary
        cv_summary = generate_cv_summary(fold_metrics_list, output_dir)
        
        # Use CV aggregated metrics for grid search summary
        metrics = {
            'cv_mode': True,
            'n_folds': args.cv_folds,
            'best_markovian_loss_mean': cv_summary['aggregated']['best_markovian_loss']['mean'],
            'best_markovian_loss_std': cv_summary['aggregated']['best_markovian_loss']['std'],
            'best_behaviour_loss_mean': cv_summary['aggregated']['best_behaviour_loss']['mean'],
            'best_behaviour_loss_std': cv_summary['aggregated']['best_behaviour_loss']['std'],
            'best_fold': cv_summary['best_fold']['fold_index'],
            'best_fold_markovian_loss': cv_summary['best_fold']['metrics']['best_markovian_loss']
        }
        
        # Calculate execution time for comprehensive config
        total_time = time.time() - run_start_time
        
        # Save comprehensive configuration with CV summary
        save_comprehensive_config(
            args=args,
            params=params,
            output_dir=output_dir,
            execution_time=format_elapsed_time(total_time),
            execution_time_seconds=total_time,
            metrics=metrics,
            cv_summary=cv_summary,
            start_timestamp=start_timestamp,
            error=None
        )
        
    else:
        # === SINGLE TRAIN/TEST SPLIT MODE (original behaviour) ===
        # Train model
        step_start = time.time()
        model, loss_array, test_loss_array = train_bundlenet(
            x_train, b_train, x_test, b_test, x_.shape, args, output_dir, n_classes=n_classes
        )
        print_step_time("Model training", run_start_time, step_start)
        
        # Free memory from full preprocessed data
        del x_, b_
        
        # Plot training loss
        step_start = time.time()
        plot_training_loss(loss_array, test_loss_array, output_dir)
        print_step_time("Training loss plotting", run_start_time, step_start)
        
        # Project training data into latent space
        step_start = time.time()
        print("Projecting training data into latent space...")
        y_train = project_latent_space(x_train, model, lazy_loading=args.lazy_loading)
        print_step_time("Training latent space projection", run_start_time, step_start)
        
        # Visualize training latent space
        step_start = time.time()
        b_train_vis = b_train[:, 0].astype(int) if hasattr(b_train, 'ndim') and b_train.ndim > 1 else b_train
        continuous_train = {'HGF belief': b_train[:, 1]} if hasattr(b_train, 'ndim') and b_train.ndim > 1 else None
        visualize_latent_space(y_train, b_train_vis, b_labels, output_dir, args.vis_samples, 
                              data_split='train', generate_gif=args.generate_gif, generate_3d_html=args.generate_3d_html, colors=b_colors_rgb,
                              continuous_vars=continuous_train)
        print_step_time("Training latent space visualization", run_start_time, step_start)
        
        # Training recurrence plot
        if args.recurrence_threshold is not None:
            step_start = time.time()
            plot_recurrence(y_train, output_dir, args.recurrence_threshold, args.vis_samples, data_split='train')
            print_step_time("Training recurrence plot generation", run_start_time, step_start)
        
        # Free memory from training data
        del y_train, x_train, b_train
        gc.collect()
        
        # Project validation data into latent space
        step_start = time.time()
        print("Projecting validation data into latent space...")
        y_test = project_latent_space(x_test, model, lazy_loading=args.lazy_loading)
        print_step_time("Validation latent space projection", run_start_time, step_start)
        
        # Visualize validation latent space
        step_start = time.time()
        b_test_vis = b_test[:, 0].astype(int) if hasattr(b_test, 'ndim') and b_test.ndim > 1 else b_test
        continuous_test = {'HGF belief': b_test[:, 1]} if hasattr(b_test, 'ndim') and b_test.ndim > 1 else None
        visualize_latent_space(y_test, b_test_vis, b_labels, output_dir, args.vis_samples,
                              data_split='validation', generate_gif=args.generate_gif, generate_3d_html=args.generate_3d_html, colors=b_colors_rgb,
                              continuous_vars=continuous_test)
        print_step_time("Validation latent space visualization", run_start_time, step_start)
        
        # Validation recurrence plot
        if args.recurrence_threshold is not None:
            step_start = time.time()
            plot_recurrence(y_test, output_dir, args.recurrence_threshold, args.vis_samples, data_split='validation')
            print_step_time("Validation recurrence plot generation", run_start_time, step_start)
        
        # Collect validation metrics before freeing memory
        metrics = {
            'cv_mode': False,
            'best_markovian_loss': float(np.min(test_loss_array[:, 0])),
            'best_markovian_epoch': int(np.argmin(test_loss_array[:, 0])),
            'best_behaviour_loss': float(np.min(test_loss_array[:, 1])),
            'best_behaviour_epoch': int(np.argmin(test_loss_array[:, 1])),
            'final_markovian_loss': float(test_loss_array[-1, 0]),
            'final_behaviour_loss': float(test_loss_array[-1, 1]),
            'final_total_loss': float(test_loss_array[-1, 2])
        }
        
        # Free memory from validation data
        del y_test, x_test, b_test, model, loss_array, test_loss_array
        gc.collect()
        
        # Calculate execution time for comprehensive config
        total_time = time.time() - run_start_time
        
        # Save comprehensive configuration
        save_comprehensive_config(
            args=args,
            params=params,
            output_dir=output_dir,
            execution_time=format_elapsed_time(total_time),
            execution_time_seconds=total_time,
            metrics=metrics,
            cv_summary=None,
            start_timestamp=start_timestamp,
            error=None
        )
    
    # Free memory from color mappings
    del b_colors, b_colors_rgb
    gc.collect()
    
    total_time = time.time() - run_start_time
    print(f"\n{'='*80}")
    print(f"RUN {run_idx + 1}/{total_runs} COMPLETE!")
    print(f"Execution time: {format_elapsed_time(total_time)}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    return {
        'output_dir': output_dir,
        'execution_time': format_elapsed_time(total_time),
        'execution_time_seconds': total_time,
        'metrics': metrics
    }


def main():
    overall_start_time = time.time()
    args = parse_args()
    
    # Generate parameter combinations
    param_combinations, param_grid = generate_param_combinations(args)
    total_runs = len(param_combinations)
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH CONFIGURATION")
    print(f"{'='*80}")
    print(f"Total parameter combinations: {total_runs}")
    print(f"Parameters being searched:")
    print(f"  - data_path: {args.data_path}")
    print(f"  - downsample_fs: {args.downsample_fs}")
    print(f"  - downsample_method: {args.downsample_method}")
    print(f"  - good_neurons_only: {args.good_neurons_only}")
    print(f"  - apply_hold_transitions: {args.apply_hold_transitions}")
    print(f"  - normalize_method: {args.normalize_method}")
    print(f"  - window: {args.window}")
    print(f"  - latent_dim: {args.latent_dim}")
    print(f"  - batch_size: {args.batch_size}")
    print(f"  - learning_rate: {args.learning_rate}")
    print(f"  - gamma: {args.gamma}")
    print(f"{'='*80}\n")
    
    # Create grid search directory
    if total_runs > 1:
        grid_dir = create_grid_search_directory(args.output_dir)
        print(f"Grid search directory: {grid_dir}\n")
        # Initialize summary JSON
        summary_path = initialize_grid_search_summary(grid_dir, param_combinations, param_grid)
    else:
        # Single run - use original structure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        grid_dir = Path(args.output_dir) / f'run_{timestamp}'
        grid_dir.mkdir(parents=True, exist_ok=True)
        summary_path = None
    
    # Run experiments for each parameter combination
    results = []
    for run_idx, params in enumerate(param_combinations):
        if total_runs > 1:
            output_dir = create_run_directory(grid_dir, run_idx, params)
        else:
            output_dir = grid_dir
            # Create subdirectories for single run
            (output_dir / 'figures').mkdir(exist_ok=True)
            (output_dir / 'model').mkdir(exist_ok=True)
            (output_dir / 'data').mkdir(exist_ok=True)
        
        try:
            result = run_single_experiment(args, params, output_dir, run_idx, total_runs)
            results.append(result)
        except Exception as e:
            error_msg = str(e)
            print(f"\n{'!'*80}")
            print(f"ERROR in run {run_idx + 1}/{total_runs}")
            print(f"Parameters: {params}")
            print(f"Error: {error_msg}")
            print(f"{'!'*80}\n")
            
            result = {
                'output_dir': output_dir,
                'execution_time': 'FAILED',
                'error': error_msg
            }
            results.append(result)
        
        # Update summary JSON after each run
        if summary_path is not None:
            update_grid_search_summary(summary_path, run_idx, params, result)
    
    # Print final summary
    total_time = time.time() - overall_start_time
    successful_runs = sum(1 for r in results if 'error' not in r)
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE!")
    print(f"{'='*80}")
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {total_runs - successful_runs}")
    print(f"Total execution time: {format_elapsed_time(total_time)}")
    print(f"All results saved to: {grid_dir}")
    if summary_path is not None:
        print(f"Summary JSON: {summary_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
