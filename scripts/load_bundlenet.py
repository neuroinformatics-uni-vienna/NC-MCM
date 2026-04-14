import argparse
import json
import numpy as np
import torch
from pathlib import Path

# Import BunDLeNet and utilities
from ncmcm.bundlenet.bundlenet import BunDLeNet


def load_model_config(run_dir):
    """Load model configuration and parameters from run_summary.json and behavior labels."""
    run_dir = Path(run_dir)
    summary_path = run_dir / 'run_summary.json'
    config_path = run_dir / 'config.json'
    config = None
    params = None
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            config = json.load(f)
        params = config['parameters']
    elif config_path.exists():
        with open(config_path, 'r') as f:
            params = json.load(f)
        config = {'parameters': params}
    else:
        raise FileNotFoundError(f"Neither run_summary.json nor config.json found in {run_dir}")
    # Try train labels first, fallback to validation
    try:
        b_train = np.load(run_dir / 'data' / 'behavior_labels_train.npy')
    except FileNotFoundError:
        b_train = np.load(run_dir / 'data' / 'behavior_labels_validation.npy')
    num_behaviour = len(np.unique(b_train))
    # Infer n_neurons from latent trajectories shape or config
    try:
        latent = np.load(run_dir / 'data' / 'latent_trajectories_train.npy')
    except FileNotFoundError:
        latent = np.load(run_dir / 'data' / 'latent_trajectories_validation.npy')
    print(f"DEBUG: latent_trajectories shape = {latent.shape}")
    # Use the last dimension of latent trajectories for n_neurons
    # The original input shape for BunDLeNet is (n_samples, 2, window-1, n_neurons)
    latent_dim = params['latent_dim']
    window = params['window']
    # Try to infer n_neurons from the original training data shape
    # If available, use the shape of the original x used for training
    # Otherwise, fallback to previous method
    n_neurons = None
    # Try to find the original x shape from config or latent
    # If latent shape is (n_samples, latent_dim), we can't get n_neurons
    # If latent shape is (n_samples, 2, window-1, n_neurons), we can get n_neurons
    if len(latent.shape) == 4:
        n_neurons = latent.shape[-1]
        input_shape = (None, 2, window-1, n_neurons)
        print(f"DEBUG: n_neurons inferred from latent shape = {n_neurons}")
    else:
        print("WARNING: latent trajectories do not contain n_neurons info. Falling back to tau_weight method.")
        model_weights = torch.load(run_dir / 'model' / 'bundlenet_model.pt', map_location='cpu')
        tau_key = 'tau.1.weight'
        if tau_key not in model_weights:
            print(f"ERROR: '{tau_key}' not found in model weights.")
            print("Available keys in state_dict:")
            for k in model_weights.keys():
                print(f"  {k}")
            raise KeyError(f"'{tau_key}' not found in model weights.")
        tau_weight = model_weights[tau_key]
        print(f"DEBUG: tau_weight.shape = {tau_weight.shape}")
        print(f"DEBUG: window = {window}")
        n_neurons = tau_weight.shape[1] // (2 * (window-1))
        print(f"DEBUG: calculated n_neurons = {n_neurons}")
        input_shape = (None, 2, window-1, n_neurons)
        print(f"DEBUG: input_shape to BunDLeNet = {input_shape}")
        if tau_weight.shape[1] % (2 * (window-1)) != 0:
            print(f"WARNING: tau_weight.shape[1] is not divisible by 2*(window-1). Check window and n_neurons calculation.")
    return {
        'latent_dim': latent_dim,
        'num_behaviour': num_behaviour,
        'input_shape': input_shape,
        'run_dir': run_dir,
        'window': window,
        'n_neurons': n_neurons,
        'params': params
    }


def load_bundlenet_model(run_dir):
    """Load BunDLeNet model and weights from a run directory."""
    config = load_model_config(run_dir)
    model = BunDLeNet(
        latent_dim=config['latent_dim'],
        num_behaviour=config['num_behaviour'],
        input_shape=config['input_shape']
    )
    weights_path = Path(run_dir) / 'model' / 'bundlenet_model.pt'
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Load BunDLeNet model from saved weights.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to run directory containing model weights and config')
    args = parser.parse_args()

    print(f"Loading BunDLeNet model from: {args.run_dir}")
    model, config = load_bundlenet_model(args.run_dir)
    print("Model loaded successfully.")
    print(f"Latent dim: {config['latent_dim']}")
    print(f"Num behaviour: {config['num_behaviour']}")
    print(f"Input shape: {config['input_shape']}")
    print(f"Window: {config['window']}")
    print(f"Num neurons: {config['n_neurons']}")
    print("Ready for inference or further analysis.")

if __name__ == '__main__':
    main()

