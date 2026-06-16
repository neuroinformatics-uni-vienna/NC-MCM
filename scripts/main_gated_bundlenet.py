import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sympy import gamma
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn

import os


BASIC_PATH = f"data/generated/TEST/gated_bundlenet_p20"


from ncmcm.bundlenet.denoiser.denoiser import Denoiser, prepare_denoiser_data, DenoiserTrainer
from ncmcm.bundlenet.denoiser.denoiserlosses import *
from ncmcm.bundlenet.denoiser.denoiser_analysis import *
from ncmcm.bundlenet.denoiser.denoiser_data import DenoiserData
from ncmcm.bundlenet.neuronal_saliency.neuronal_saliency import NeuronalSaliencyAnalyzer, NeuronalSaliencyPlotter
from ncmcm.bundlenet.denoiser.gated_bundlendet import GatedBunDLeNet, train_model as gated_train_model


def save_bundlenet(model: BunDLeNet, Y0_, B_, worm_num):
    """Saves the trained BunDLeNet model and the corresponding latent representations and behavioural labels."""
    algorithm = 'BunDLeNet'
    os.makedirs(BASIC_PATH, exist_ok=True)
    torch.save(model.state_dict(), f'{BASIC_PATH}/BunDLeNet_model_worm_{worm_num}.pt')
    os.makedirs(f'{BASIC_PATH}/saved_Y', exist_ok=True)
    np.savetxt(f'{BASIC_PATH}/saved_Y/Y0__{algorithm}_worm_{worm_num}', Y0_)
    np.savetxt(f'{BASIC_PATH}/saved_Y/B__{algorithm}_worm_{worm_num}', B_)
    Y0_ = np.loadtxt(f'{BASIC_PATH}/saved_Y/Y0__{algorithm}_worm_{worm_num}')
    B_ = np.loadtxt(f'{BASIC_PATH}/saved_Y/B__{algorithm}_worm_{worm_num}').astype(int)

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data (excluding behavioural neurons) and plot
    worm_num = 0
    algorithm = 'BunDLeNet'
    b_neurons = [
        'AVAR',
        'AVAL',
        'SMDVR',
        'SMDVL',
        'SMDDR',
        'SMDDL',
        'RIBR',
        'RIBL'
    ]
    data_path = 'datasets/raw/c_elegans/NoStim_Data.mat'
    data = Database(data_path=data_path, dataset_no=worm_num)
    data.exclude_neurons(b_neurons)
    X = data.neuron_traces.T
    B = data.behaviour
    neuron_names = list(getattr(data, 'neuron_names', [f'Neuron {i}' for i in range(X.shape[1])]))

    # Ensure behaviour names are present
    if not hasattr(data, 'behaviour_names') or data.behaviour_names is None or len(getattr(data, 'behaviour_names', [])) == 0:
        # Default behaviour names for the dataset
        data.behaviour_names = ['Forward', 'Backward', 'Turn']

    # Prepare data for BunDLe Net
    label_encoder = LabelEncoder()
    B = label_encoder.fit_transform(B)
    X_, B_ = prep_data(X, B, win=1)

    model: GatedBunDLeNet = GatedBunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names), input_shape=X_.shape)
    os.makedirs(BASIC_PATH, exist_ok=True)

    loss_array, _, pruned_mask, num_pruned = gated_train_model(x_train=X_, b_train_1=B_, model=model, gamma=0.9, device=device, b_type='discrete', learning_rate=0.01, n_epochs=1000)

    for i, label in enumerate([
        r"$\mathcal{L}_{\mathrm{Markov}}$",
        r"$\mathcal{L}_{\mathrm{Behavior}}$",
        r"Total loss $\mathcal{L}$"
    ]):
        plt.plot(loss_array[:, i], label=label)

    plt.legend()
    plt.show()

    model.eval()
    Y0_ = project_into_latent_space(X_, model)
    save_bundlenet(model, Y0_, B_, worm_num)

    print("Pruned features mask:", pruned_mask)
    print("Number of remaining features:", num_pruned)

    model.plot_gating_parameters(neuronal_names = neuron_names, save_path=f"{BASIC_PATH}/gating_parameters_worm_{worm_num}.png")

    