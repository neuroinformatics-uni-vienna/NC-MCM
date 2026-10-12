# File: denoiser.py
# This file defines and implements the standalone Denoiser class for the BunDLe-Net model.

# The denoiser is a component which extends BunDLe-Net to produce denoised representations of the latent states in the 
# original neuronal space. Through training, we minimize the distance between the abstracted denoised representations and the latent states,
# to ensure force the denoiser to be injective.

from ncmcm.bundlenet.bundlenet import BunDLeNet
import torch

import torch.nn as nn
import numpy as np
from ncmcm.bundlenet.utils import GaussianNoise, prep_data
from .correction.convex_combination import ConvexCombinationCorrection

from ncmcm.bundlenet.denoiser.printer import print_status, print_warning, print_success, print_hidden_layers

class ExponentialActivation(nn.Module):
    def __init__(self):
        super(ExponentialActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)
    

class Denoiser(nn.Module):
    r"""Denoiser network for the BunDLeNet model

    Let $\tau$ be the abstraction function learned by BunDLeNet. Then, a denoiser $\tau_D^{-1}$ is
    a function such that $\tau \circ \tau_D^{-1} = \mathrm{id}_Y$.

    The main objective of de-noising is to construct a synthetic de-noised representation of the latent state in the original neuronal space, which can be used for downstream analyses. Ideally,
    it ensures that behaviorally relevant information is preserved.

    """

    def __init__(self,
        bundlenet_model: BunDLeNet,
        window_size: int, 
        den_hidden_layers=None, 
        bundlenet_hidden_layers=None,
        add_correction=False, 
        correction_bias=0.0
    ):
        
        """Initializes the denoiser module with a (possibly pre-trained) BunDLe-Net algorithm

        Args:
            bundlenet_model (BunDLeNet): BunDLe-Net Model
            window_size (int, optional): Window size for the input data. Currently, only window_size of 1 is supported. Defaults to 1.
            enforce_final_gating_layer (bool, optional): Whether to enforce the final gating layer of the denoiser to match that of the GatedBunDLeNet model. Defaults to True.
            input_shape (tuple, optional): Shape of the input data. Defaults to None.
            hidden_layers (list, optional): List of number of neurons for each hidden layer of the denoiser. Defaults to None. If None, a default architecture is used. 
        """
        assert isinstance(bundlenet_model, BunDLeNet), "bundlenet_model must be an instance of BunDLeNet"

        super(Denoiser, self).__init__()

        # BunDLe-Net model.
        self.bundlenet_model: BunDLeNet = bundlenet_model

        # Window size for the input data
        self.window = window_size

        # Latent dimension of the BunDLe-Net model.
        self.bundlenet_latent_dim: int = bundlenet_model.latent_dim

        # BunDLe-Net model abstraction function (tau). 
        self.bundlenet_tau: nn.Sequential = bundlenet_model.tau
        self.bundlenet_tau.eval()

        # BunDLe-Net abstract dynamics block
        self.bundlenet_t_y: nn.Sequential = bundlenet_model.T_Y
        self.bundlenet_t_y.eval()

        # Hidden layer size for the bundlenet abstraction function. If None, a default architecture is used.
        self.bundlenet_hidden_layers = bundlenet_hidden_layers

        # Input features of the BunDLe-Net model. Since BunDLe-net does not store
        # the number of input features, we extract them from the first layer of the abstraction 
        # function, tau.
        bundle_net_first_layer: nn.Linear = self.bundlenet_tau[1]
        self.bundlenet_input_features: int = bundle_net_first_layer.in_features
        self.number_of_neurons = int(self.bundlenet_input_features / self.window)

        # ============= Prepare the Denoiser and BunDLe-Net if hidden layer sizes are provided ============= 
        
        self.den_hidden_layers = den_hidden_layers
        self.bundlenet_hidden_layers = bundlenet_hidden_layers

        # Number of hidden layers for the inverse of tau. If None, a default architecture is used.
        self.tau_d_inv_hidden_layers = []

        if den_hidden_layers is None:
            print_status(message="No hidden layers provided for the denoiser. Using default architecture.", details={})

            self.tau_d_inv_hidden_layers = [
                nn.Linear(self.bundlenet_latent_dim, 10),
                nn.BatchNorm1d(10),
                nn.GELU(),
                nn.Linear(10, 30),
                nn.BatchNorm1d(30),
                nn.GELU(),
                nn.Linear(30, 50),
                nn.BatchNorm1d(50),
                nn.GELU(),
                nn.Linear(50, self.bundlenet_input_features)
            ]

            print_status(message="Using default hidden layers for the denoiser.", details={})
        else:
            print_status(message="Using provided hidden layers for the denoiser.", details={"den_hidden_layers": den_hidden_layers})
            self.tau_d_inv_hidden_layers = []
            # Count the number of hidden layers provided for the denoiser
            n_hidden_layers = len(den_hidden_layers)
            
            # Construct the hidden layers for the denoiser based on the provided architecture. 
            # Each hidden layer is followed by a BatchNorm and GELU activation.
            for i in range(n_hidden_layers-1):
                self.tau_d_inv_hidden_layers.append(nn.Linear(den_hidden_layers[i], den_hidden_layers[i+1]))
                self.tau_d_inv_hidden_layers.append(nn.BatchNorm1d(den_hidden_layers[i+1])) 
                self.tau_d_inv_hidden_layers.append(nn.GELU())

            # And finally, delete the last GELU and the last BatchNorm layer
            self.tau_d_inv_hidden_layers.pop()
            self.tau_d_inv_hidden_layers.pop()

            print_hidden_layers(den_hidden_layers, title="Hidden Layers - Denoiser")

            
            if bundlenet_hidden_layers is not None:
                print_status(message="Modifying the BunDLe-Net model to match the denoiser architecture.", details={})

                bundlenet_n_hidden_layers = bundlenet_hidden_layers

                bundlenet_hidden_layers = []
                for i in range(len(bundlenet_n_hidden_layers)-1):
                    bundlenet_hidden_layers.append(nn.Linear(bundlenet_n_hidden_layers[i], bundlenet_n_hidden_layers[i+1]))
                    bundlenet_hidden_layers.append(nn.GELU())
                bundlenet_hidden_layers.pop()  # Remove the last GELU

                self.bundlenet_model.tau = nn.Sequential(
                    nn.Flatten(),
                    *bundlenet_hidden_layers,
                    GaussianNoise(mean=0, stddev=0.05),
                )

                print_hidden_layers(bundlenet_n_hidden_layers, title="Hidden Layers - Modified BunDLe-Net model τ")


        assert self.tau_d_inv_hidden_layers is not None, "Hidden layers for the denoiser must be provided or default architecture must be used."

        # Trainable denoising function. We keep the structure of the denoiser equal to that of the 
        # BunDLe-Net encoder.
        self.tau_d_inv = nn.Sequential(
            nn.Flatten(),
            *self.tau_d_inv_hidden_layers,
            nn.BatchNorm1d(self.bundlenet_input_features),
            nn.Softplus()
        )

        self.add_correction = add_correction
        self.correction_bias = correction_bias
        if self.add_correction:
            print_status(message="Adding correction term to the denoiser with bias:", details={"bias": self.correction_bias})
        self.activation_correction = ConvexCombinationCorrection(input_dim=self.bundlenet_input_features, bias=correction_bias, bundlenet_tau=self.bundlenet_tau)

        print_success(message="Denoiser is initialized successfully.", details={
            "BunDLe-Net latent dimension": self.bundlenet_latent_dim,
            "BunDLe-Net input features": self.bundlenet_input_features,
            "Denoiser hidden layers": self.den_hidden_layers,
            "BunDLe-Net hidden layers": self.bundlenet_hidden_layers,
            "Add correction term": self.add_correction,
        })

    def get_init_config(self):
        return {
            "window_size": self.window,
            "den_hidden_layers": self.den_hidden_layers,
            "bundlenet_hidden_layers": self.bundlenet_hidden_layers,
            "add_correction": self.add_correction,
            "correction_bias": self.correction_bias,

            "bundlenet_model": self.bundlenet_model,
            "number_of_neurons": self.number_of_neurons,
        }

    def forward(self, latent_state, original_neuronal=None, previous_neuronal=None, is_first=None, correct_current=True):
        """De-noises a latent state, and re-abstracts it. Returns the de-noised state and the
        re-abstracted latent state.

        Args:
            original_neuronal: Original neuronal state.
            latent_state: Current latent state.
            previous_neuronal: Previous neuronal state.
            is_first: Boolean indicating if this is the first time step.
        """
        # Compute the de-noised neuronal state
        denoised = None

        if correct_current is True:
            with torch.no_grad():
                denoised = self.tau_d_inv(latent_state)
        else:
            denoised = self.tau_d_inv(latent_state)

        denoised = denoised.reshape(original_neuronal.shape)

        denoised_copy = None
        correction = torch.zeros_like(denoised, device=denoised.device)
        if correct_current is True:
            neuronal, correction = self.activation_correction.apply_correction(denoised, original_neuronal, previous_neuronal, is_first)
            denoised_copy = denoised.clone().detach()
        else:
            neuronal = denoised

        abstracted_state = None
        if correct_current is True:
            with torch.no_grad():
                abstracted_state = self.bundlenet_tau(neuronal)
        else:
            abstracted_state = self.bundlenet_tau(neuronal)

        return neuronal, abstracted_state, correction, denoised_copy
    
    def pipeline(self, neuronal_states, activate_correction=True, corrections=None, device=None):
        """Full pipeline of the denoiser, which takes in neuronal states, projects them into the latent space, de-noises them, and re-abstracts them.

        Args:
            neuronal_states: Current neuronal states.

        Returns:
            denoised_states: De-noised neuronal states.
            
            re_abstracted_states: Re-abstracted latent states obtained by applying the BunDLe-Net abstraction function to the de-noised neuronal states.

            -> Returns are on CPU and in numpy format.
        """

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        correct_current = activate_correction and self.add_correction

        latent_states = None
        denoised_states = None
        re_abstracted_states = None

        _neuronal_states = neuronal_states.copy()
        if isinstance(neuronal_states, np.ndarray):
            _neuronal_states = torch.from_numpy(neuronal_states).float().to(device)

        original_neuronal = _neuronal_states[:, 0, :].to(device)
        original_previous_neuronal = torch.zeros_like(original_neuronal, device=device)
        original_previous_neuronal[1:] = original_neuronal[:-1]

        is_first = torch.zeros_like(original_neuronal, device=device)
        is_first[0] = 1

        with torch.no_grad():
                latent_states = self.bundlenet_tau(torch.from_numpy(neuronal_states[:,0]).float().to(device))

                self.activation_correction.set_inference()
                denoised_states, re_abstracted_states, correction, _ = self.forward(
                    latent_state=latent_states,
                    original_neuronal=original_neuronal,
                    previous_neuronal=original_previous_neuronal,
                    is_first=is_first,
                    correct_current=correct_current
                )

                if corrections is not None:
                    corrections.append(correction.cpu().numpy())

        return denoised_states.cpu().numpy(), re_abstracted_states.cpu().numpy()
    
    def pipeline_unprepared_data(self, neuronal_states, activate_correction=True, corrections=None):
        """Full pipeline of the denoiser, which takes in neuronal states, projects them into the latent space, de-noises them, and re-abstracts them.

        Args:
            neuronal_states: Current neuronal states.

        Returns:
            denoised_states: De-noised neuronal states.
            
            re_abstracted_states: Re-abstracted latent states obtained by applying the BunDLe-Net abstraction function to the de-noised neuronal states.

            -> Returns are on CPU and in numpy format.
        """
        device = self.tau_d_inv[1].weight.device

        behaviors = np.zeros((neuronal_states.shape[0], 1))  # Placeholder for behavioral labels
        neuronal_prepared, _ = prep_data(neuronal_states, behaviors, win=self.window)
        neuronal_prepared_t = neuronal_prepared[:, 0, :, :]

        return self.pipeline(neuronal_prepared_t, activate_correction=activate_correction, corrections=corrections, device=device)

    def get_bundlendet_handle(self):
        """Returns the handle to the BunDLe-Net model used in the denoiser."""
        return self.bundlenet_model

    def save_model(self, path: str):
        """Saves the denoiser model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)
        print(f"[bold green]Denoiser model saved to {path}[/bold green]")

    def load_model(self, path: str):
        """Loads the denoiser model from the specified path.

        Args:
            path (str): Path to load the model from.
        """
        self.load_state_dict(torch.load(path, map_location=self.tau_d_inv[1].weight.device))
        print(f"[bold green]Denoiser model loaded from {path}[/bold green]")
