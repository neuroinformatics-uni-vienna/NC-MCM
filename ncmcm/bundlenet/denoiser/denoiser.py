# File: denoiser.py
# This file defines and implements the standalone Denoiser class for the BunDLe-Net model.

# The denoiser is a component which extends BunDLe-Net to produce denoised representations of the latent states in the 
# original neuronal space. Through training, we minimize the distance between the abstracted denoised representations and the latent states,
# to ensure force the denoiser to be injective.

from ncmcm.bundlenet.bundlenet import BunDLeNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm
from .denoiser_data import DenoiserData
from .denoiserlosses import CompositeLoss, LossTerm


class Denoiser(nn.Module):
    r"""Denoiser network for the BunDLeNet model

    Let $\tau$ be the abstraction function learned by BunDLeNet. Then, a denoiser $\tau_D^{-1}$ is
    a function such that $\tau \circ \tau_D^{-1} = \mathrm{id}_Y$.

    **Note**: The current implementation of the Denoiser class assumes that the input data is provided in a windowed format, with a window size of 1.
    """

    def __init__(self, bundlenet_model: BunDLeNet, window_size: int):
        """Initializes the denoiser module with a (possibly pre-trained) BunDLe-Net algorithm

        Args:
            bundlenet_model (BunDLeNet): BunDLe-Net Model
            window_size (int, optional): Window size for the input data. Currently, only window_size of 1 is supported. Defaults to 1.
        """
        assert isinstance(bundlenet_model, BunDLeNet), "bundlenet_model must be an instance of BunDLeNet"
        assert window_size == 1, "Currently, only window_size of 1 is supported. Please set window_size to 1."


        super(Denoiser, self).__init__()

        # BunDLe-Net model.
        self.bundlenet_model: BunDLeNet = bundlenet_model
        
        # Latent space dimension of the BunDLe-Net model
        self.bundlenet_latent_dim: int = bundlenet_model.latent_dim
        
        # BunDLe-Net abstraction function
        self.bundlenet_tau: nn.Sequential = bundlenet_model.tau
        self.bundlenet_tau.eval()
        
        # BunDLe-Net abstract dynamics block
        self.bundlenet_t_y: nn.Sequential = bundlenet_model.T_Y
        self.bundlenet_t_y.eval()

        # Input features of the BunDLe-Net model. Since BunDLe-net does not store
        # the number of input features, we extract them from the first layer of the abstraction 
        # function, tau.
        bundle_net_first_layer: nn.Linear = self.bundlenet_tau[1]
        self.bundlenet_input_features: int = bundle_net_first_layer.in_features

        # Trainable denoising function. We keep the structure of the denoiser equal to that of the 
        # BunDLe-Net encoder.
        self.tau_d_inv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.bundlenet_latent_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 30),
            nn.ReLU(),
            nn.Linear(30, 50),
            nn.ReLU(),
            nn.Linear(50, self.bundlenet_input_features),
            nn.ReLU()
        )

    def forward(self, latent_state):
        """De-noises a latent state, and re-abstracts it. Returns the de-noised state and the
        re-abstracted latent state.

        Args:
            latent_state: Current latent state.
        """
        denoised_state = self.tau_d_inv(latent_state)
        abstracted_state = self.bundlenet_tau(denoised_state)
        return denoised_state, abstracted_state
    
    def pipeline(self, neuronal_states):
        """Full pipeline of the denoiser, which takes in neuronal states, projects them into the latent space, de-noises them, and re-abstracts them.

        Args:
            neuronal_states: Current neuronal states.
        """
        device = self.tau_d_inv[1].weight.device


        latent_states = None
        denoised_states = None
        re_abstracted_states = None

        with torch.no_grad():
                latent_states = self.bundlenet_tau(torch.from_numpy(neuronal_states[:,0]).float().to(device))
                denoised_states, re_abstracted_states = self.forward(latent_states)
                print(f"Latent state shape: {latent_states.shape}")
                print(f"Denoised state shape: {denoised_states.shape}")
                print(f"Re-abstracted state shape: {re_abstracted_states.shape}")

        return denoised_states.cpu().numpy(), re_abstracted_states.cpu().numpy()
class DenoiserTrainer:
    """Trainer Manager for the Denoiser extension model of BunDLe-Net. If the denoiser object
    is supplied an untrained BunDLe-Net model, the model is trained, frozen, and only then is the 
    de-noiser module trained.
    """

    def __init__(self,
                denoiser: Denoiser,
                denoiser_optimizer: torch.optim.Optimizer = None,
                denoiser_loss_fn: CompositeLoss | LossTerm = None,
                denoiser_train_loader: DataLoader = None,
                denoiser_test_loader: DataLoader = None,
                denoiser_num_epochs: int = 1000,
                device: torch.device = torch.device('cpu'),
                ):
        
        assert denoiser is not None, "Denoiser object must be provided"
        assert denoiser_optimizer is not None, "Denoiser optimizer must be provided"
        assert denoiser_loss_fn is not None, "Denoiser loss function must be provided"
        assert denoiser_train_loader is not None, "Denoiser train loader must be provided"
        assert denoiser_test_loader is not None, "Denoiser test loader must be provided"


        # Denoiser module
        self.denoiser : Denoiser = denoiser
        
        # Optimizer for the denoiser module training
        self.optimizer : torch.optim.Optimizer = denoiser_optimizer
        
        # BunDLe-Net model, as passed to the denoiser module
        self.bundlenet_model: BunDLeNet = denoiser.bundlenet_model
        
        # Loss function for the denoiser module training
        self.loss_fn : CompositeLoss | LossTerm = denoiser_loss_fn.to(device)

        # Train Loader
        self.train_loader : DataLoader = denoiser_train_loader

        # Test Loader
        self.test_loader : DataLoader = denoiser_test_loader

        # Device 
        self.device = device

        # Training epochs
        self.num_epochs = denoiser_num_epochs

        print(f"DenoiserTrainer initialized with {self.num_epochs} epochs, device: {self.device}, "
              f"denoiser optimizer: {self.optimizer}, denoiser loss function: {self.loss_fn}, "
              f"train loader: {self.train_loader}, test loader: {self.test_loader}")

    def _freeze_bundlenet(self):
        """Freezes the weights of the BunDLe-Net model."""
        for param in tqdm(self.bundlenet_model.parameters(), desc="Freezing BunDLe-Net weights"):
            param.requires_grad = False


    def _train_step(self, sample):
        """Performs a single training step for the denoiser module.

        Args:
            X_train: Input data for the training step.
            y_train: Target data for the training step.
        """
        self.optimizer.zero_grad()

        denoised_state, re_abstracted_state = self.denoiser(sample[1])

        # Compute loss between the re-abstracted state and the original latent state
        loss = self.loss_fn(DenoiserData(
            original_neuronal=sample[0],
            original_latent=sample[1],
            denoised_neuronal=denoised_state,
            reconstructed_latent=re_abstracted_state,
            behavioral_label=sample[2]
        ))

        # Backward pass and optimization step
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _test_step(self, sample):
        """Performs a single test step for the denoiser module.

        Args:
            X_test: Input data for the test step.
            y_test: Target data for the test step.
        """

        # Forward pass through the denoiser
        denoised_state, re_abstracted_state = self.denoiser(sample[1])

        # Compute loss between the re-abstracted state and the original latent state
        loss = self.loss_fn(DenoiserData(
            original_neuronal=sample[0],
            original_latent=sample[1],
            denoised_neuronal=denoised_state,
            reconstructed_latent=re_abstracted_state,
            behavioral_label=sample[2]
            ))
        
        return loss

    def _train_epoch(self):
        """Handles the training within a single epoch and logs losses."""
        self.denoiser.train()
        self.denoiser.bundlenet_model.eval()
        
        tot_loss = 0.0
        for i, sample in enumerate(self.train_loader):
            loss = self._train_step(sample)
            tot_loss += loss.item()
        
        avg_train_loss = tot_loss / len(self.train_loader)
        return avg_train_loss

    def _test_epoch(self):
        """Handles the testing within a single epoch and logs losses."""
        self.denoiser.eval()

        tot_loss = 0.0
        with torch.no_grad():
            for i, sample in enumerate(self.test_loader):
                loss = self._test_step(sample)
                tot_loss += loss.item()

        avg_test_loss = tot_loss / len(self.test_loader)

        return avg_test_loss


    def train(self):
        """
        Trains the denoiser module, assuming that the BunDLe-Net model is already trained.
        As a pre-processing, BunDLe-Net weights are frozen.
        """

        self._freeze_bundlenet()
        progress: tqdm = tqdm(range(1, self.num_epochs + 1), desc="Training", unit="epoch", smoothing=0.1)
        for epoch in progress:
            self.loss_fn.record_training()
            avg_train_loss = self._train_epoch()
            self.loss_fn.handle_epoch_end()
            self.loss_fn.record_testing()
            avg_test_loss = self._test_epoch()
            self.loss_fn.handle_epoch_end()
            
            progress.set_postfix(train_loss=f"{avg_train_loss:.4f}", test_loss=f"{avg_test_loss:.4f}")

    def summarize(self):
        """Summarizes the training configuration for the denoiser module."""
        summary = f"DenoiserModel_epochs_{self.num_epochs}_loss_{self.loss_fn.summarize()}"
        return summary

def prepare_denoiser_data(X, B, bundlenet_model: BunDLeNet = None, device: torch.device = None, train_split: float = 0.8, batch_size: int = 8):
    assert bundlenet_model is not None, "BunDLe-Net model must be provided"
    assert device is not None, "Device must be provided"

    # Project the neuronal data into the latent space using the BunDLe-Net model
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float, device=device)
        B_tensor = torch.tensor(B, dtype=torch.float, device=device)
        latent_states = bundlenet_model.tau(X_tensor)

    denoiser_data = TensorDataset(
        X_tensor,
        latent_states,
        B_tensor        
    )
    
    train_size = int(len(denoiser_data) * train_split)
    test_size = len(denoiser_data) - train_size

    train_dataset, test_dataset = random_split(denoiser_data, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader