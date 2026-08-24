# File: denoiser.py
# This file defines and implements the standalone Denoiser class for the BunDLe-Net model.

# The denoiser is a component which extends BunDLe-Net to produce denoised representations of the latent states in the 
# original neuronal space. Through training, we minimize the distance between the abstracted denoised representations and the latent states,
# to ensure force the denoiser to be injective.

from sklearn.model_selection import train_test_split

from ncmcm.bundlenet.bundlenet import BunDLeNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
import numpy as np
from tqdm import tqdm

from ncmcm.bundlenet.denoiser.gated_bundlendet import GatingLayer, GatedBunDLeNet
from ncmcm.bundlenet.denoiser.temporal_split import _no_leakage_split
from ncmcm.bundlenet.utils import GaussianNoise
from .denoiser_data import DenoiserData, DenoiserDistributionMatchingData
from .denoiserlosses import CompositeLoss, LossTerm, StatisticalCompositeLoss, StatisticalLossTerm


class Denoiser(nn.Module):
    r"""Denoiser network for the BunDLeNet model

    Let $\tau$ be the abstraction function learned by BunDLeNet. Then, a denoiser $\tau_D^{-1}$ is
    a function such that $\tau \circ \tau_D^{-1} = \mathrm{id}_Y$.

    The main objective of de-noising is to construct a synthetic de-noised representation of the latent state in the original neuronal space, which can be used for downstream analyses. Ideally,
    it ensures that behaviorally relevant information is preserved.

    **Note**: The current implementation of the Denoiser class assumes that the input data is provided in a windowed format, with a window size of 1.
    """

    def __init__(self, bundlenet_model: BunDLeNet, window_size: int, enforce_final_gating_layer: bool = True):
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


        self.enforce_final_gating_layer = enforce_final_gating_layer
        self.final_layer = GatingLayer(self.bundlenet_input_features) if isinstance(self.bundlenet_model, GatedBunDLeNet) and enforce_final_gating_layer else nn.Identity()
        
        if not isinstance(self.bundlenet_model, GatedBunDLeNet) and enforce_final_gating_layer:
            print("Warning: enforce_final_gating_layer is set to True, but the provided BunDLe-Net model is not a GatedBunDLeNet. The final layer will be an identity layer.")

        # Trainable denoising function. We keep the structure of the denoiser equal to that of the 
        # BunDLe-Net encoder.
        self.tau_d_inv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.bundlenet_latent_dim, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(30, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, self.bundlenet_input_features),
            GaussianNoise(mean=0, stddev=0.05),
            nn.Softplus(),
            self.final_layer
        )

        if isinstance(self.bundlenet_model, GatedBunDLeNet) and enforce_final_gating_layer:
            print("Initializing the final layer of the denoiser with the mask and gate_scale from the GatedBunDLeNet model.")
            print("Final layer before initialization:", self.final_layer)

            # Set the mask, and the gate_scale can be set to 1s
            self.final_layer.init_layer(mask=self.bundlenet_model.tau[1].mask.cpu().numpy(), gate_scale=torch.ones(self.bundlenet_input_features).cpu().numpy())
            self.final_layer.freeze_all_parameters()  # Freeze the gating layer parameters to prevent them from being updated during training
        
    def attach_gating_mask(self, mask: np.ndarray):
        """Attaches a gating mask to the final layer of the denoiser. This is useful for ensuring that the denoiser respects the same gating as the GatedBunDLeNet model.

        Args:
            mask (np.ndarray): Binary mask to be applied to the final layer of the denoiser.
        """
        assert isinstance(mask, np.ndarray), "Mask must be a numpy array"
        assert mask.ndim == 1, "Mask must be a 1D array"
        assert mask.shape[0] == self.bundlenet_input_features, f"Mask length must match input features of BunDLeNet ({self.bundlenet_input_features})"


        if isinstance(self.final_layer, GatingLayer):
            if not self.enforce_final_gating_layer:
                raise RuntimeWarning("enforce_final_gating_layer is set to False. The gating mask will not be attached to the final layer of the denoiser."
                " If you want to enforce the gating mask, please set enforce_final_gating_layer to True when initializing the Denoiser.")

            self.final_layer.init_layer(mask=mask, gate_scale=torch.ones(self.bundlenet_input_features).cpu().numpy())
            self.final_layer.freeze_all_parameters()  # Freeze the gating layer parameters to prevent them from being updated during training
            print("Gating mask attached to the final layer of the denoiser.")
        else:
            raise ValueError("Final layer is not a GatingLayer. Cannot attach gating mask.")

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

        Returns:
            denoised_states: De-noised neuronal states.
            
            re_abstracted_states: Re-abstracted latent states obtained by applying the BunDLe-Net abstraction function to the de-noised neuronal states.

            -> Returns are on CPU and in numpy format.
        """
        device = self.tau_d_inv[1].weight.device


        latent_states = None
        denoised_states = None
        re_abstracted_states = None

        with torch.no_grad():
                latent_states = self.bundlenet_tau(torch.from_numpy(neuronal_states[:,0]).float().to(device))
                denoised_states, re_abstracted_states = self.forward(latent_states)
        return denoised_states.cpu().numpy(), re_abstracted_states.cpu().numpy()
    
    def pipeline_unprepared_data(self, neuronal_states):
        """Full pipeline of the denoiser, which takes in neuronal states, projects them into the latent space, de-noises them, and re-abstracts them.

        Args:
            neuronal_states: Current neuronal states.

        Returns:
            denoised_states: De-noised neuronal states.
            
            re_abstracted_states: Re-abstracted latent states obtained by applying the BunDLe-Net abstraction function to the de-noised neuronal states.

            -> Returns are on CPU and in numpy format.
        """
        device = self.tau_d_inv[1].weight.device


        latent_states = None
        denoised_states = None
        re_abstracted_states = None

        with torch.no_grad():
                latent_states = self.bundlenet_tau(torch.from_numpy(neuronal_states).float().to(device))
                denoised_states, re_abstracted_states = self.forward(latent_states)
        return denoised_states.cpu().numpy(), re_abstracted_states.cpu().numpy()
class DenoiserTrainer:
    """Trainer Manager for the Denoiser extension model of BunDLe-Net. The BunDLe-Net model is frozen, and the Denoiser is trained.
    """

    def __init__(self,
                denoiser: Denoiser,
                optimizer: torch.optim.Optimizer = None,
                loss_fn: CompositeLoss | LossTerm = None,
                train_loader: DataLoader = None,
                test_loader: DataLoader = None,
                num_epochs: int = 1000,
                statistical_fit: bool = False,
                statistical_loss_fn: StatisticalLossTerm | StatisticalCompositeLoss = None,
                statistical_epochs: float = 0.1,
                device: torch.device = torch.device('cpu'),
                ):
        """Initializes the trainer for the Denoiser module.

        Args:
            denoiser (Denoiser): Denoiser module to be trained.
            optimizer (torch.optim.Optimizer, optional): Optimizer for training the denoiser module. Defaults to None.
            loss_fn (CompositeLoss | LossTerm, optional): Loss function for training the denoiser module. Defaults to None.
            train_loader (DataLoader, optional): Data loader for training data. Defaults to None.
            test_loader (DataLoader, optional): Data loader for test data. Defaults to None.
            num_epochs (int, optional): Number of training epochs. Defaults to 1000.
            statistical_fit (bool, optional): Flag for using statistical fitting. Defaults to False.
            statistical_loss_fn (StatisticalLossTerm, optional): Statistical loss function. Defaults to None.
            statistical_epochs (float, optional): Fraction of training epochs for which the statistical fit loss will be applied. Defaults to 0.1.
            device (torch.device, optional): Device for training. Defaults to torch.device('cpu').
        """
        assert denoiser is not None, "Denoiser object must be provided"
        assert optimizer is not None, "Denoiser optimizer must be provided"
        assert loss_fn is not None, "Denoiser loss function must be provided"
        assert train_loader is not None, "Denoiser train loader must be provided"
        assert test_loader is not None, "Denoiser test loader must be provided"
        assert device is not None, "Device must be provided"
        assert statistical_fit is not None, "Statistical fit flag must not be None"
        assert statistical_epochs >= 0 and statistical_epochs <= 1, "statistical_epochs must be a float between 0 and 1 representing the fraction of training epochs for which the statistical fit loss will be applied."

        # Denoiser module
        self.denoiser : Denoiser = denoiser
        
        # Optimizer for the denoiser module training
        self.optimizer : torch.optim.Optimizer = optimizer
        
        # BunDLe-Net model, as passed to the denoiser module
        self.bundlenet_model: BunDLeNet = denoiser.bundlenet_model
        
        # Loss function for the denoiser module training
        self.loss_fn : CompositeLoss = loss_fn.to(device)

        # Train Loader
        self.train_loader : DataLoader = train_loader

        # Test Loader
        self.test_loader : DataLoader = test_loader

        # Device 
        self.device = device

        # Training epochs
        self.num_epochs = num_epochs

        # Statistical fit flag and moments
        self.statistical_fit = statistical_fit
        
        # Statistical loss function
        self.statistical_loss_fn : StatisticalLossTerm | StatisticalCompositeLoss = statistical_loss_fn if statistical_fit else None
        
        # Fraction of training epochs for which the statistical fit loss will be applied
        self.statistical_epochs = statistical_epochs

        print("DenoiserTrainer is initialized.")

    def _freeze_bundlenet(self):
        """Freezes the weights of the BunDLe-Net model."""

        for param in tqdm(self.bundlenet_model.parameters(), desc="Freezing BunDLe-Net weights"):
            param.requires_grad = False
    
    def _train_epoch(self, include_statistics: bool = False) -> tuple[float, float]:
        """Trains the denoiser for a single epoch. If include_statistics is True, the statistical fit loss is also computed and optimized for within the same epoch.    
        If required, losses are recorded for both the pointwise loss and the statistical fit loss.

        Args:
            include_statistics (bool, optional): Whether to include statistical loss in training. Defaults to False.

        Returns:
            tuple[float, float]: Average training loss for the epoch, and average statistical loss for the epoch (if include_statistics is True, otherwise 0.0).
        """

        # Put the denoiser and BunDLe-Net model in the appropriate modes
        self.denoiser.train()
        self.denoiser.bundlenet_model.eval()

        tot_loss = 0.0


        if include_statistics:
            self.statistical_loss_fn.record_training()
        self.loss_fn.record_training()

        # Iterate over the training data and optimize the denoiser module
        for i, sample in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            if not include_statistics:
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

            # Accumulate loss for logging
            tot_loss += loss.item()

        avg_train_loss = tot_loss / len(self.train_loader)

        stat_loss_value = 0.0
        stat_loss = 0.0

        # If statistical fit is enable, reset the gradients and compute the statistical fit loss over the different sub-classes of the data.
        if include_statistics:
            denoised_all = []
            labels_all = []
            latent_all = []
            self.optimizer.zero_grad()
            
            # Denoise and collect training data.
            for sample in self.train_loader:
                denoised_state, re_abstracted_state = self.denoiser(sample[1])
                denoised_all.append(denoised_state)
                labels_all.append(sample[2])
                latent_all.append(re_abstracted_state)

            # Concatenate the denoised states, labels, and extract a dictionary containing relevant statistics or information for each class.
            # NOTE: The current implementation assumes that the statistical loss function can take in the entire training data for computing the loss. 
            # NOTE: The information which is extracted from each class depends on the type of statistical loss function which is used.

            denoised_X_train_all = torch.cat(denoised_all, dim=0)
            Y_train_all = torch.cat(latent_all, dim=0)
            B_train_all = torch.cat(labels_all, dim=0).view(-1).to(self.device)
            dictionary = self.statistical_loss_fn.build_conditioned_dictionary(denoised_X_train_all, Y_train_all, B_train_all)
            valid_classes = 0
            
            # Iterate over the classes, compute the loss, and optimize if there are enough samples for the class (so that the statistics are meaningful).
            for label in dictionary.keys():
                if dictionary[label][0].shape[0] < 2:
                    continue

                stat_loss += self.statistical_loss_fn(
                    DenoiserDistributionMatchingData(
                        conditioned_neuronal=dictionary[label][0],
                        conditioned_latent=dictionary[label][1],
                        label=label,
                        indicator='train'
                    )
                )
                valid_classes += 1

            if valid_classes > 0:
                stat_loss = stat_loss / valid_classes
                stat_loss.backward()
                self.optimizer.step()

                stat_loss_value = stat_loss.item()

        if include_statistics:
            self.statistical_loss_fn.handle_epoch_end()
        self.loss_fn.handle_epoch_end()

        # Finally, return the average pointwise loss and the average statistical fit loss for the epoch (if include_statistics is True, otherwise 0.0).
        # to sum up the epoch.
        return avg_train_loss, stat_loss_value
        

    def _test_epoch(self, include_statistics: bool = False) -> tuple[float, float]:
        """Runs a single test epoch for the denoiser. If include_statistics is true, the statistical loss is computed on the test set and is added to the logs

        Args:
            include_statistics (bool, optional): Whether to include statistical loss computation. Defaults to False.

        Returns:
            tuple[float, float]: Average test loss for the epoch, and average statistical loss for the epoch (if include_statistics is True, otherwise 0.0).
        """

        # Put the denoiser and BunDLe-Net model in the appropriate modes
        self.denoiser.eval()

        tot_loss = 0.0
        stat_loss_value = 0.0
        stat_loss = 0.0

        if include_statistics:
            self.statistical_loss_fn.record_testing()
        self.loss_fn.record_testing()

        denoised_all = []
        latent_all = []
        labels_all = []

        with torch.no_grad():
            for i, sample in enumerate(self.test_loader):
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
                
                tot_loss += loss.item()
                
                # If statistics are to be included, gather the data during the forward pass.
                if include_statistics:
                    denoised_all.append(denoised_state)
                    latent_all.append(sample[1])
                    labels_all.append(sample[2])

            if include_statistics:
                # Concatenate the denoised states, labels, and extract a dictionary containing relevant statistics or information for each class.
                denoised_X_test_all = torch.cat(denoised_all, dim=0)
                Y_test_all = torch.cat(latent_all, dim=0)
                B_test_all = torch.cat(labels_all, dim=0).view(-1).to(self.device)

                dictionary = self.statistical_loss_fn.build_conditioned_dictionary(denoised_X_test_all, Y_test_all, B_test_all)
                # And if there is enough data for the classes, compute the statistical loss for each class and its average.
                for label in dictionary.keys():
                    if dictionary[label][0].shape[0] < 2:
                        continue

                    stat_loss += self.statistical_loss_fn(
                        DenoiserDistributionMatchingData(
                            conditioned_neuronal=dictionary[label][0],
                            conditioned_latent=dictionary[label][1],
                            label=label,
                            indicator='test'
                        )
                    )

                stat_loss = stat_loss / len(dictionary.keys())
                stat_loss_value = stat_loss.item()

        avg_test_loss = tot_loss / len(self.test_loader)

        if include_statistics:
            self.statistical_loss_fn.handle_epoch_end()
        self.loss_fn.handle_epoch_end()

        # Finally, return the average pointwise loss and the average statistical fit loss for the epoch (if include_statistics is True, otherwise 0.0).
        return avg_test_loss, stat_loss_value


    def train(self):
        """
        Trains the denoiser module, assuming that the BunDLe-Net model is already trained.
        As a pre-processing, BunDLe-Net weights are frozen.

        If statistical_fit is enabled, a pre-processing step is performed to compute the relevant statistics for the statistical fit loss for the training and test sets. The training loop is modified to include the
        statistical fit loss in the optimization and logging starting from the epoch defined by statistical_epochs.
        """

        # First, freeze the weights of the BunDLe-Net model.
        self._freeze_bundlenet()
        
        # If the statistical fit is enabled...
        if self.statistical_fit:
            # ... compute the starting epoch
            self.statistical_fit_start = int(self.num_epochs * (1-self.statistical_epochs)) if self.statistical_fit else None
            # ... preprocess the datasets (by accessing the data loaders) to load the training and test data.
            self.statistical_loss_fn.preprocess_dataloaders(self.train_loader, self.test_loader)
            # ... and compute the relevant statistics or information.
            self.statistical_loss_fn.preprocess()

            print(f"Statistical fit enabled from epoch {self.statistical_fit_start}. Processing completed.")

        # Flag for including the statistical fit loss in the training loop. True after the epochs reach the statistical fit starting epoch, and False beforehand.
        include_statistics: bool = False

        # Iterate over the epochs, train and test the model.
        progress: tqdm = tqdm(range(1, self.num_epochs + 1), desc="Training", unit="epoch", smoothing=0.1)
        for epoch in progress:

            if self.statistical_fit and epoch == self.statistical_fit_start:
                include_statistics = True

            avg_train_loss, stat_train_loss = self._train_epoch(include_statistics)
            
            avg_test_loss, stat_test_loss = self._test_epoch(include_statistics)

            if self.statistical_fit and epoch >= self.statistical_fit_start:
                progress.set_postfix(train_loss=f"{avg_train_loss:.4f}", test_loss=f"{avg_test_loss:.4f}", pointwise_difference=f"{avg_train_loss - avg_test_loss:.4f}", stat_train_loss=f"{stat_train_loss:.4f}", stat_test_loss=f"{stat_test_loss:.4f}", stat_difference=f"{stat_train_loss - stat_test_loss:.4f}")
            else:
                progress.set_postfix(train_loss=f"{avg_train_loss:.4f}", test_loss=f"{avg_test_loss:.4f}", pointwise_difference=f"{avg_train_loss - avg_test_loss:.4f}")

    def get_pointwise_loss_handle(self):
        """Returns the handle to the pointwise loss function."""
        return self.loss_fn

    def get_statistical_loss_handle(self):
        """Returns the handle to the statistical fit loss function, if it is enabled."""
        if self.statistical_fit:
            return self.statistical_loss_fn
        else:
            raise ValueError("Statistical fit is not enabled for this DenoiserTrainer instance.")

    def summarize(self):
        """Summarizes the training configuration for the denoiser module."""
        summary = f"DEN_e{self.num_epochs}_{self.loss_fn.summarize()}"
        if self.statistical_fit:
            summary += f"_{self.statistical_loss_fn.summarize()}"
        return summary

def prepare_denoiser_data(
        X: np.ndarray,
        B: np.ndarray, 
        bundlenet_model: BunDLeNet = None, 
        device: torch.device = None, 
        train_split: float = 0.8, 
        batch_size: int = 8, 
        random_state: int = 42,
        force_behavioral_presence: bool = False,
        block_size: int = 128,
        train_indices_out: np.ndarray = None,
        test_indices_out: np.ndarray = None
        ) -> tuple[DataLoader, DataLoader]:
    """Prepares the training and test datasets for training and testing the denoiser

    Args:
        X (np.ndarray): Neuronal data in windowed format, of shape (num_samples, window_size, num_neurons). Currently, only window_size of 1 is supported, so the expected shape is (num_samples, 1, num_neurons).
        B (np.ndarray): Behavioral labels for the neuronal data, of shape (num_samples, 1).
        bundlenet_model (BunDLeNet, optional): The BunDLe-Net model to use for projecting the neuronal data into the latent space. Defaults to None.
        device (torch.device, optional): The device to use for training. Defaults to None.
        train_split (float, optional): The fraction of the data to use for training. Defaults to 0.8.
        batch_size (int, optional): The batch size for training and testing. Defaults to 8.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        force_behavioral_presence (bool, optional): Whether to enforce a no-leakage split of the data. Defaults to False.
        block_size (int, optional): The size of the blocks to use for the no-leakage split. Defaults to 128.

    Returns:
        Tuple[DataLoader, DataLoader]: The training and test data loaders.
    """
    
    assert bundlenet_model is not None, "BunDLe-Net model must be provided"
    assert device is not None, "Device must be provided"

    # Project the neuronal data into the latent space using the BunDLe-Net model
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float, device=device)
        B_tensor = torch.tensor(B, dtype=torch.float, device=device)
        latent_states = bundlenet_model.tau(X_tensor)

    # Create a TensorDataset for the denoiser, which includes the original neuronal data, the latent states, and the behavioral labels.
    denoiser_data = TensorDataset(
        X_tensor,
        latent_states,
        B_tensor        
    )
    print(len(denoiser_data), "samples prepared for the denoiser.")
    # Compute the sizes for the training and test sets. Then, build the dataloaders
    train_size = int(len(denoiser_data) * train_split)
    test_size = len(denoiser_data) - train_size

    train_dataset, test_dataset, train_indices, test_indices = _no_leakage_split(
        denoiser_data, 
        train_size=train_split, 
        random_state=random_state, 
        force_behavioral_presence=force_behavioral_presence, 
        block_size=block_size,
        balance_tolerance=0.1,
        max_correction_passes=10
    )

    if train_indices_out is not None:
        train_indices_out[:] = train_indices
    if test_indices_out is not None:
        test_indices_out[:] = test_indices
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader