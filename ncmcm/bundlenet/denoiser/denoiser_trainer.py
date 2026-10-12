from ncmcm.bundlenet.denoiser.denoiser import Denoiser
from ncmcm.bundlenet.bundlenet import BunDLeNet
from ncmcm.bundlenet.denoiser.denoiserlosses import *
from .printer import _make_progress, print_success, print_error, print_warning, print_status, print_split_summary

from .denoiser_data import DenoiserData
from .denoiserlosses import CompositeLoss, LossTerm, StatisticalCompositeLoss, StatisticalLossTerm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from .temporal_split import _no_leakage_split


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
                min_samples_for_stat_relevance = 30,
                correction_epochs: int = 0
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
        assert device is not None, "Device must be provided"
        assert statistical_fit is not None, "Statistical fit flag must not be None"
        assert statistical_epochs >= 0 and statistical_epochs <= 1, "statistical_epochs must be a float between 0 and 1 representing the fraction of training epochs for which the statistical fit loss will be applied."

        # Denoiser module
        self.denoiser : Denoiser = denoiser

        # Window size
        self.window_size = self.denoiser.window
        
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

        # Minimum number of samples required for a class to be considered relevant for statistical loss computation
        self.min_samples_for_stat_relevance = min_samples_for_stat_relevance

        # Train and test indices for the data loaders
        self.train_indices = None

        # Test indices for the data loaders
        self.test_indices = None

        # Whether the correction fit is enabled or not
        self.is_correction_enabled = False

        # Number of epochs for which the correction term will be applied in the second training phase
        self.correction_epochs = correction_epochs

        print_success(message="DenoiserTrainer Initialized", details={"epochs": self.num_epochs, "correction_epochs": self.correction_epochs})

    def _freeze_bundlenet(self):
        """Freezes the weights of the BunDLe-Net model."""
        for param in self.bundlenet_model.parameters():
            param.requires_grad = False

    def _freeze_denoiser(self):
        """Freezes the weights of the Denoiser model."""

        for param in self.denoiser.parameters():
            param.requires_grad = False

        # Unfreeze only the weights of the active correction term
        for param in self.denoiser.activation_correction.parameters():
            param.requires_grad = True
        
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
        tot_correction_loss = 0.0
        loss = 0.0

        if include_statistics:
            self.statistical_loss_fn.record_training()
        self.loss_fn.record_training()

        if self.is_correction_enabled:
            self.denoiser.activation_correction.bundlenet_tau.eval()
            self.denoiser.activation_correction.record_training()

        # Iterate over the training data and optimize the denoiser module
        for i, sample in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            if not include_statistics:
                self.optimizer.zero_grad()
            

            x_t = sample[0]
            y_t = sample[1]
            b_t = sample[2]

            x_t_plus_1 = sample[3]
            y_t_plus_1 = sample[4]

            x_t_minus_1 = sample[5]
            y_t_minus_1 = sample[6]

            x_t_minus_2 = sample[7]
            y_t_minus_2 = sample[8]

            indicator = sample[9]

            denoised_copy = None

            # Forward pass through the denoiser

            # Compute loss between the re-abstracted state and the original latent state
            if not self.is_correction_enabled:
                x_t_denoised, y_t_denoised, correction_t, _ = self.denoiser(y_t, x_t, x_t_minus_1, indicator, self.is_correction_enabled)
                x_t_plus_1_denoised = None
                with torch.no_grad():
                    x_t_plus_1_denoised, _, _, _ = self.denoiser(y_t_plus_1, x_t_plus_1, x_t, torch.zeros_like(indicator), self.is_correction_enabled)
                loss = self.loss_fn(DenoiserData(
                    original_neuronal=x_t,
                    original_latent=y_t,
                    original_next_neuronal=x_t_plus_1,
                    denoised_neuronal=x_t_denoised,
                    denoised_next_neuronal=x_t_plus_1_denoised,
                    reconstructed_latent=y_t_denoised,
                    behavioral_label=b_t
                ))
                loss.backward()
            else:
                correction_t_prev = None
                x_t_minus_1_denoised = None

                x_t_corrected, y_t_denoised, correction_t, f_xt = self.denoiser(y_t, x_t, x_t_minus_1, indicator, self.is_correction_enabled)
                
                with torch.no_grad():
                    x_t_prev_corrected, _, correction_t_prev, f_xt_prev = self.denoiser(y_t_minus_1, x_t_minus_1, x_t_minus_2, indicator, self.is_correction_enabled)

                correction_loss = self.denoiser.activation_correction.loss(
                    original_xt=x_t,
                    original_xt_prev=x_t_minus_1,
                    f_xt=f_xt,
                    f_xt_prev=f_xt_prev,
                    correction_t=correction_t,
                    correction_t_prev=correction_t_prev,
                    is_first=indicator
                )
                correction_loss.backward()

            # Backward pass and optimization step
            self.optimizer.step()

            # Accumulate loss for logging

            if not self.is_correction_enabled:
                tot_loss += loss.item()
            else:
                tot_correction_loss += correction_loss.item()

        avg_train_loss = tot_loss / len(self.train_loader)
        avg_train_correction_loss = tot_correction_loss / len(self.train_loader)

        stat_loss_value = 0.0
        # stat_loss = 0.0

        # # If statistical fit is enable, reset the gradients and compute the statistical fit loss over the different sub-classes of the data.
        # if include_statistics:
        #     denoised_all = []
        #     labels_all = []
        #     latent_all = []
        #     self.optimizer.zero_grad()
            
        #     # Denoise and collect training data.
        #     for sample in self.train_loader:
        #         denoised_state, re_abstracted_state = self.denoiser(sample[1], sample[0], sample[5], sample[6])  
        #         denoised_state = denoised_state.reshape(sample[0].shape[0], self.window_size, -1)

        #         denoised_all.append(denoised_state[:, self.window_size-1, :])
        #         labels_all.append(sample[2])
        #         latent_all.append(re_abstracted_state)

        #     # Concatenate the denoised states, labels, and extract a dictionary containing relevant statistics or information for each class.
        #     # NOTE: The current implementation assumes that the statistical loss function can take in the entire training data for computing the loss. 
        #     # NOTE: The information which is extracted from each class depends on the type of statistical loss function which is used.

        #     denoised_X_train_all = torch.cat(denoised_all, dim=0)
        #     Y_train_all = torch.cat(latent_all, dim=0)
        #     B_train_all = torch.cat(labels_all, dim=0).view(-1).to(self.device)
        #     dictionary = self.statistical_loss_fn.build_conditioned_dictionary(denoised_X_train_all, Y_train_all, B_train_all)
        #     valid_classes = 0
            
        #     # Iterate over the classes, compute the loss, and optimize if there are enough samples for the class (so that the statistics are meaningful).
        #     for label in dictionary.keys():

        #         if label in self.statistical_loss_fn.labels_to_exclude:
        #             continue

        #         stat_loss += self.statistical_loss_fn(
        #             DenoiserDistributionMatchingData(
        #                 conditioned_neuronal=dictionary[label][0],
        #                 conditioned_latent=dictionary[label][1],
        #                 label=label,
        #                 indicator='train'
        #             )
        #         )
        #         valid_classes += 1

        #     if valid_classes > 0:
        #         stat_loss = stat_loss / valid_classes
        #         stat_loss.backward()
        #         self.optimizer.step()

        #         stat_loss_value = stat_loss.item()

        # if include_statistics:
        #     self.statistical_loss_fn.handle_epoch_end()
        self.loss_fn.handle_epoch_end()
        if self.is_correction_enabled:
            self.denoiser.activation_correction.handle_epoch_end()

        # Finally, return the average pointwise loss and the average statistical fit loss for the epoch (if include_statistics is True, otherwise 0.0).
        # to sum up the epoch.
        return avg_train_loss, stat_loss_value, avg_train_correction_loss
        

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
        tot_correction_loss = 0.0
        loss = 0.0
        correction_loss = 0.0
        # stat_loss = 0.0

        if include_statistics:
            self.statistical_loss_fn.record_testing()
        self.loss_fn.record_testing()

        if self.is_correction_enabled:
            self.denoiser.activation_correction.record_testing()

        denoised_all = []
        latent_all = []
        labels_all = []

        with torch.no_grad():
            for i, sample in enumerate(self.test_loader):

                x_t = sample[0]
                y_t = sample[1]
                b_t = sample[2]

                x_t_plus_1 = sample[3]
                y_t_plus_1 = sample[4]

                x_t_minus_1 = sample[5]
                y_t_minus_1 = sample[6]

                x_t_minus_2 = sample[7]
                y_t_minus_2 = sample[8]

                indicator = sample[9]

                # Forward pass through the denoiser

                if not self.is_correction_enabled:
                    x_t_denoised, y_t_denoised, _, _ = self.denoiser(y_t, x_t, x_t_minus_1, indicator, self.is_correction_enabled)
                    x_t_plus_1_denoised, _, _, _ = self.denoiser(y_t_plus_1, x_t_plus_1, x_t, indicator, self.is_correction_enabled)
                    loss = self.loss_fn(DenoiserData(
                        original_neuronal=x_t,
                        original_next_neuronal=x_t_plus_1,
                        original_latent=y_t,
                        denoised_neuronal=x_t_denoised,
                        denoised_next_neuronal=x_t_plus_1_denoised,
                        reconstructed_latent=y_t_denoised,
                        behavioral_label=b_t,
                        ))
                else:
                    x_t_corrected, y_t_denoised, correction_t, f_xt = self.denoiser(y_t, x_t, x_t_minus_1, indicator, self.is_correction_enabled)
                    x_t_minus_1_denoised, _, correction_t_prev, f_xt_prev = self.denoiser(y_t_minus_1, x_t_minus_1, x_t_minus_2, indicator, self.is_correction_enabled)
                    correction_loss = self.denoiser.activation_correction.loss(
                        original_xt=x_t,
                        original_xt_prev=x_t_minus_1,
                        f_xt=f_xt,
                        f_xt_prev=f_xt_prev,
                        correction_t=correction_t,
                        correction_t_prev=correction_t_prev,
                        is_first=indicator
                    )

                if not self.is_correction_enabled:
                    tot_loss += loss.item()
                else:
                    tot_correction_loss += correction_loss.item()
                
            #     # If statistics are to be included, gather the data during the forward pass.
            #     if include_statistics:
            #         denoised_all.append(denoised_neuronal[:, self.window_size-1, :])
            #         latent_all.append(re_abstracted_state)
            #         labels_all.append(sample[2])

            # if include_statistics:
            #     # Concatenate the denoised states, labels, and extract a dictionary containing relevant statistics or information for each class.
            #     denoised_X_test_all = torch.cat(denoised_all, dim=0)
            #     Y_test_all = torch.cat(latent_all, dim=0)
            #     B_test_all = torch.cat(labels_all, dim=0).view(-1).to(self.device)

            #     dictionary = self.statistical_loss_fn.build_conditioned_dictionary(denoised_X_test_all, Y_test_all, B_test_all)
            #     # And if there is enough data for the classes, compute the statistical loss for each class and its average.
            #     valid_classes = 0
            #     for label in dictionary.keys():

            #         if label in self.statistical_loss_fn.labels_to_exclude:
            #             continue

            #         stat_loss += self.statistical_loss_fn(
            #             DenoiserDistributionMatchingData(
            #                 conditioned_neuronal=dictionary[label][0],
            #                 conditioned_latent=dictionary[label][1],
            #                 label=label,
            #                 indicator='test'
            #             )
            #         )
            #         valid_classes += 1

            #     stat_loss = stat_loss / valid_classes if valid_classes > 0 else 0
            #     stat_loss_value = stat_loss.item()

        avg_test_loss = tot_loss / len(self.test_loader) 
        avg_correction_loss = tot_correction_loss / len(self.test_loader)

        # if include_statistics:
        #     self.statistical_loss_fn.handle_epoch_end()

        self.loss_fn.handle_epoch_end()
        if self.is_correction_enabled:
            self.denoiser.activation_correction.handle_epoch_end()

        # Finally, return the average pointwise loss and the average statistical fit loss for the epoch (if include_statistics is True, otherwise 0.0).
        return avg_test_loss, stat_loss_value, avg_correction_loss


    def train(self):
        """
        Trains the denoiser module, assuming that the BunDLe-Net model is already trained.
        As a pre-processing, BunDLe-Net weights are frozen.

        If statistical_fit is enabled, a pre-processing step is performed to compute the relevant statistics for the statistical fit loss for the training and test sets. The training loop is modified to include the
        statistical fit loss in the optimization and logging starting from the epoch defined by statistical_epochs.
        """
        assert self.train_loader is not None, "Train loader must be provided for training."
        assert self.test_loader is not None, "Test loader must be provided for testing."

        # First, freeze the weights of the BunDLe-Net model.
        self._freeze_bundlenet()

        # Preprocess the frequencies of behaviors for the losses
        print_status(message="Preprocessing training and test data for loss computation")
        self.loss_fn.preprocess_dataloaders(self.train_loader, self.test_loader, self.window_size)
        
        # If the statistical fit is enabled...
        if self.statistical_fit:
            # ... compute the starting epoch
            self.statistical_fit_start = int(self.num_epochs * (1-self.statistical_epochs)) if self.statistical_fit else None
            # ... preprocess the datasets (by accessing the data loaders) to load the training and test data.
            self.statistical_loss_fn.preprocess_dataloaders(self.train_loader, self.test_loader, self.window_size)
            # ... and compute the relevant statistics or information.
            self.statistical_loss_fn.preprocess()

            print_status(message="Statistical fit enabled", details={"from_epoch": self.statistical_fit_start})

        # Flag for including the statistical fit loss in the training loop. True after the epochs reach the statistical fit starting epoch, and False beforehand.
        include_statistics: bool = False


        with _make_progress() as progress:
            task = progress.add_task(
                "Training", total=self.num_epochs, metrics=""
            )

            for epoch in range(1, self.num_epochs + 1):
                if self.statistical_fit and epoch == self.statistical_fit_start:
                    include_statistics = True

                avg_train_loss, stat_train_loss, _ = self._train_epoch(include_statistics)
                avg_test_loss, stat_test_loss, _ = self._test_epoch(include_statistics)

                metrics = (
                    f"[green]Train {avg_train_loss:.6f}[/green]",
                    f"[yellow]Test {avg_test_loss:.6f}[/yellow]",
                    f"[dim]Δ {avg_train_loss - avg_test_loss:.4f}[/dim]"
                )

                if self.statistical_fit and epoch >= self.statistical_fit_start:
                    metrics += (
                        f" [dim]|[/dim] "
                        f"[green]stat_train {stat_train_loss:.4f}[/green] "
                        f"[yellow]stat_test {stat_test_loss:.4f}[/yellow] "
                        f"[dim]stat_Δ {stat_train_loss - stat_test_loss:.4f}[/dim]"
                    )

                progress.update(task, advance=1, metrics=metrics)

        if self.denoiser.add_correction:
            self._freeze_denoiser()
            self.is_correction_enabled = True
            self.loss_fn.record_losses = False

            with _make_progress() as progress:
                task = progress.add_task(
                    "Training Correction", total=self.correction_epochs, metrics=""
                )

                for epoch in range(self.correction_epochs):

                    _, _, avg_train_correction_loss = self._train_epoch(include_statistics=False)
                    _, _, avg_test_correction_loss = self._test_epoch(include_statistics=False)

                    progress.update(
                        task,
                        advance=1,
                        metrics=(
                            f"[green]train {avg_train_correction_loss:.4f}[/green] "
                            f"[yellow]test {avg_test_correction_loss:.4f}[/yellow]"
                        ),
                    )

                    
    def train_pipeline(
        self,
        bundlenet_train_handle: callable,
        bundlenet_train_config: dict,
        denoising_data_prep_config: dict = None,
    ):
        """Trains the entire pipeline by first calling a BunDLe-Net training handle by supplying the parameter
        configuration. Then, trains the denoiser with the parameters provided in the constructor

        Args:
            bundlenet_train_handle (callable): Handle to the BunDLe-Net training function
            bundlenet_train_config (dict): Training configuration for the BunDLe-Net training function

        Returns:
            Any: Output of the BunDLe-Net training handle, which may contain relevant information about the trained model, training history, or other outputs.
        """
        # Start by training the BunDLe-Net model using the provided training handle and configuration.
        # Then, train the denoiser module using the train method of this class.

        printable_bundlenet_config = {k: v for k, v in bundlenet_train_config.items() if (k != 'x_train' and k != 'b_train_1')}
        printable_denoising_config = {k: v for k, v in denoising_data_prep_config.items() if (k != 'X' and k != 'B')}
        print_status(message="Training Pipeline Started", details={"bundlenet_train_config": printable_bundlenet_config, "denoising_data_prep_config": printable_denoising_config})
        bundlenet_train_out = bundlenet_train_handle(**bundlenet_train_config)
        print_status(message="BunDLe-Net training completed")


        self.bundlenet_model.eval()
        train_indices, test_indices = [], []

        try: 
            train_indices = denoising_data_prep_config['train_indices_out']
            test_indices = denoising_data_prep_config['test_indices_out']
        except KeyError:
            print_warning(message="Train and test indices not found in denoising_data_prep_config. They will be generated during data preparation.")
            denoising_data_prep_config['train_indices_out'] = train_indices
            denoising_data_prep_config['test_indices_out'] = test_indices
        finally:
            pass 



        train_loader, test_loader = prepare_denoiser_data(
            **denoising_data_prep_config,
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_indices = train_indices.copy() if train_indices is not None else None
        self.test_indices = test_indices.copy() if test_indices is not None else None

        assert train_indices is not None and test_indices is not None, "Internal Error: Train and test indices not be prepared for the denoiser training."

        print_status(message="Train & Test Split Computed", details={"train_indices": len(train_indices), "test_indices": len(test_indices)})
        self.train()

        # Returns the output of the BunDLe-Net training handle, which may contain relevant information about the trained model, training history, or other outputs.
        return bundlenet_train_out

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
        window_size: int = None,
        no_leakage_strategy: str = 'lossy',
        train_indices_out: np.ndarray = None,
        test_indices_out: np.ndarray = None,
        strict_balance: bool = False,
        strategy_kwargs: dict = None
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
    assert window_size is not None, "Window size must be provided"

    print_status(message="Preparing data for the denoiser")

    X_t = X[:, 0, :, :]
    # Project the neuronal data into the latent space using the BunDLe-Net model
    with torch.no_grad():
        X_t_tensor = torch.tensor(X_t, dtype=torch.float, device=device)
        X_t_plus_1_tensor = torch.tensor(X[:, 1, :, :], dtype=torch.float, device=device)
        B_tensor = torch.tensor(B, dtype=torch.float, device=device)

        latent_states = bundlenet_model.tau(X_t_tensor)
        latent_t_plus_1_tensor = bundlenet_model.T_Y(latent_states)


    # Create a TensorDataset for the denoiser, which includes the original neuronal data, the latent states, and the behavioral labels.
    
    print_status(message="Creating TensorDataset for the denoiser", details={"Xt tensor shape": X_t_tensor.shape, "Latent states shape": latent_states.shape, "B tensor shape": B_tensor.shape})

    # Create a tensor with the same shape of X_t_tensor
    X_t_minus_1_tensor = torch.zeros_like(X_t_tensor, dtype=torch.float, device=device)
    # Set the first element of X_t_minus_1_tensor to be 0
    X_t_minus_1_tensor[0] = 0
    # Copy the remaining elements of X_t_tensor to X_t_minus_1_tensor, shifted by one time step
    X_t_minus_1_tensor[1:] = X_t_tensor[:-1]

    X_t_minus_2_tensor = torch.zeros_like(X_t_minus_1_tensor, dtype=torch.float, device=device)
    # Set the first two elements of X_t_minus_2_tensor to be 0
    X_t_minus_2_tensor[0] = 0
    X_t_minus_2_tensor[1] = 0
    # Copy the remaining elements of X_t_tensor to X_t_minus_2_tensor, shifted by two time steps
    X_t_minus_2_tensor[2:] = X_t_minus_1_tensor[2:]

    is_first_tensor = torch.zeros_like(X_t_tensor, dtype=torch.float, device=device)
    is_first_tensor[1] = 0
    is_first_tensor[2] = 0

    with torch.no_grad():
        latent_t_minus_1_tensor = bundlenet_model.tau(X_t_minus_1_tensor)
        latent_t_minus_2_tensor = bundlenet_model.tau(X_t_minus_2_tensor)

    denoiser_data = TensorDataset(
        X_t_tensor,                       # 0. Original neuronal state at time t
        latent_states,                    # 1. Latent state at time t
        B_tensor,                         # 2. Behavioral label
        X_t_plus_1_tensor,                # 3. Original neuronal state at time t+1
        latent_t_plus_1_tensor,           # 4. Latent state at time t+1 (advanced)
        X_t_minus_1_tensor,               # 5. Original neuronal state at time t-1
        latent_t_minus_1_tensor,          # 6. Latent state at time t-1
        X_t_minus_2_tensor,               # 7. Original neuronal state at time t-2
        latent_t_minus_2_tensor,          # 8. Latent state at time t-2
        is_first_tensor,                  # 9. Indicator tensor for the first two time steps
    )

    train_dataset, test_dataset, train_indices, test_indices = _no_leakage_split(
        denoiser_data, 
        train_size=train_split, 
        random_state=random_state, 
        force_behavioral_presence=force_behavioral_presence, 
        block_size=block_size,
        balance_tolerance=0.1,
        max_correction_passes=10,
        window_size=window_size,
        strategy=no_leakage_strategy,
        strategy_kwargs=strategy_kwargs
    )

    train_labels = B[train_indices]
    test_labels = B[test_indices]
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    print_split_summary(B, train_indices, test_indices, title="Splits per behavioral class (denoiser data)")

    if strict_balance:
        # Drop samples which are not represented in both the train and test splits to ensure strict balance.
        train_labels_set = set(train_labels.flatten())
        test_labels_set = set(test_labels.flatten())
        common_labels = train_labels_set.intersection(test_labels_set)
        if len(common_labels) < len(train_labels_set) or len(common_labels) < len(test_labels_set):
            print_warning(message="Strict balance is enabled, but some behavioral classes are not represented in both the train and test splits. These classes will be dropped from both splits to ensure strict balance.", details={"common_labels": common_labels, "train_labels_set": train_labels_set, "test_labels_set": test_labels_set})
            train_indices = np.array([idx for idx in train_indices if B[idx] in common_labels])
            test_indices = np.array([idx for idx in test_indices if B[idx] in common_labels])

    if train_indices_out is not None:
        train_indices_out[:] = train_indices
    if test_indices_out is not None:
        test_indices_out[:] = test_indices
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader