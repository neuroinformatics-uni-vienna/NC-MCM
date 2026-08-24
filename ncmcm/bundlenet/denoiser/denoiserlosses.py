import torch
import torch.nn as nn
import numpy as np
from .denoiser_data import DenoiserData, DenoiserDistributionMatchingData


# Loss terms for the denoiser model. Each loss term is implemented as a separate class.
# There are two main types of loss terms: pointwise loss terms, which are computed on individual samples (or batches) and statistical loss terms, which are computed
# on the entire samples belonging to a subclass.
# Pointwise loss terms enforce reconstruction of the latent spaces, or regularization of the denoised neuronal activity.
# Statistical loss terms enforce the matching of statistical properties or distribution
#
#
# This file also contains the Recordable and LossRecorder classes, which are utilities to record the loss values during training and testing for later analysis and visualization.
# More specifically, Recordable defines the interface for recording losses during training and testing via a LossRecorder instance.
#
# The way that inheritance is organized is the following
#
# LossTerm (base class for all loss terms) -> nn.Module 
# CompositeLoss (combines multiple loss terms together) -> nn.Module, Recordable.
# StatisticalLossTerm (base class for statistical loss terms) -> nn.Module
#
# Any specific pointwise loss term inherits from LossTerm
# Instead, any specific statistical loss inherits from StatisticalLossTerm and Recordable.

class LossTerm(nn.Module):
    """Base class for loss terms used in the Denoiser model."""
    def __init__(self):
        super(LossTerm, self).__init__()

    def forward(self, sample: DenoiserData):
        """Computes the loss for a given sample. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    def name(self):
        """Returns a human-readable name for the loss term."""
        return self.__class__.__name__
    
    def summarize(self):
        """Provides a summary of the loss term for logging purposes."""
        return self.__class__.__name__

# Loss terms for the denoiser model

class MSELatentLoss(LossTerm):
    """Mean Squared Error loss on the latent space."""

    def __init__(self, weight=1.0):
        """Initializes the MSELatentLoss with a specified weight for the loss term. MSELatentLoss is the l2 loss between the original and reconstructed latent representations of 
        neuronal states through BunDLe-Net's tau function."""
        super(MSELatentLoss, self).__init__()
        self.mse_loss_fn = nn.MSELoss()
        self.weight = weight

    def summarize(self):
        return f"MSEL(w{self.weight})"
    
    def name(self):
        return f"MSELatentLoss"

    def forward(self, sample: DenoiserData):
        return self.weight * self.mse_loss_fn(sample.reconstructed_latent, sample.original_latent)

class LInftyNeuronalLoss(LossTerm):
    """L-infinity loss on the neuronal space."""
    def __init__(self, weight=0.1):
        """Initializes the LInftyNeuronalLoss with a specified weight for the loss term. LInftyNeuronalLoss is the l-infinity loss between the denoised and original states to enforce 
        that the maximum values match."""
        super(LInftyNeuronalLoss, self).__init__()
        self.weight = weight

    def summarize(self):
        return f"LInftyN(w{self.weight})"
    
    def name(self):
        return f"LInftyNeuronalLoss"

    def forward(self, sample: DenoiserData):
        return self.weight * torch.norm(sample.denoised_neuronal - sample.original_neuronal, p=float('inf'), dim=1).mean()

class LSENeuronalLoss(LossTerm):
    """Least Squares Error loss on the neuronal space."""
    def __init__(self, weight=0.1, beta=0.5):
        """Initializes the LSENeuronalLoss with a specified weight for the loss term. LSENeuronalLoss is the l2 loss between the denoised and original states to enforce that the 
        overall shape of the neuronal activity is preserved."""
        super(LSENeuronalLoss, self).__init__()
        self.weight = weight
        self.beta = beta

    def summarize(self):
        return f"LSEN(w{self.weight}, β{self.beta})"
    
    def name(self):
        return f"LSENeuronalLoss"

    def forward(self, sample: DenoiserData):
        # Esempio stabile con logsumexp nativo
        diff = torch.abs(sample.denoised_neuronal - sample.original_neuronal)
        return self.weight * torch.logsumexp(self.beta * diff, dim=1).mean() / self.beta


# Regularization terms
class L1NeuronalRegularization(LossTerm):
    """L1 regularization on the neuronal space."""
    def __init__(self, weight=0.1):
        """Initializes the L1NeuronalRegularization with a specified weight for the loss term. L1NeuronalRegularization is the l1 regularization on the denoised neuronal activity to encourage sparsity."""
        super(L1NeuronalRegularization, self).__init__()
        self.weight = weight

    def summarize(self):
        return f"L1NReg(w{self.weight})"

    def name(self):
        return f"L1NeuronalRegularization"

    def forward(self, sample: DenoiserData):
        return self.weight * torch.abs(sample.denoised_neuronal).mean()

class L2NeuronalRegularization(LossTerm):
    """L2 regularization on the neuronal space."""
    def __init__(self, weight=0.1):
        """Initializes the L2NeuronalRegularization with a specified weight for the loss term. L2NeuronalRegularization is the l2 regularization on the denoised neuronal activity to encourage smoothness."""
        super(L2NeuronalRegularization, self).__init__()
        self.weight = weight

    def summarize(self):
        return f"L2NReg(w{self.weight})"

    def name(self):
        return f"L2NeuronalRegularization"
    
    def forward(self, sample: DenoiserData):
        return self.weight * torch.square(sample.denoised_neuronal).mean()
    
class NeuronalSaliencyRegularization(LossTerm):
    """Loss term that encourages the denoised neuronal activity to be more salient for the behaviors of interest."""
    def __init__(self, weight=0.1, saliency_maps=None):
        """Initializes the NeuronalSaliencyRegularization with a specified weight for the loss term and a dictionary of saliency maps for each behavior. NeuronalSaliencyRegularization encourages the denoised neuronal activity to be more aligned with the provided saliency maps, which represent the importance of each neuron for the behaviors of interest."""
        super(NeuronalSaliencyRegularization, self).__init__()
        self.weight = weight
        self.saliency_maps = saliency_maps

        # Normalize the saliency maps to have values between 0 and 1
        if self.saliency_maps is not None:
            for key in self.saliency_maps:
                saliency_map = self.saliency_maps[key]
                if saliency_map.max() > 0:
                    self.saliency_maps[key] = saliency_map / saliency_map.max()

    def summarize(self):
        return f"SalReg(w{self.weight})"
    
    def name(self):
        return f"NeuronalSaliencyRegularization"

    def forward(self, sample: DenoiserData):
        if self.saliency_maps is None:
            raise ValueError("Saliency maps must be provided for NeuronalSaliencyRegularization.")
        
        # Assuming sample.behavioral_label is an integer index corresponding to the behavior
        behavior_label = sample.behavioral_label
        # print(f"Behavioral label for saliency regularization: {behavior_label}")
        
        # In case of batch training, we need to extract the relevant saliency map for each sample separately. Then,
        # we can compute the loss together.

        local_behaviors = behavior_label.cpu().numpy() 
        saliency_vectors = torch.stack([self.saliency_maps[behavior] for behavior in local_behaviors], dim=0).to(sample.denoised_neuronal.device)
        
        return self.weight * torch.mean(1 - torch.nn.functional.cosine_similarity(sample.denoised_neuronal, saliency_vectors, dim=1))


# Loss to compose all the loss and regularizationterms together

class Recordable():
    """Interface for recording losses during training and testing via a LossRecorder instance."""
    def __init__(self, *loss_terms, record_losses=False):
        """Initializes the Recordable instance with a list of loss terms and an optional flag to enable loss recording. If record_losses is True, a LossRecorder instance is created to track the losses during training and testing."""
        self.loss_recorder = LossRecorder(loss_terms) if record_losses else None
        self.record_losses = record_losses

    def record_training(self):
        """Switches the recording mode to training losses."""
        if self.record_losses:
            self.loss_recorder.record_training()
        else:
            raise ValueError("Loss recording was not enabled for this CompositeLoss.")

    def record_testing(self):
        """Switches the recording mode to testing losses."""
        if self.record_losses:
            self.loss_recorder.record_testing()
        else:
            raise ValueError("Loss recording was not enabled for this CompositeLoss.")
    
    def handle_epoch_end(self):
        """Handles the end of an epoch by averaging the cached losses and recording them in the LossRecorder instance."""
        if self.record_losses:
            self.loss_recorder.handle_epoch_end()
        else:
            raise ValueError("Loss recording was not enabled for this CompositeLoss.")

    def get_loss_recordings(self):
        """Returns the recorded training and testing losses from the LossRecorder instance."""
        if self.loss_recorder:
            return self.loss_recorder.get_training_losses(), self.loss_recorder.get_testing_losses()
        else:
            raise ValueError("Loss recording was not enabled for this CompositeLoss.")
        
    def cache_loss(self, loss_name, loss_value):
        """Caches a loss value in the LossRecorder instance under the specified loss name."""
        if self.record_losses:
            self.loss_recorder.cache_loss(loss_name, loss_value)
        else:
            raise ValueError("Loss recording was not enabled for this CompositeLoss.")
        
class CompositeLoss(nn.Module, Recordable):
    """Composite loss that combines multiple loss terms."""
    def __init__(self, *loss_terms, record_losses=False):
        print(f"Initializing CompositeLoss with loss terms: {[term.name() for term in loss_terms]} and record_losses={record_losses}")
        nn.Module.__init__(self)
        Recordable.__init__(self, *loss_terms, record_losses=record_losses)

        self.loss_terms = nn.ModuleList(loss_terms)

    def forward(self, sample: DenoiserData):
        total_loss = 0.0
        for loss_term in self.loss_terms:
            loss = loss_term(sample)
            total_loss += loss
            self.cache_loss(loss_term.name(), loss.item())
        
        self.cache_loss("TotalLoss", total_loss.item())
        return total_loss

    def summarize(self):
        summary = "".join([term.summarize() for term in self.loss_terms])
        return summary 
    
# Statistical loss terms for distribution matching and statistical fit. 
class StatisticalLossTerm(nn.Module):
    def __init__(self):
        """Initializes the StatisticalLossTerm. This is a base class for loss terms that enforce distribution matching or statistical fit of the denoised neuronal activity to the original neuronal activity.
        This class provides functions that preprocess the dataloaders to temporarily extract the original neuronal data before the actual pre-processing takes place."""
        nn.Module.__init__(self)
        self.conditioned_dictionary = None
        
    def summarize(self):
        return f"StatisticalLossTerm()"
    
    def name(self):
        return f"StatisticalLossTerm"
    
    def forward(self, sample: DenoiserDistributionMatchingData):
        raise NotImplementedError("Subclasses must implement the forward method for StatisticalLossTerm, which takes a DenoiserDistributionMatchingData sample and a current_label as input.")

    def preprocess(self):
        """Precompute any necessary statistics from the original and denoised neuronal data before training."""
        raise NotImplementedError("Subclasses must implement the preprocess method for StatisticalLossTerm, which takes a DenoiserDistributionMatchingData sample as input.")    

    def build_conditioned_dictionary(self, X_data: torch.Tensor, Y_data: torch.Tensor, B_labels: torch.Tensor):
        """Utility function to build a dictionary of neuronal data conditioned on the provided labels."""

        dictionary = {}
        for label in torch.unique(B_labels):
            label_mask = (B_labels == label)
            dictionary[label.item()] = (X_data[label_mask], Y_data[label_mask])

        return dictionary

    def preprocess_dataloaders(self, train_dataloader, test_dataloader):
        """Preprocesses the training and testing dataloaders to build a dictionary of neuronal data conditioned on the provided labels for both training and testing data."""
        all_training_data = []
        all_testing_data = []
        all_training_labels = []
        all_testing_labels = []
        all_latent_training_data = []
        all_latent_testing_data = []
        
        for sample in train_dataloader:
            all_training_data.append(sample[0])
            all_latent_training_data.append(sample[1])
            all_training_labels.append(sample[2])
            

        for sample in test_dataloader:
            all_testing_data.append(sample[0])
            all_latent_testing_data.append(sample[1])
            all_testing_labels.append(sample[2])

        X_train = torch.cat(all_training_data, dim=0)
        Y_train = torch.cat(all_latent_training_data, dim=0)
        B_train = torch.cat(all_training_labels, dim=0)
        train_dictionary = self.build_conditioned_dictionary(X_train, Y_train, B_train)

        X_test = torch.cat(all_testing_data, dim=0)
        Y_test = torch.cat(all_latent_testing_data, dim=0)
        B_test = torch.cat(all_testing_labels, dim=0)
        test_dictionary = self.build_conditioned_dictionary(X_test, Y_test, B_test)

        self.conditioned_dictionary = { label: {"train": train_dictionary[label], "test": test_dictionary[label]} for label in train_dictionary.keys() }


# Losses for distribution matching and statistical fit
class ConditionedNeuronalMomentMatching(StatisticalLossTerm, Recordable):
    """Loss term encouraging the denoised activity to match the statistics of the original activity."""

    def __init__(self, weight=0.1, moments_to_match=4, standardized_moments=True, record_losses=False, different_weights=False):
        """Initializes the ConditionedNeuronalMomentMatching loss term with specified parameters. This loss term encourages the denoised neuronal activity to match the moments of the original neuronal activity."""
        assert moments_to_match > 0, "moments_to_match must be a positive integer indicating how many moments to compute for the loss."

        StatisticalLossTerm.__init__(self)
        Recordable.__init__(self, self, record_losses=record_losses)

        self.weight = weight
        self.moments_to_match = moments_to_match
        self.standardized_moments = standardized_moments
        self.loss_fn = nn.MSELoss()
        self.moments_cache = {}
        self.different_weights = different_weights
        self.num_of_samples = 0
        self.samples_per_label = {}

    def summarize(self):
        return f"CondMomMatch(w{self.weight})"
    
    def name(self):
        return f"ConditionedNeuronalMomentMatching"

    def _compute_moments(self, data):
        """Internal function that computes the specified number of moments for the given data. If standardized_moments is True, the data is standardized before computing the moments."""
        moments = []

        if self.standardized_moments:
            data = (data - data.mean(dim=0)) / (data.std(dim=0) + 1e-6)

        for moment in range(1, self.moments_to_match + 1):
            if moment == 1:
                moments.append(data.mean(dim=0))
            else:
                moments.append(torch.mean((data - data.mean(dim=0)) ** moment, dim=0))

        return moments

    def preprocess(self):
        """Precompute the moments for each condition in the original and denoised neuronal data."""
        assert self.conditioned_dictionary is not None, "Conditioned dictionary must be built before calling preprocess."

        for label, data in self.conditioned_dictionary.items():
            self.moments_cache[label] = {
                "train": self._compute_moments(data["train"][0]),
                "test": self._compute_moments(data["test"][0])
            }
            self.num_of_samples += data["train"][0].shape[0] + data["test"][0].shape[0]
            self.samples_per_label[label] = self.samples_per_label.get(label, 0) + data["train"][0].shape[0] + data["test"][0].shape[0]
        self.conditioned_dictionary = None  # Clear the original data to save memory after computing moments

    def forward(self, sample: DenoiserDistributionMatchingData):
        """Computes the moment matching loss for a given sample of conditioned neuronal data. """
        assert self.moments_cache is not None, "Moments cache must be computed before calling forward on ConditionedNeuronalMomentMatching loss."
        assert sample.conditioned_neuronal is not None, "Conditioned neuronal data must be provided in the sample for ConditionedNeuronalMomentMatching loss."
        assert sample.label is not None, "Label must be provided in the sample for ConditionedNeuronalMomentMatching loss."
        assert sample.indicator is not None, "Indicator must be provided in the sample for ConditionedNeuronalMomentMatching loss."

        denoised_moments = self._compute_moments(sample.conditioned_neuronal)
        original_moments = self.moments_cache[sample.label][sample.indicator]

        

        total_loss = 0.0
        for moment in range(0, min(len(denoised_moments), len(original_moments))):
            total_loss += self.loss_fn(denoised_moments[moment], original_moments[moment])

        if self.different_weights:
            # the coefficient is the inverse of the fraction of the ratio of behavioral samples in the batch to the total number of behavioral samples in the dataset. 
            all_samples = self.num_of_samples
            behavior_samples = self.samples_per_label.get(sample.label, 0)

            coefficient = 1.0 / (behavior_samples / (all_samples + 1e-6))  # Avoid division by zero
            total_loss *= coefficient

        if self.record_losses:
            self.loss_recorder.cache_loss(self.name(), self.weight * total_loss.item())


        return self.weight * total_loss 


class StatisticalCompositeLoss(nn.Module, Recordable):
    """Composite loss that combines multiple statistical loss terms."""
    def __init__(self, *loss_terms, record_losses=False):
        print(f"Initializing StatisticalCompositeLoss with loss terms: {[term.name() for term in loss_terms]} and record_losses={record_losses}")
        nn.Module.__init__(self)
        Recordable.__init__(self, *loss_terms, record_losses=record_losses)

        self.loss_terms = nn.ModuleList(loss_terms)

    def forward(self, sample: DenoiserDistributionMatchingData):
        total_loss = 0.0
        for loss_term in self.loss_terms:
            loss = loss_term(sample)
            total_loss += loss
            if self.record_losses:
                self.cache_loss(loss_term.name(), loss.item())
        
        if self.record_losses:
            self.cache_loss("TotalLoss", total_loss.item())

        return total_loss

    def preprocess_dataloaders(self, train_dataloader, test_dataloader):
        """Preprocesses the training and testing dataloaders to build a dictionary of neuronal data conditioned on the provided labels for both training and testing data."""
        for loss_term in self.loss_terms:
            loss_term.preprocess_dataloaders(train_dataloader, test_dataloader)
        
    def preprocess(self):
        """Precompute any necessary statistics from the original and denoised neuronal data before training."""
        for loss_term in self.loss_terms:
            loss_term.preprocess()

    def summarize(self):
        summary = "(".join([term.summarize() for term in self.loss_terms]) + ")"
        return summary

    def build_conditioned_dictionary(self, X_data: torch.Tensor, Y_data: torch.Tensor, B_labels: torch.Tensor):
        """Utility function to build a dictionary of neuronal data conditioned on the provided labels."""
        dictionaries = []
        for loss_term in self.loss_terms:
            dictionary = loss_term.build_conditioned_dictionary(X_data, Y_data, B_labels)
            dictionaries.append(dictionary)
        
        # check if all dictionaries have the same keys
        keys = [set(dictionary.keys()) for dictionary in dictionaries]
        if not all(k == keys[0] for k in keys):
            raise ValueError("All loss terms must have the same set of labels in their conditioned dictionaries.")

        return dictionaries[0]  # return the first dictionary, since all are the same


class StatNoLatentCollapseSTDDevRegularization(StatisticalLossTerm, Recordable):
    """Loss term that encourages the standard deviation of the denoised latent representations to be above a certain threshold, preventing latent collapse."""
    def __init__(self, weight=0.1, stddev_threshold=0.1, record_losses=False):
        """Initializes the StatlNoLatentCollapseSTDDevRegularization with a specified weight for the loss term and a threshold for the standard deviation. StatisticalNoLatentCollapseSTDDevRegularization encourages the standard deviation of the denoised latent representations to be above the specified threshold, preventing latent collapse."""
        super(StatNoLatentCollapseSTDDevRegularization, self).__init__()
        self.weight = weight
        self.stddev_threshold = stddev_threshold
        self.record_losses = record_losses

    def summarize(self):
        return f"StatNoLatentCollapseSTDDevRegularization(weight={self.weight}, stddev_threshold={self.stddev_threshold})"

    def name(self):
        return f"StatNoLatentCollapseSTDDevRegularization"
    
    def preprocess(self):
        pass
    
    def forward(self, sample: DenoiserDistributionMatchingData):
        latent_representations = sample.conditioned_latent

        stddev_latent = torch.std(latent_representations, dim=0) + 1e-6
        collapse_penalty = torch.mean(torch.relu(self.stddev_threshold - stddev_latent))

        if self.record_losses:
            self.loss_recorder.cache_loss(self.name(), self.weight * collapse_penalty.item())

        return self.weight * collapse_penalty        

class LossRecorder:
    """Utility class to record and summarize loss values during training."""
    def __init__(self, loss_terms):
        """Initializes the LossRecorder with a list of loss terms. It creates separate dictionaries to store the history of training and testing losses for each loss term,
          as well as a buffer to cache losses during an epoch before averaging and recording them at the end of the epoch."""
        self.train_loss_history = { term.name(): [] for term in loss_terms }
        self.train_loss_history["TotalLoss"] = []

        self.loss_buffer = { term.name(): [] for term in loss_terms} 
        self.loss_buffer["TotalLoss"] = []
    

        self.test_loss_history = { term.name(): [] for term in loss_terms}
        self.test_loss_history["TotalLoss"] = []
        
        self.current_recorded = self.train_loss_history

        self.loss_terms = loss_terms

    def cache_loss(self, loss_name, loss_value):
        """Caches a loss value in the buffer for the specified loss name. This method is called during the forward passes
        to store individual loss values before averaging them at the end of the epoch."""
        if loss_name in self.loss_buffer:
            self.loss_buffer[loss_name].append(loss_value)
        else:
            raise ValueError(f"Loss name '{loss_name}' not recognized in LossRecorder.")

    def record_training(self):
        """Switches the recording mode to training losses."""
        self.current_recorded = self.train_loss_history
    
    def record_testing(self):
        """Switches the recording mode to testing losses."""
        self.current_recorded = self.test_loss_history

    def handle_epoch_end(self):
        """Handles the end of an epoch by averaging the cached losses and recording them in the LossRecorder instance. This method should be called at the end of each epoch to summarize the losses for that epoch."""
        for loss_name in self.loss_buffer:
            if self.loss_buffer[loss_name]:  # Only record if there are values
                avg_loss = np.mean(self.loss_buffer[loss_name])
                self.current_recorded[loss_name].append(avg_loss)
                self.loss_buffer[loss_name] = []  # Clear buffer after recording

    def record(self, loss_name, loss_value):
        """Records a loss value for the specified loss name in the current recording mode (training or testing). This method can be used to record losses at any point during training or testing, and it will store them in the appropriate history based on the current recording mode."""
        if loss_name in self.current_recorded:
            self.current_recorded[loss_name].append(loss_value)
        else:
            raise ValueError(f"Loss name '{loss_name}' not recognized in LossRecorder.")
        
    def get_training_losses(self):
        """Returns the recorded training losses for all loss terms as a dictionary."""
        return self.train_loss_history.copy()
    
    def get_testing_losses(self):
        """Returns the recorded testing losses for all loss terms as a dictionary."""
        return self.test_loss_history.copy()    