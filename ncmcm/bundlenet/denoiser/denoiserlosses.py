import torch
import torch.nn as nn
import numpy as np
from .denoiser_data import DenoiserData


class LossTerm(nn.Module):
    """Base class for loss terms used in the Denoiser model."""
    def __init__(self):
        super(LossTerm, self).__init__()

    def forward(self, sample: DenoiserData):
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
        super(MSELatentLoss, self).__init__()
        self.mse_loss_fn = nn.MSELoss()
        self.weight = weight

    def summarize(self):
        return f"MSELatentLoss(weight={self.weight})"
    
    def name(self):
        return f"MSELatentLoss"

    def forward(self, sample: DenoiserData):
        return self.weight * self.mse_loss_fn(sample.reconstructed_latent, sample.original_latent)

class L1NeuronalLoss(LossTerm):
    """L1 loss on the neuronal space."""
    def __init__(self, weight=0.1):
        super(L1NeuronalLoss, self).__init__()
        self.l1_loss_fn = nn.L1Loss()
        self.weight = weight

    def summarize(self):
        return f"L1NeuronalLoss(weight={self.weight})"
    
    def name(self):
        return f"L1NeuronalLoss"

    def forward(self, sample: DenoiserData):
        return self.weight * self.l1_loss_fn(sample.denoised_neuronal, torch.zeros_like(sample.denoised_neuronal))

class L2NeuronalLoss(LossTerm):
    """L2 loss on the neuronal space."""
    def __init__(self, weight=0.1):
        super(L2NeuronalLoss, self).__init__()
        self.l2_loss_fn = nn.MSELoss()
        self.weight = weight

    def summarize(self):
        return f"L2NeuronalLoss(weight={self.weight})"
    
    def name(self):
        return f"L2NeuronalLoss"

    def forward(self, sample: DenoiserData):
        return self.weight * self.l2_loss_fn(sample.denoised_neuronal, torch.zeros_like(sample.denoised_neuronal))

class LInftyNeuronalLoss(LossTerm):
    """L-infinity loss on the neuronal space."""
    def __init__(self, weight=0.1):
        super(LInftyNeuronalLoss, self).__init__()
        self.weight = weight

    def summarize(self):
        return f"LInftyNeuronalLoss(weight={self.weight})"
    
    def name(self):
        return f"LInftyNeuronalLoss"

    def forward(self, sample: DenoiserData):
        return self.weight * torch.norm(sample.denoised_neuronal - sample.original_neuronal, p=float('inf'), dim=1).mean()


# Regularization terms
class L1NeuronalRegularization(LossTerm):
    """L1 regularization on the neuronal space."""
    def __init__(self, weight=0.1):
        super(L1NeuronalRegularization, self).__init__()
        self.weight = weight

    def summarize(self):
        return f"L1NeuronalRegularization(weight={self.weight})"
    
    def name(self):
        return f"L1NeuronalRegularization"

    def forward(self, sample: DenoiserData):
        return self.weight * torch.abs(sample.denoised_neuronal).mean()

class L2NeuronalRegularization(LossTerm):
    """L2 regularization on the neuronal space."""
    def __init__(self, weight=0.1):
        super(L2NeuronalRegularization, self).__init__()
        self.weight = weight

    def summarize(self):
        return f"L2NeuronalRegularization(weight={self.weight})"

    def name(self):
        return f"L2NeuronalRegularization"
    
    def forward(self, sample: DenoiserData):
        return self.weight * torch.square(sample.denoised_neuronal).mean()
    

class NeuronalSaliencyRegularization(LossTerm):
    """Loss term that encourages the denoised neuronal activity to be more salient for the behaviors of interest."""
    def __init__(self, weight=0.1, saliency_maps=None):
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
        return f"NeuronalSaliencyRegularization(weight={self.weight})"
    
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
class CompositeLoss(nn.Module):
    """Composite loss that combines multiple loss terms."""
    def __init__(self, *loss_terms, record_losses=False):
        super(CompositeLoss, self).__init__()
        self.loss_terms = nn.ModuleList(loss_terms)
        
        self.record_losses = record_losses
        self.loss_recorder = LossRecorder(loss_terms) if record_losses else None

    def forward(self, sample: DenoiserData):
        total_loss = 0.0
        for loss_term in self.loss_terms:
            loss = loss_term(sample)
            total_loss += loss
            if self.record_losses:
                self.loss_recorder.cache_loss(loss_term.name(), loss.item())
        
        if self.record_losses:
            self.loss_recorder.cache_loss("TotalLoss", total_loss.item())
        return total_loss

    def summarize(self):
        summary = "CompositeLoss(" + ", ".join([term.summarize() for term in self.loss_terms]) + ")"
        return summary 

    
    def record_training(self):
        if self.record_losses:
            self.loss_recorder.record_training()
        else:
            raise ValueError("Loss recording was not enabled for this CompositeLoss.")

    def record_testing(self):
        if self.record_losses:
            self.loss_recorder.record_testing()
        else:
            raise ValueError("Loss recording was not enabled for this CompositeLoss.")
    
    def handle_epoch_end(self):
        if self.record_losses:
            self.loss_recorder.handle_epoch_end()
        else:
            raise ValueError("Loss recording was not enabled for this CompositeLoss.")

    def get_loss_recordings(self):
        if self.loss_recorder:
            return self.loss_recorder.get_training_losses(), self.loss_recorder.get_testing_losses()
        else:
            raise ValueError("Loss recording was not enabled for this CompositeLoss.")
    

class GlobalStatisticFitLoss(CompositeLoss):
    """Loss term encouraging the denoised activity to match the statistics of the original activity."""

    def __init__(self, weight=0.1, record_losses=False):
        super(GlobalStatisticFitLoss, self).__init__(self,record_losses=record_losses)
        self.weight = weight
        self.loss_fn = nn.MSELoss()

    def summarize(self):
        return f"GlobalStatisticFitLoss(weight={self.weight})"
    
    def name(self):
        return f"GlobalStatisticFitLoss"

    def forward(self, original_moments, denoised_moments):
        
        total_loss = 0.0
        for moment in range(0, min(len(denoised_moments), len(original_moments))):
            total_loss += self.loss_fn(denoised_moments[moment], original_moments[moment])

        if self.record_losses:
            self.loss_recorder.cache_loss(self.name(), self.weight * total_loss.item())

        return self.weight * total_loss

class LossRecorder:

    """Utility class to record and summarize loss values during training."""
    def __init__(self, loss_terms):
        self.train_loss_history = { term.name(): [] for term in loss_terms }
        self.train_loss_history["TotalLoss"] = []

        self.loss_buffer = { term.name(): [] for term in loss_terms} 
        self.loss_buffer["TotalLoss"] = []
    

        self.test_loss_history = { term.name(): [] for term in loss_terms}
        self.test_loss_history["TotalLoss"] = []
        
        self.current_recorded = self.train_loss_history

        self.loss_terms = loss_terms

    def cache_loss(self, loss_name, loss_value):
        if loss_name in self.loss_buffer:
            self.loss_buffer[loss_name].append(loss_value)
        else:
            raise ValueError(f"Loss name '{loss_name}' not recognized in LossRecorder.")

    def record_training(self):
        self.current_recorded = self.train_loss_history
    
    def record_testing(self):
        self.current_recorded = self.test_loss_history

    def handle_epoch_end(self):
        for loss_name in self.loss_buffer:
            if self.loss_buffer[loss_name]:  # Only record if there are values
                avg_loss = np.mean(self.loss_buffer[loss_name])
                self.current_recorded[loss_name].append(avg_loss)
                self.loss_buffer[loss_name] = []  # Clear buffer after recording
        

    def record(self, loss_name, loss_value):
        if loss_name in self.current_recorded:
            self.current_recorded[loss_name].append(loss_value)
        else:
            raise ValueError(f"Loss name '{loss_name}' not recognized in LossRecorder.")
        
    def get_training_losses(self):
        return self.train_loss_history.copy()
    
    def get_testing_losses(self):
        return self.test_loss_history.copy()    