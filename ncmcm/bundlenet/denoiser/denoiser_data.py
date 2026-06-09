from dataclasses import dataclass
import torch

@dataclass
class DenoiserDataBase:
    """Base type of the internal data representations used by the denoiser
    """
    def __init__(self):
        pass
@dataclass
class DenoiserData(DenoiserDataBase):
    """Internal data representation for training and test samples used during pointwise
    training and testing.
    """
    # Original representations
    original_neuronal: torch.Tensor = None
    original_latent: torch.Tensor = None
    # Reconstructed representations
    denoised_neuronal: torch.Tensor = None
    reconstructed_latent: torch.Tensor = None
    # Behavioral label (used for saliency loss)
    behavioral_label: torch.Tensor = None

    def __init__(self, original_latent=None, reconstructed_latent=None, original_neuronal=None, denoised_neuronal=None, behavioral_label=None):
        self.original_neuronal = original_neuronal
        self.original_latent = original_latent
        self.denoised_neuronal = denoised_neuronal
        self.reconstructed_latent = reconstructed_latent
        self.behavioral_label = behavioral_label


@dataclass
class DenoiserDistributionMatchingData(DenoiserDataBase):
    """Internal data representation for training and test samples used during statistical fitting / distribution matching.
    """
    # Neuronal data conditioned on an observable variable (e.g. behavioral label)
    conditioned_neuronal: torch.Tensor = None
    # Observable variable instance (e.g. specific behavioral label)
    label = None
    # 'train' or 'test' indicator for the sample, to determine whether to use the training or test data loader for the statistical fit loss computation.
    indicator = None

    def __init__(self, conditioned_neuronal=None, label=None, indicator=None):
        self.conditioned_neuronal = conditioned_neuronal
        self.label = label
        self.indicator = indicator

