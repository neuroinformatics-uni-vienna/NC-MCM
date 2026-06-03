from dataclasses import dataclass
import torch

@dataclass
class DenoiserData():
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


