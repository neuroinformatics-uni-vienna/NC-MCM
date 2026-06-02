# neuronal_saliency.py

# Performs guided gradient backpropagation on a trained BunDLe-Net model to identify which features are chosen to be salient 
# during the forward pass.
#
# Results respect to specific behaviours or in general

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ncmcm.bundlenet.bundlenet import BunDLeNet
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from ncmcm.data_loaders.matlab_dataset import Database

class NeuronalSaliencyAnalyzer:
    def __init__(self, model: BunDLeNet, batch_size=1):
        self.model : BunDLeNet = model

        self.model = self.model.to(next(self.model.parameters()).device)
        self.model.eval()

        self.batch_size = batch_size
        self.saliency_maps = None


    def compute_behavioural_saliency(self, neuronal_data, behavioral_data, behavioral_labels):
        """Compute saliency maps for each behavior
            ** NEURONAL DATA ** are expected to be passed after bundle-net data preprocessing. right now we support window = 1**
        """
        assert neuronal_data.shape[1]==2 and neuronal_data.shape[2]==1, "Neuronal data should be provided after bundle-net preprocessing with window = 1. Please preprocess your data accordingly before passing it to this function." 

        self.saliency_maps = {}
        for behavioral_label in tqdm(behavioral_labels, desc="Computing Saliency Maps"):
            relevant_neuronal_data = neuronal_data[behavioral_data == behavioral_label]
            relevant_neuronal_data = relevant_neuronal_data[:, 0]

            self.saliency_maps[behavioral_label] = self._internal_compute_behavioral_saliency(relevant_neuronal_data)

        return self.saliency_maps
    
    def _internal_compute_behavioral_saliency(self, behavioral_neuronal_data):
        """
        Saliency is computed as the l2 norm of the gradients of the model output with respect to the input, for each latent dimension.
        Then, the saliency map is averaged across the number of samples in the dataset which are labeled with the same behavior, to get a single saliency value for each neuron, for each
        behavior.
        """
        n_neurons = behavioral_neuronal_data.shape[-1]
        saliency_map = torch.tensor([0.0] * n_neurons).to(next(self.model.parameters()).device)


        behavioral_neuronal_data = behavioral_neuronal_data.to(next(self.model.parameters()).device)
        loader = DataLoader(behavioral_neuronal_data, batch_size=self.batch_size, shuffle=False)
        
        for batch in loader:
            batch.requires_grad_(True)
            model_output = self.model.tau(batch) 
            batch_grads = []
            
            for i in range(self.model.latent_dim):
                if batch.grad is not None:
                    batch.grad.zero_()
                
                grad_outputs = torch.zeros_like(model_output)
                grad_outputs[:, i] = 1.0
                
                model_output.backward(gradient=grad_outputs, retain_graph=True)
                batch_grads.append(batch.grad.clone())
            
            batch_grads = torch.stack(batch_grads)
            
            magnitude_per_sample = torch.norm(batch_grads, p=2, dim=0) 
            reduced_magnitude = magnitude_per_sample.sum(dim=0).flatten() 
            saliency_map += reduced_magnitude
        
        saliency_map /= len(loader.dataset)
        return saliency_map.cpu()
    
class NeuronalSaliencyPlotter:
    def __init__(self, saliency_analyzer: NeuronalSaliencyAnalyzer, dataset: Database, path_to_save=None):
        self.saliency_analyzer = saliency_analyzer
        self.dataset = dataset
        self.path_to_save = path_to_save

    def plot_saliency_maps(self):
        if self.saliency_analyzer.saliency_maps is None or len(self.saliency_analyzer.saliency_maps) == 0:
            raise ValueError("No saliency maps found. Run compute_behavioural_saliency first.")

        labels = self.dataset.behaviour_names
        n = len(labels)

        ncols = 1
        nrows = n

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 2 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for i, label in enumerate(labels):
            saliency = self.saliency_analyzer.saliency_maps[label]
            if isinstance(saliency, torch.Tensor):
                saliency = saliency.detach().cpu().numpy()

            saliency = np.expand_dims(saliency, axis=0)

            ax = axes_flat[i]
            im = ax.imshow(saliency, aspect='auto', cmap='viridis')
            ax.set_title(f"Behavior: {labels[i]}")
            neuron_names = self.dataset.neuron_names if hasattr(self.dataset, "neuron_names") else None
            if neuron_names is None:
                neuron_names = getattr(self.dataset, "neuronal_names", None)
            if neuron_names is not None and len(neuron_names) == saliency.shape[-1]:
                ax.set_xticks(range(len(neuron_names)))
                ax.set_xticklabels(neuron_names, rotation=90)
                ax.set_xlabel("Neurons")
            else:
                ax.set_xlabel("Neuron")

            ax.set_yticks([])

            

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis('off')

        fig.suptitle("Neuronal Saliency Maps by Behavioral Label", fontsize=14)
        fig.tight_layout()
        if self.path_to_save:
            fig.savefig(self.path_to_save, dpi=300, bbox_inches='tight')
        