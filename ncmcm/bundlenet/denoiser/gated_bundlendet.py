from ncmcm.bundlenet.bundlenet import BunDLeNet, BunDLeTrainer
from ncmcm.bundlenet.bundlenet import train_model as original_train_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GatingLayer(nn.Module):
    def __init__(self, input_dim: int, percentile_cutoff: float = 0.2):
        super(GatingLayer, self).__init__()
        self.in_features = input_dim
        self.gate_scale = nn.Parameter(torch.randn(input_dim))
        self.register_buffer('mask', torch.ones(input_dim))
        self.percentile_cutoff = percentile_cutoff

    def forward(self, x):
        gate_values = x * self.gate_scale * self.mask
        return gate_values
    
    def l1_regularization(self):
        return torch.sum(torch.abs(self.gate_scale))
    
    def prune(self):
        with torch.no_grad():
            threshold_value = torch.quantile(torch.abs(self.gate_scale), self.percentile_cutoff)
            self.mask = (torch.abs(self.gate_scale) >= threshold_value).float()
            self.gate_scale.data *= self.mask

        return self.mask.cpu().numpy(), self.mask.sum().item()

class GatedBunDLeNet(BunDLeNet):
    def __init__(self, latent_dim: int, num_behaviour: int, input_shape: tuple, percentile_cutoff: float = 0.2):
        super(GatedBunDLeNet, self).__init__(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shape=input_shape)
        in_features = np.prod(input_shape[-2:])

        # Introduce the gating mechanism
        self.tau.insert(1, GatingLayer(in_features, percentile_cutoff))
        print(self.tau)

    def forward(self, x):
        # Forward pass through the original BunDLeNet
        yt1_upper = self.tau(x[:, 1])
        bt1_upper = self.predictor(yt1_upper)

        yt_lower = self.tau(x[:, 0])
        yt1_lower = yt_lower + self.T_Y(yt_lower)

        return yt1_upper, yt1_lower, bt1_upper

def train_model(x_train, b_train_1, model: GatedBunDLeNet, b_type, gamma, learning_rate, n_epochs, initialisation=None,
                validation_data=None, device=None, report_ray_tune=False, pca_file_save=False):
    
    # Two phases: first phase in which we train everything the model. Then, we prune the features which are 
    # not important, and we train the model again.

    train_history = None
    test_history = None

    train_history, test_history = original_train_model(
        x_train, b_train_1, model, b_type, gamma, learning_rate, n_epochs, initialisation,
        validation_data, device, report_ray_tune, pca_file_save
    )

    print("First phase of training completed. Starting pruning and second phase of training...")

    # We prune the features
    gating_layer: GatingLayer = model.tau[1]
    assert isinstance(gating_layer, GatingLayer), "Expected the second layer to be a GatingLayer"
    

    pruned_mask, num_pruned = gating_layer.prune()
    # And train the model again for twice the epochs with half the learning rate
    print("Pruning completed. Starting second phase of training with pruned features...")

    new_train_history, _ = original_train_model(
        x_train, b_train_1, model, b_type, gamma, learning_rate / 2, n_epochs * 2, initialisation,
        validation_data, device, report_ray_tune, pca_file_save
    )

    train_history = train_history.tolist() + new_train_history.tolist()
    if test_history is not None:
        test_history = test_history.tolist() + new_train_history.tolist()
        test_history = np.array(test_history)

    return np.array(train_history), test_history, pruned_mask, num_pruned