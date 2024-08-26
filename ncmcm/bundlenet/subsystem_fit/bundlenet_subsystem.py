"""
@authors:
Akshey Kumar
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ..losses import BccDccLoss
from ..initialisations import pca_initialisation, best_of_5_runs
from ..utils import torch_batch_prep, GaussianNoise


class BunDLeNet(nn.Module):
    """
    Subsystem Behaviour and Dynamical Learning Network (BunDLeNet) model.

    """

    def __init__(self, latent_dim: int, num_behaviour: int, input_shapes: tuple):
        super(BunDLeNet, self).__init__()
        self.latent_dim = latent_dim
        self.num_behaviour = num_behaviour
        self.tau_s = self._build_tau_network(np.prod(input_shapes[0][-2:]))
        self.tau_i = self._build_tau_network(np.prod(input_shapes[1][-2:]))
        self.tau_m = self._build_tau_network(np.prod(input_shapes[2][-2:]))
        self.post_tau = nn.Sequential(
            GaussianNoise(0.01)
        )
        self.T_Y = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, num_behaviour),
        )

    def _build_tau_network(self, in_features):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 7),
            nn.ReLU(),
            nn.Linear(7, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

    def forward(self, inputs):
        xs, xi, xm = inputs

        # Upper arm of commutativity diagram
        yt1_upper_s = self.tau_s(xs[:, 1])
        yt1_upper_i = self.tau_i(xi[:, 1])
        yt1_upper_m = self.tau_m(xm[:, 1])
        yt1_upper = torch.cat((yt1_upper_s, yt1_upper_i, yt1_upper_m), dim=1)
        yt1_upper = self.post_tau(yt1_upper)
        bt1_upper = self.predictor(yt1_upper)

        # Lower arm of commutativity diagram
        yt1_lower_s = self.tau_s(xs[:, 0])
        yt1_lower_i = self.tau_i(xi[:, 0])
        yt1_lower_m = self.tau_m(xm[:, 0])
        yt_lower = torch.cat((yt1_lower_s, yt1_lower_i, yt1_lower_m), dim=1)
        yt_lower = self.post_tau(yt_lower)
        yt1_lower = yt_lower + self.T_Y(yt_lower)

        return yt1_upper, yt1_lower, bt1_upper
