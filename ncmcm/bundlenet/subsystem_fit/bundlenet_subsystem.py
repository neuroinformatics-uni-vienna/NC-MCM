"""
@authors:
Akshey Kumar
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tqdm import tqdm
from ..losses import BccDccLoss
from ..initialisations import pca_initialisation, best_of_5_runs
from ..utils import tf_batch_prep


#########################################################################
# BunDLe Net --- Architecture and functions for training - continuous B #
#########################################################################


class BunDLeNet(Model):
    """Behaviour and Dynamical Learning Network (BunDLeNet) model.

    This model represents the BunDLe Net's architecture for deep learning and is
    based on the commutativity diagrams. The resulting model is dynamically
    consistent (DC) and behaviourally consistent (BC) as per the notion described
    in the paper.

    Args:
        latent_dim (int):
            Dimension of the latent space.

        num_behaviour (int):
            For discrete-valued behaviours, this stipulates the number of
            discrete behavioural states
            For continuous-valued behaviours, this stipulates the number of
            behaviour variables
    """


    def __init__(self, latent_dim: int, num_behaviour: int):
        super(BunDLeNet, self).__init__()
        self.latent_dim = latent_dim
        self.num_behaviour = num_behaviour
        self.tau_s = self._build_tau_network()
        self.tau_i = self._build_tau_network()
        self.tau_m = self._build_tau_network()
        self.post_tau = tf.keras.Sequential([
            layers.Concatenate(axis=1),
            layers.GaussianNoise(0.01)
        ])
        self.T_Y = tf.keras.Sequential([
            layers.Dense(latent_dim, activation='linear'),
        ])
        self.predictor = tf.keras.Sequential([
            layers.Dense(num_behaviour, activation='linear')
        ])

    def _build_tau_network(self):
        return tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dense(20, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(7, activation='relu'),
            layers.Dense(3, activation='relu'),
            layers.Dense(1, activation='linear'),
        ])

    def call(self, inputs):
        Xs, Xi, Xm = inputs
        # Upper arm of commutativity diagram
        Yt1_upper_s = self.tau_s(Xs[:, 1])
        Yt1_upper_i = self.tau_i(Xi[:, 1])
        Yt1_upper_m = self.tau_m(Xm[:, 1])
        Yt1_upper = self.post_tau([Yt1_upper_s, Yt1_upper_i, Yt1_upper_m])
        Bt1_upper = self.predictor(Yt1_upper)

        # Lower arm of commutativity diagram
        Yt1_lower_s = self.tau_s(Xs[:, 0])
        Yt1_lower_i = self.tau_i(Xi[:, 0])
        Yt1_lower_m = self.tau_m(Xm[:, 0])
        Yt_lower = self.post_tau([Yt1_lower_s, Yt1_lower_i, Yt1_lower_m])
        Yt1_lower = Yt_lower + self.T_Y(Yt_lower)

        return Yt1_upper, Yt1_lower, Bt1_upper

    def get_config(self):
        config = {
            'latent_dim': self.latent_dim,
            'num_behaviour': self.num_behaviour,
        }
        base_config = super(BunDLeNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(
            latent_dim=config['latent_dim'],
            num_behaviour=config['num_behaviour'],
        )

