"""
@authors:
Akshey Kumar
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tqdm import tqdm
from .losses import BccDccLoss
from .initialisations import pca_initialisation, best_of_5_runs
from .utils import tf_batch_prep


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
        self.tau = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(200, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(latent_dim, activation='linear'),
            layers.Normalization(axis=-1),
            layers.GaussianNoise(0.05)
        ])
        self.T_Y = tf.keras.Sequential([
            layers.Dense(latent_dim, activation='linear'),
            layers.Normalization(axis=-1),
        ])
        self.predictor = tf.keras.Sequential([
            layers.Dense(num_behaviour, activation='linear')
        ])

    def call(self, X):
        # Upper arm of commutativity diagram
        yt1_upper = self.tau(X[:, 1])
        bt1_upper = self.predictor(yt1_upper)

        # Lower arm of commutativity diagram
        yt_lower = self.tau(X[:, 0])
        yt1_lower = yt_lower + self.T_Y(yt_lower)

        return yt1_upper, yt1_lower, bt1_upper


class BunDLeTrainer:
    """
    Trainer for the BunDLe Net model.
    
    This class handles the training process for the BunDLe Net model.

    Args:
        model: Instance of the BunDLeNet class.
        optimizer: Optimizer for model training.
        b_type (str): type of behaviour variable 'discrete' or 'continuous'
        gamma: Hyper-parameter of BunDLe-Net loss function
    """

    def __init__(self, model, optimizer, b_type, gamma):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.bccdcc_loss = BccDccLoss(b_type, gamma)

    @tf.function
    def train_step(self, x_train, b_train_1):
        with tf.GradientTape() as tape:
            # forward pass
            yt1_upper, yt1_lower, bt1_upper = self.model(x_train, training=True)
            # loss calculation
            dcc_loss, behaviour_loss, total_loss = self.bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_train_1)

        # gradient calculation
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        # weights update
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return dcc_loss, behaviour_loss, total_loss

    @tf.function
    def test_step(self, x_test, b_test_1):
        # forward pass
        yt1_upper, yt1_lower, bt1_upper = self.model(x_test, training=False)
        # loss calculation
        dcc_loss, behaviour_loss, total_loss = self.bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_test_1)

        return dcc_loss, behaviour_loss, total_loss

    def train_loop(self, train_dataset):
        """
        Handles the training within a single epoch and logs losses
        """
        loss_array = np.zeros((0, 3))
        for batch, (x_train, b_train_1) in enumerate(train_dataset):
            dcc_loss, behaviour_loss, total_loss = self.train_step(x_train, b_train_1)
            loss_array = np.append(loss_array, [[dcc_loss, behaviour_loss, total_loss]], axis=0)

        avg_train_loss = loss_array.mean(axis=0)

        return avg_train_loss

    def test_loop(self, test_dataset):
        """
        Handles testing within a single epoch and logs losses
        """
        loss_array = np.zeros((0, 3))
        for batch, (x_test, b_test_1) in enumerate(test_dataset):
            dcc_loss, behaviour_loss, total_loss = self.test_step(x_test, b_test_1)
            loss_array = np.append(loss_array, [[dcc_loss, behaviour_loss, total_loss]], axis=0)

        avg_test_loss = loss_array.mean(axis=0)

        return avg_test_loss


def train_model(x_train, b_train_1, model, b_type, gamma, learning_rate, n_epochs, initialisation=None,
                validation_data=None):
    """
    Training BunDLe Net

    Args:
        x_train (np.array): training neuronal time-series data
        b_train_1 (np.array): training behavioural time-series data
        b_type (str): type of behaviour variable 'discrete' or 'continuous'
        model: Instance of the BunDLeNet class.
        gamma (float): Weight for the DCC loss component.
        learning_rate (float): Learning rate for the Adam optimiser.
        n_epochs (int): Number of training epochs.
        initialisation (str): 'pca_init' or 'best_of_5_init'
        validation_data: (x_test, b_test_1)
    Returns:
        numpy.ndarray: Array of loss values during training.
    """
    train_dataset = tf_batch_prep(x_train, b_train_1)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

    if validation_data is not None:
        X_test, B_test_1 = validation_data
        test_dataset = tf_batch_prep(X_test, B_test_1)

    if initialisation == 'pca_init':
        pca_initialisation(x_train, model.tau, model.latent_dim)
        model.tau.load_weights('temp/tau_pca_weights.h5')
    elif initialisation == 'best_of_5_init':
        model = best_of_5_runs(x_train, b_train_1, model, b_type, gamma, learning_rate, validation_data)
    elif initialisation is None:
        pass
    else:
        raise ValueError(f"Unknown initialization method: {initialisation}")

    trainer = BunDLeTrainer(model, optimizer, b_type, gamma)
    epochs = tqdm(np.arange(n_epochs))
    train_history = []
    test_history = [] if validation_data is not None else None

    for epoch in epochs:
        train_loss = trainer.train_loop(train_dataset)
        train_history.append(train_loss)

        if validation_data is not None:
            test_loss = trainer.test_loop(test_dataset)
            test_history.append(test_loss)

        epochs.set_description("Loss [Markov, Behaviour, Total]: " + str(np.round(train_loss, 4)))

    train_history = np.array(train_history)
    test_history = np.array(test_history) if test_history is not None else None

    return train_history, test_history
