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


###########################################################
# BunDLe Net --- Architecture and functions for training  #
###########################################################


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


    def __init__(self, latent_dim: int, num_behaviour: int, reg_coef: float = 0):
        super(BunDLeNet, self).__init__()
        self.latent_dim = latent_dim
        self.num_behaviour = num_behaviour
        self.reg_coef = reg_coef
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
            layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(self.reg_coef)),
            layers.Dense(20, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(7, activation='relu'),
            layers.Dense(3, activation='relu'),
            layers.Dense(1, activation='linear'),
        ])

    # def _build_tau_network(self):
    #     return tf.keras.Sequential([
    #         layers.Flatten(),
    #         layers.Dense(50, activation='relu'),
    #         layers.Dense(20, activation='relu'),
    #         layers.Dense(10, activation='relu'),
    #         layers.Dense(7, activation='relu'),
    #         layers.Dense(3, activation='relu'),
    #         layers.Dense(1, activation='linear'),
    #     ])

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
            'reg_coef': self.reg_coef,
        }
        base_config = super(BunDLeNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(
            latent_dim=config['latent_dim'],
            num_behaviour=config['num_behaviour'],
            reg_coef=config['reg_coef'],
        )


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
            reg_loss = tf.reduce_sum(self.model.losses)
            total_loss = total_loss + reg_loss

        # gradient calculation
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        # weights update
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return dcc_loss, behaviour_loss, reg_loss, total_loss

    @tf.function
    def test_step(self, x_test, b_test_1):
        # forward pass
        yt1_upper, yt1_lower, bt1_upper = self.model(x_test, training=False)
        # loss calculation
        dcc_loss, behaviour_loss, total_loss = self.bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_test_1)
        reg_loss = tf.reduce_sum(self.model.losses)
        total_loss = total_loss + reg_loss

        return dcc_loss, behaviour_loss, reg_loss, total_loss

    def train_loop(self, train_dataset):
        """
        Handles the training within a single epoch and logs losses
        """
        loss_array = np.zeros((0, 4))
        for batch, (x_train, b_train_1) in enumerate(train_dataset):
            dcc_loss, behaviour_loss, reg_loss, total_loss = self.train_step(x_train, b_train_1)
            loss_array = np.append(loss_array, [[dcc_loss, behaviour_loss, reg_loss, total_loss]], axis=0)

        avg_train_loss = loss_array.mean(axis=0)

        return avg_train_loss

    def test_loop(self, test_dataset):
        """
        Handles testing within a single epoch and logs losses
        """
        loss_array = np.zeros((0, 4))
        for batch, (x_test, b_test_1) in enumerate(test_dataset):
            dcc_loss, behaviour_loss, reg_loss, total_loss = self.test_step(x_test, b_test_1)
            loss_array = np.append(loss_array, [[dcc_loss, behaviour_loss, reg_loss, total_loss]], axis=0)

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
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if validation_data is not None:
        x_test, b_test_1 = validation_data
        test_dataset = tf_batch_prep(x_test, b_test_1)

    if initialisation == 'pca_init':
        pca_initialisation(x_train, model.tau, model.latent_dim)
        model.tau.load_weights('temp/tau_pca.weights.h5')
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

        epochs.set_description("Loss [Markov, Behaviour, Reg, Total]: " + str(np.round(train_loss, 4)))

    train_history = np.array(train_history)
    test_history = np.array(test_history) if test_history is not None else None

    return train_history, test_history

