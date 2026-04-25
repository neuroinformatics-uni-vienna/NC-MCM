"""
@authors:
Akshey Kumar
"""

import numpy as np
import ray
import torch
import torch.nn as nn
from tqdm import tqdm
from .utils_subsystem import torch_batch_prep, GaussianNoise
from .initialisations_subsystem import best_of_5_runs, best_of_n_runs
from ..losses import BccDccLoss

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

    def train_step(self, x_train, b_train_1):
        self.model.train()
        self.optimizer.zero_grad()

        # forward pass
        yt1_upper, yt1_lower, bt1_upper = self.model(x_train)
        # loss calculation
        dcc_loss, behaviour_loss, total_loss, *_ = self.bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_train_1)

        total_loss.backward()
        self.optimizer.step()

        return dcc_loss.item(), behaviour_loss.item(), total_loss.item()

    def test_step(self, x_test, b_test_1):
        self.model.eval()

        with torch.no_grad():
            # forward pass
            yt1_upper, yt1_lower, bt1_upper = self.model(x_test)

        # loss calculation
        dcc_loss, behaviour_loss, total_loss, *_ = self.bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_test_1)

        return dcc_loss.item(), behaviour_loss.item(), total_loss.item()

    def train_loop(self, train_loader):
        """
        Handles the training within a single epoch and logs losses
        """
        loss_array = np.zeros((0, 3))
        for x_train, b_train_1 in train_loader:
            dcc_loss, behaviour_loss, total_loss = self.train_step(x_train, b_train_1)
            loss_array = np.append(loss_array, [[dcc_loss, behaviour_loss, total_loss]], axis=0)

        avg_train_loss = loss_array.mean(axis=0)

        return avg_train_loss

    def test_loop(self, test_loader):
        """
        Handles testing within a single epoch and logs losses
        """
        loss_array = np.zeros((0, 3))
        for x_test, b_test_1 in test_loader:
            dcc_loss, behaviour_loss, total_loss = self.test_step(x_test, b_test_1)
            loss_array = np.append(loss_array, [[dcc_loss, behaviour_loss, total_loss]], axis=0)

        avg_test_loss = loss_array.mean(axis=0)

        return avg_test_loss


def train_model(x_train, b_train_1, model, b_type, gamma, learning_rate, n_epochs, initialisation=None,
                validation_data=None, device=None, report_ray_tune=False, pca_file_save=False):
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
        initialisation (str): 'pca_init' or 'best_of_5_init' or tuple (n, n_epochs) for
                                'best_of_n_init'
        validation_data: (x_test, b_test_1)
        device (torch.device): Device where the model should be trained.
        report_ray_tune (bool): reports validation loss per epoch to ray tune for
                                hyperparameter optimisiaton
        pca_file_save (bool): Whether save weights file or not 
            in case 'initialisation' is 'pca_init'.
    Returns:
        numpy.ndarray: Array of loss values during training.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch_batch_prep(x_train, b_train_1, device=device)

    model = model.to(device)

    if validation_data is not None:
        x_test, b_test_1 = validation_data
        test_loader = torch_batch_prep(x_test, b_test_1, device=device, shuffle=False)

    if initialisation == 'pca_init':
        ret = pca_initialisation(x_train, model.tau, model.latent_dim, device, pca_file_save)
        if pca_file_save: # ret is file path of the weights
            model.tau.load_state_dict(torch.load(ret))
        else: # ret is encoder
            model.tau.load_state_dict(ret.state_dict())
    elif initialisation == 'best_of_5_init':
        model = best_of_5_runs(x_train, b_train_1, model, b_type, gamma, learning_rate, validation_data, device)
    elif isinstance(initialisation, tuple):
        model = best_of_n_runs(initialisation[0], initialisation[1], x_train, b_train_1, model, b_type, gamma,
                               learning_rate, validation_data, device)
    elif initialisation is None:
        pass
    else:
        raise ValueError(f"Unknown initialization method: {initialisation}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = BunDLeTrainer(model, optimizer, b_type, gamma)
    epochs = tqdm(np.arange(n_epochs))
    train_history = []
    test_history = [] if validation_data is not None else None

    for epoch in epochs:
        train_loss = trainer.train_loop(train_loader)
        train_history.append(train_loss)

        if validation_data is not None:
            test_loss = trainer.test_loop(test_loader)
            test_history.append(test_loss)
            if report_ray_tune is True:
                ray.train.report({"epoch": epoch, "val_loss": test_loss[-1]})

        epochs.set_description("Loss [Markov, Behaviour, Total]: " + str(np.round(train_loss, 4)))

    train_history = np.array(train_history)
    test_history = np.array(test_history) if test_history is not None else None

    return train_history, test_history


def project_into_latent_space(x_, model):
    """
    Inference using BunDLe Net

    Args:
        x_ (np.array): Neuronal time-series data for model inference.
        model: Instance of the BunDLeNet class.
    Returns:
        numpy.ndarray: Model predictions.
    """
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        y0_ = model.tau(torch.tensor(x_[:, 0], dtype=torch.float, device=device)).cpu().numpy()

    return y0_
