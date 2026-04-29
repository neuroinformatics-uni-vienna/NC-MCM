"""
@authors:
Akshey Kumar
Vittorio Boarini
"""

import numpy as np
import ray
import torch
import torch.nn as nn
from tqdm import tqdm
from .losses import BccDccLoss
from .initialisations import pca_initialisation, best_of_5_runs, best_of_n_runs
from .utils import torch_batch_prep, GaussianNoise


class BunDLeNet(nn.Module):
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

        input_shape (tuple):
            Shape of the input data.

    """

    def __init__(self, latent_dim: int, num_behaviour: int, input_shape: tuple):
        super(BunDLeNet, self).__init__()
        in_features = np.prod(input_shape[-2:])
        self.latent_dim = latent_dim
        self.num_behaviour = num_behaviour
        self.tau = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, latent_dim),
            nn.BatchNorm1d(latent_dim),
            GaussianNoise(0.05),
        )
        self.T_Y = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, num_behaviour),
        )

    def forward(self, x):
        # Upper arm of commutativity diagram
        yt1_upper = self.tau(x[:, 1])
        bt1_upper = self.predictor(yt1_upper)

        # Lower arm of commutativity diagram
        yt_lower = self.tau(x[:, 0])
        yt1_lower = yt_lower + self.T_Y(yt_lower)

        return yt1_upper, yt1_lower, bt1_upper

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
        dcc_loss, behaviour_loss, total_loss = self.bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_train_1)

        total_loss.backward()
        self.optimizer.step()

        return dcc_loss.item(), behaviour_loss.item(), total_loss.item()

    def test_step(self, x_test, b_test_1):
        self.model.eval()

        with torch.no_grad():
            # forward pass
            yt1_upper, yt1_lower, bt1_upper = self.model(x_test)

        # loss calculation
        dcc_loss, behaviour_loss, total_loss = self.bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_test_1)

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
        model = best_of_n_runs(initialisation[0], initialisation[1], x_train, b_train_1, model, b_type, gamma, learning_rate, validation_data, device)
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

#
# def project_into_latent_space(x_, model):
#     """
#     Inference using BunDLe Net
#
#     Args:
#         x_ (np.array): Neuronal time-series data for model inference.
#         model: Instance of the BunDLeNet class.
#     Returns:
#         numpy.ndarray: Model predictions.
#     """
#     device = next(model.parameters()).device
#
#     model.eval()
#     with torch.no_grad():
#         y0_ = model.tau(torch.tensor(x_[:, 0], dtype=torch.float, device=device)).cpu().numpy()
#
#     return y0_


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
        y0_ = model.tau(torch.from_numpy(x_[:, 0]).float().to(device)).cpu().numpy()

    return y0_


def predict_from_latent(y_, model):
    """
    Predict labels from latent variables learned by BunDLe-Net

    Args:
        y_ (np.array): Latent time-series data (time points, latent dimensions)
        model: Instance of the BunDLeNet class.
    Returns:
        numpy.ndarray: Model predictions.
    """
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        yp = model.predictor(torch.from_numpy(y_).float().to(device)).cpu().numpy()

    return yp


