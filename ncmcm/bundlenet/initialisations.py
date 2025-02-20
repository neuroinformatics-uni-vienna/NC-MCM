"""
@authors:
Akshey Kumar
Vittorio Boarini
"""
import os
import copy
import uuid
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA


def pca_initialisation(X_, tau, latent_dim, device, flag_file_save=False):
    """
    Initialises BunDLe Net's tau such that its output is the PCA of 
    the input traces.
    PCA initialisation may make the embeddings more reproducible across runs.
    This function is called within the train_model() function and saves 
    the learned tau weights in a .pt file in the same repository, if 
    flag_save_file is True.

    Args:
        X_ (np.ndarray): Input data.
        tau (object): BunDLe Net tau (tf sequential layer).
        latent_dim (int): Dimension of the latent space.
        device (torch.device): Device where the model should be run.
        flag_file_save (bool): Whether to save the weights as a file or not.

    Returns:
        weights_filepath (str): Path to the weights file.
        or
        pcaencoder.encoder (torch.nn.modules.container.Sequential)
    """
    # Performing PCA on the time slice
    X0_ = X_[:, 0, :, :]
    X_pca = X_.reshape(X_.shape[0], 2, 1, -1)[:, 0, 0, :]
    pca = PCA(n_components=latent_dim, whiten=True)
    pca.fit(X_pca)
    Y0_ = pca.transform(X_pca)

    # Training tau to reproduce the PCA
    class PCA_encoder(nn.Module):
        def __init__(self, latent_dim):
            super(PCA_encoder, self).__init__()
            self.latent_dim = latent_dim
            self.encoder = tau

        def forward(self, x):
            encoded = self.encoder(x)
            return encoded

    X0_tensor = torch.tensor(X0_, dtype=torch.float, device=device)
    Y0_tensor = torch.tensor(Y0_, dtype=torch.float, device=device)

    dataset = TensorDataset(X0_tensor, Y0_tensor)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    pcaencoder = PCA_encoder(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(pcaencoder.parameters(), lr=0.01)
    mse = nn.MSELoss()

    pcaencoder.train()
    epochs = 10
    for _ in range(epochs):
        for batch_X, batch_Y in dataloader:
            opt.zero_grad()
            outputs = pcaencoder(batch_X)
            loss = mse(outputs, batch_Y)
            loss.backward()
            opt.step()

    if flag_file_save: 
        # Saving weights of this model
        unique_id = str(uuid.uuid4())
        os.makedirs(f"temp/{unique_id}/", exist_ok=True)
        weights_filepath = f"temp/{unique_id}/tau_pca.weights.pt"
        torch.save(pcaencoder.encoder.state_dict(), weights_filepath)
        return weights_filepath
    else:
        return pcaencoder.encoder


def best_of_5_runs(
    x_train, 
    b_train_1, 
    model, 
    b_type, 
    gamma, 
    learning_rate, 
    validation_data, 
    device
):
    """
    Initialises BunDLe net with the best of 5 runs

    Performs 200 epochs of training for 5 random model initialisations
    and picks the model with the lowest loss
    """
    if validation_data is None:
        import warnings

        msg = "No validation data given. Will proceed to use train dataset"
        msg += " loss as deciding factor for the best model"
        warnings.warn(msg)
        validation_data = (x_train, b_train_1)

    best_loss = float('inf')
    best_weights = None

    for i in range(5):
        from ncmcm.bundlenet.bundlenet import train_model
        model_ = copy.deepcopy(model)

        train_history, test_history = train_model(
            x_train,
            b_train_1,
            model_,
            b_type=b_type,
            gamma=gamma,
            learning_rate=learning_rate,
            n_epochs=100,
            validation_data=validation_data,
            initialisation=None,
            device=device,
            report_ray_tune=False,
        )

        # Store the best weights in memory
        current_loss = test_history[-1, -1]
        print("model:", i, "val loss:", current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = model_.state_dict()

    # Set the best weights back to the original model
    model.load_state_dict(best_weights)
    return model


def best_of_n_runs(
    n, 
    n_epochs, 
    x_train, 
    b_train_1, 
    model, 
    b_type, 
    gamma, 
    learning_rate, 
    validation_data, 
    device
):
    """
    Initialises BunDLe net with the best of n runs

    Performs n_epochs epochs of training for n random model initialisations
    and picks the model with the lowest loss
    """
    if validation_data is None:
        import warnings

        msg = "No validation data given. Will proceed to use train dataset"
        msg += " loss as deciding factor for the best model" 
        warnings.warn(msg)
        validation_data = (x_train, b_train_1)

    best_loss = float('inf')
    best_weights = None

    for i in range(n):
        from ncmcm.bundlenet.bundlenet import train_model
        model_ = copy.deepcopy(model)
        train_history, test_history = train_model(
            x_train,
            b_train_1,
            model_,
            b_type=b_type,
            gamma=gamma,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            validation_data=validation_data,
            initialisation=None,
            device=device,
            report_ray_tune=False,
        )

        # Store the best weights in memory
        current_loss = test_history[-1, -1]
        print("model:", i, "val loss:", current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = model_.state_dict()

    # Set the best weights back to the original model
    model.load_state_dict(best_weights)
    return model


