"""
@authors:
Akshey Kumar
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from sklearn.decomposition import PCA
import keras
import os

def pca_initialisation(X_, tau, latent_dim):
    """
    Initialises BunDLe Net's tau such that its output is the PCA of the input traces.
    PCA initialisation may make the embeddings more reproduceable across runs.
    This function is called within the train_model() function and saves the learned tau weights
    in a .h5 file in the same repository.

    Parameters:
        X_ (np.ndarray): Input data.
        tau (object): BunDLe Net tau (tf sequential layer).
        latent_dim (int): Dimension of the latent space.

    """
    ### Performing PCA on the time slice
    X0_ = X_[:, 0, :, :]
    X_pca = X_.reshape(X_.shape[0], 2, 1, -1)[:, 0, 0, :]
    pca = PCA(n_components=latent_dim, whiten=True)
    pca.fit(X_pca)
    Y0_ = pca.transform(X_pca)

    ### Training tau to reproduce the PCA
    class PCA_encoder(Model):
        def __init__(self, latent_dim):
            super(PCA_encoder, self).__init__()
            self.latent_dim = latent_dim
            self.encoder = tau

        def call(self, x):
            encoded = self.encoder(x)
            return encoded

    pcaencoder = PCA_encoder(latent_dim=latent_dim)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    pcaencoder.compile(optimizer=opt, loss="mse", metrics=["mse"])
    history = pcaencoder.fit(
        X0_,
        Y0_,
        epochs=10,
        batch_size=100,
        verbose=0,
    )
    Y0_pred = pcaencoder(X0_).numpy()
    ### Saving weights of this model
    os.makedirs(os.path.dirname('temp/'), exist_ok=True)
    pcaencoder.encoder.save_weights("temp/tau_pca_weights.h5")


def best_of_5_runs(X_train, B_train_1, model, optimizer, gamma, validation_data):
    """
    Initialises BunDLe net with the best of 5 runs

    Performs 200 epochs of training for 5 random model initialisations
    and picks the model with the lowest loss
    """
    if validation_data is None:
        import warnings

        warnings.warn(
            "No validation data given. Will proceed to use train dataset loss as deciding factor for the best model"
        )
        validation_data = (X_train, B_train_1)

    model_loss = []
    for i in range(5):
        model_ = keras.models.clone_model(model)
        model_.build(input_shape=X_train.shape)
        train_history, test_history = train_model(
            X_train,
            B_train_1,
            model_,
            optimizer,
            gamma=gamma,
            n_epochs=100,
            pca_init=False,
            best_of_5_init=False,
            validation_data=validation_data,
        )
        model_.save_weights("data/generated/best_of_5_runs_models/model_" + str(i))
        model_loss.append(test_history[-1, -1])

    for n, i in enumerate(model_loss):
        print("model:", n, "val loss:", i)

    ### Load model with least loss
    model.load_weights(
        "data/generated/best_of_5_runs_models/model_" + str(np.argmin(model_loss))
    )
    return model