"""
@authors:
Akshey Kumar
"""
import os
import uuid
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras import Model
from sklearn.decomposition import PCA


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
    # Performing PCA on the time slice
    X0_ = X_[:, 0, :, :]
    X_pca = X_.reshape(X_.shape[0], 2, 1, -1)[:, 0, 0, :]
    pca = PCA(n_components=latent_dim, whiten=True)
    pca.fit(X_pca)
    Y0_ = pca.transform(X_pca)

    # Training tau to reproduce the PCA
    class PCA_encoder(Model):
        def __init__(self, latent_dim):
            super(PCA_encoder, self).__init__()
            self.latent_dim = latent_dim
            self.encoder = tau

        def call(self, x):
            encoded = self.encoder(x)
            return encoded

    pcaencoder = PCA_encoder(latent_dim=latent_dim)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    pcaencoder.compile(optimizer=opt, loss="mse", metrics=["mse"])
    history = pcaencoder.fit(
        X0_,
        Y0_,
        epochs=10,
        batch_size=100,
        verbose=0,
    )

    # Saving weights of this model
    unique_id = str(uuid.uuid4())
    os.makedirs(f"temp/{unique_id}/", exist_ok=True)
    pcaencoder.encoder.save_weights(f"temp/{unique_id}/tau_pca.weights.h5")

    return f"temp/{unique_id}/tau_pca.weights.h5"


def best_of_5_runs(x_train, b_train_1, model, b_type, gamma, learning_rate, validation_data):
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
        validation_data = (x_train, b_train_1)

    best_loss = float('inf')
    best_weights = None

    for i in range(5):
        from ncmcm.bundlenet.bundlenet import train_model
        model_ = keras.models.clone_model(model)

        train_history, test_history = train_model(
            x_train,
            b_train_1,
            model_,
            b_type=b_type,
            gamma=gamma,
            learning_rate=learning_rate,
            n_epochs=200,
            validation_data=validation_data,
            initialisation=None,
            report_ray_tune=False,
        )

        # Store the best weights in memory
        current_loss = test_history[-1, -1]
        print("model:", i, "val loss:", current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = model_.get_weights()

    # Set the best weights back to the original model
    _ = model(x_train)  # build model
    model.set_weights(best_weights)
    return model


def best_of_n_runs(n, n_epochs, x_train, b_train_1, model, b_type, gamma, learning_rate, validation_data):
    """
    Initialises BunDLe net with the best of n runs

    Performs n_epochs epochs of training for n random model initialisations
    and picks the model with the lowest loss
    """
    if validation_data is None:
        import warnings

        warnings.warn(
            "No validation data given. Will proceed to use train dataset loss as deciding factor for the best model"
        )
        validation_data = (x_train, b_train_1)

    best_loss = float('inf')
    best_weights = None

    for i in range(n):
        from ncmcm.bundlenet.bundlenet import train_model
        model_ = keras.models.clone_model(model)

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
        )

        # Store the best weights in memory
        current_loss = test_history[-1, -1]
        print("model:", i, "val loss:", current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_weights = model_.get_weights()

    # Set the best weights back to the original model
    _ = model(x_train) # build model
    model.set_weights(best_weights)
    return model
