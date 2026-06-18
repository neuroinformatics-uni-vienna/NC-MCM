import torch
import numpy as np
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.bundlenet.bundlenet import BunDLeNet, project_into_latent_space
from ncmcm.bundlenet.initialisations import pca_initialisation, best_of_n_runs


def test_pca_initialisation():
    from sklearn.decomposition import PCA

    X = np.random.rand(200, 10)
    B = np.random.randint(5, size=(200,))
    X_, B_ = prep_data(X, B, win=3)

    latent_dim = 3
    device = torch.device('cpu')
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=5, input_shape=X_.shape)

    X_pca = X_[:, 0].reshape(X_.shape[0], -1)
    pca = PCA(n_components=latent_dim, whiten=True)
    Y_pca = torch.tensor(pca.fit_transform(X_pca), dtype=torch.float)
    X0_tensor = torch.tensor(X_[:, 0], dtype=torch.float)
    mse = torch.nn.MSELoss()

    model.tau.eval()
    with torch.no_grad():
        loss_before = mse(model.tau(X0_tensor), Y_pca).item()

    tau_init = pca_initialisation(X_, model.tau, latent_dim, device)
    tau_init.eval()
    with torch.no_grad():
        loss_after = mse(tau_init(X0_tensor), Y_pca).item()

    assert loss_after < loss_before


def test_best_of_n_runs():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)
    X_train, X_test, B_train, B_test = timeseries_train_test_split(X_, B_)

    latent_dim = 3
    num_behaviour = np.unique(B_train).shape[0]
    device = torch.device('cpu')
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shape=X_train.shape)

    best_model = best_of_n_runs(
        n=2, n_epochs=5,
        x_train=X_train, b_train_1=B_train,
        model=model, b_type='discrete',
        gamma=0.9, learning_rate=0.001,
        validation_data=(X_test, B_test),
        device=device
    )

    assert isinstance(best_model, BunDLeNet)
    Y = project_into_latent_space(X_train, best_model)
    assert Y.shape == (X_train.shape[0], latent_dim)
