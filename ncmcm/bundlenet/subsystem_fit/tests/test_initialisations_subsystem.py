import torch
import numpy as np
from ncmcm.bundlenet.subsystem_fit.utils_subsystem import prep_data
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import BunDLeNet
from ncmcm.bundlenet.subsystem_fit.initialisations_subsystem import best_of_n_runs
from ncmcm.bundlenet.utils import timeseries_train_test_split


def test_best_of_n_runs():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)

    Xs_ = X_[:, :, :, [0, 1, 2]]
    Xi_ = X_[:, :, :, [3, 4, 5]]
    Xm_ = X_[:, :, :, [6, 7, 8, 9]]

    Xs_train, Xs_test, B_train, B_test = timeseries_train_test_split(Xs_, B_)
    Xi_train, Xi_test, _, _ = timeseries_train_test_split(Xi_, B_)
    Xm_train, Xm_test, _, _ = timeseries_train_test_split(Xm_, B_)

    latent_dim = 3
    num_behaviour = np.unique(B).shape[0]
    device = torch.device('cpu')
    input_shapes = Xs_train.shape, Xi_train.shape, Xm_train.shape
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shapes=input_shapes)

    best_model = best_of_n_runs(
        n=2, n_epochs=5,
        x_train=(Xs_train, Xi_train, Xm_train),
        b_train_1=B_train,
        model=model,
        b_type='discrete',
        gamma=0.9, learning_rate=0.001,
        validation_data=((Xs_test, Xi_test, Xm_test), B_test),
        device=device
    )

    assert isinstance(best_model, BunDLeNet)
