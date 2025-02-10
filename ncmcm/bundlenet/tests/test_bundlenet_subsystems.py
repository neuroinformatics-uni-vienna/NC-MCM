import numpy as np
import functools
import torch
from ncmcm.bundlenet.subsystem_fit.utils_subsystem import prep_data
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import BunDLeNet
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import train_model

assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)


def test_bundlenet_architecture():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))

    X_, B_ = prep_data(X, B, win=3)
    Xs_ = torch.tensor(X_[:, :, :, [0,1,2]], dtype=torch.float32)
    Xi_ = torch.tensor(X_[:, :, :, [3,4,5]], dtype=torch.float32)
    Xm_ = torch.tensor(X_[:, :, :, [6,7,8,9]], dtype=torch.float32)

    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B).shape[0]
    input_shapes = Xs_.shape, Xi_.shape, Xm_.shape
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shapes=input_shapes)

    yt1_upper, yt1_lower, bt1_upper = model((Xs_, Xi_, Xm_)) # build model by providing input

    assert_equal(yt1_upper.shape, (len(X_), latent_dim))
    assert_equal(yt1_lower.shape, (len(X_), latent_dim))
    assert_equal(bt1_upper.shape, (len(X_), num_behaviour))
    # assert model.T_Y.input_shape[-1] == latent_dim
    # assert model.T_Y.output_shape[-1] == latent_dim
    # assert model.predictor.output_shape[-1] == num_behaviour

def test_bundlenet_training():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))

    X_, B_ = prep_data(X, B, win=3)
    Xs_ = X_[:, :, :, [0,1,2]]
    Xi_ = X_[:, :, :, [3,4,5]]
    Xm_ = X_[:, :, :, [6,7,8,9]]
    print(Xs_.shape, Xi_.shape, Xm_.shape)

    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B).shape[0]
    input_shapes = Xs_.shape, Xi_.shape, Xm_.shape
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shapes=input_shapes)
    n_epochs = 5
    loss_array, _ = train_model(
        (Xs_, Xi_, Xm_),
        B_,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=n_epochs
    )
    assert loss_array.shape == (n_epochs, 3)

def test_bundlenet_training_best_of_5_init():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))

    X_, B_ = prep_data(X, B, win=3)
    Xs_ = X_[:, :, :, [0,1,2]]
    Xi_ = X_[:, :, :, [3,4,5]]
    Xm_ = X_[:, :, :, [6,7,8,9]]
    print(Xs_.shape, Xi_.shape, Xm_.shape)

    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B).shape[0]
    input_shapes = Xs_.shape, Xi_.shape, Xm_.shape
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shapes=input_shapes)
    n_epochs = 5
    loss_array, _ = train_model(
        (Xs_, Xi_, Xm_),
        B_,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=n_epochs,
        initialisation='best_of_5_init'
    )
    assert loss_array.shape == (n_epochs, 3)


def test_bundlenet_training_validation_data():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)
    Xs_ = X_[:, :, :, [0, 1, 2]]
    Xi_ = X_[:, :, :, [3, 4, 5]]
    Xm_ = X_[:, :, :, [6, 7, 8, 9]]
    from ncmcm.bundlenet.utils import timeseries_train_test_split
    Xs_train, Xs_test, B_train, B_test = timeseries_train_test_split(Xs_, B_)
    Xi_train, Xi_test, B_train, B_test = timeseries_train_test_split(Xi_, B_)
    Xm_train, Xm_test, B_train, B_test = timeseries_train_test_split(Xm_, B_)

    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B_train).shape[0]
    input_shapes = Xs_.shape, Xi_.shape, Xm_.shape
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shapes=input_shapes)
    n_epochs = 5
    train_history, test_history = train_model(
        (Xs_train, Xi_train, Xm_train),
        B_train,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=n_epochs,
        validation_data=((Xs_test, Xi_test, Xm_test), B_test),
        )
    assert train_history.shape == test_history.shape
