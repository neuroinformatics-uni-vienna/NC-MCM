import numpy as np
from ncmcm.bundlenet.utils import prep_data
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model


def test_bundlenet_architecture():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))

    X_, B_ = prep_data(X, B, win=3)

    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B).shape[0]
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour)
    model.build(input_shape=X_.shape)

    assert model.tau.layers[-1].input_shape[-1] == latent_dim
    assert model.T_Y.layers[0].input_shape[-1] == latent_dim
    assert model.T_Y.layers[-1].output_shape[-1] == latent_dim
    assert model.predictor.layers[-1].output_shape[-1] == num_behaviour


def test_bundlenet_training():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)
    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B).shape[0]
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour)
    model.build(input_shape=X_.shape)

    n_epochs = 5
    loss_array, _ = train_model(
        X_,
        B_,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=n_epochs
    )
    assert loss_array.shape == (n_epochs, 3)


def test_bundlenet_training_pca_init():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)
    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B).shape[0]
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour)
    model.build(input_shape=X_.shape)

    n_epochs = 5
    loss_array, _ = train_model(
        X_,
        B_,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=n_epochs,
        initialisation='pca_init'
    )
    assert loss_array.shape == (n_epochs, 3)


def test_bundlenet_training_best_of_5_init():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)
    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B).shape[0]
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour)
    model.build(input_shape=X_.shape)
    n_epochs = 5

    loss_array, _ = train_model(
        X_,
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
    from ncmcm.bundlenet.utils import timeseries_train_test_split
    X_train, X_test, B_train, B_test = timeseries_train_test_split(X_, B_)

    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B_train).shape[0]
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour)
    model.build(input_shape=X_train.shape)
    n_epochs = 5
    train_history, test_history = train_model(
        X_train,
        B_train,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=n_epochs,
        validation_data=(X_test, B_test),
        )
    assert train_history.shape == test_history.shape