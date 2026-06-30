import torch
import functools
import numpy as np
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.bundlenet.bundlenet import BunDLeNet, BunDLeTrainer, train_model, project_into_latent_space


assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)


def test_project_into_latent_space_eval():
    latent_dim = 3
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)

    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=np.unique(B).shape[0], input_shape=X_.shape)

    project_into_latent_space(X_, model)

    assert not model.training


def test_project_into_latent_space_shape():
    latent_dim = 3
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)

    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=np.unique(B).shape[0], input_shape=X_.shape)

    Y_ = project_into_latent_space(X_, model)

    assert_equal(Y_.shape, (X_.shape[0], latent_dim))
    

def test_project_into_latent_space_deterministic():
    latent_dim = 3
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)

    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=np.unique(B).shape[0], input_shape=X_.shape)

    Y1 = project_into_latent_space(X_, model)
    Y2 = project_into_latent_space(X_, model)
    np.testing.assert_array_equal(Y1, Y2)


def test_bundlenet_architecture():
    latent_dim = 3
    num_behaviour = 8
    X = torch.randn(50, 2, 3, 10)

    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shape=X.shape)

    upper_ys, lower_ys, upper_bs = model(X)

    assert len(upper_ys) == 1
    assert len(lower_ys) == 1
    assert len(upper_bs) == 1
    assert_equal(upper_ys[0].shape, (len(X), latent_dim))
    assert_equal(lower_ys[0].shape, (len(X), latent_dim))
    assert_equal(upper_bs[0].shape, (len(X), num_behaviour))


def test_bundlenet_architecture_unrolled():
    latent_dim = 3
    num_behaviour = 8
    n_steps = 3
    X = torch.randn(50, n_steps + 1, 3, 10)

    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shape=X.shape, n_steps=n_steps)

    upper_ys, lower_ys, upper_bs = model(X)

    assert len(upper_ys) == n_steps
    assert len(lower_ys) == n_steps
    assert len(upper_bs) == n_steps
    for j in range(n_steps):
        assert_equal(upper_ys[j].shape, (len(X), latent_dim))
        assert_equal(lower_ys[j].shape, (len(X), latent_dim))
        assert_equal(upper_bs[j].shape, (len(X), num_behaviour))


def test_bundlenet_training_no_validation():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)
    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B).shape[0]
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shape=X_.shape)
    n_epochs = 5
    _, test_history = train_model(
        X_,
        B_,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=n_epochs
    )
    assert test_history is None


def test_bundlenet_training():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)
    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B).shape[0]
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shape=X_.shape)
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
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shape=X_.shape)
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
    # split data
    X_train, X_test, B_train, B_test = timeseries_train_test_split(X_, B_)
    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B_train).shape[0]
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shape=X_train.shape)
    n_epochs = 5
    loss_array, _ = train_model(
        X_train,
        B_train,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=n_epochs,
        initialisation='best_of_5_init',
        validation_data=(X_test, B_test)
    )
    assert loss_array.shape == (n_epochs, 3)
    
    
def test_bundlenet_training_validation_data():
    X = np.random.rand(50, 10)
    B = np.random.randint(5, size=(50,))
    X_, B_ = prep_data(X, B, win=3)
    X_train, X_test, B_train, B_test = timeseries_train_test_split(X_, B_)

    # Deploy BunDLe Net
    latent_dim = 3
    num_behaviour = np.unique(B_train).shape[0]
    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shape=X_train.shape)
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


def test_bundletrainer_gradients():
    X_ = torch.randn(50, 2, 3, 10)
    B_ = torch.empty(50, dtype=torch.long).random_(5)

    latent_dim = 3
    gamma = torch.rand(1)
    num_behaviour = np.unique(B_).shape[0]

    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shape=X_.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = BunDLeTrainer(model, optimizer, 'discrete', gamma)

    trainer.train_step(X_, B_)

    assert all(param.grad is not None for param in trainer.model.parameters())
    assert not any(torch.isnan(param.grad).any() for param in trainer.model.parameters())
    assert not any(torch.isinf(param.grad).any() for param in trainer.model.parameters())


def test_bundletrainer_loss():
    X_ = torch.randn(50, 2, 3, 10)
    B_ = torch.empty(50, dtype=torch.long).random_(5)

    latent_dim = 3
    gamma = torch.rand(1)
    num_behaviour = np.unique(B_).shape[0]

    model = BunDLeNet(latent_dim=latent_dim, num_behaviour=num_behaviour, input_shape=X_.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = BunDLeTrainer(model, optimizer, 'discrete', gamma)

    dcc_loss, behaviour_loss, total_loss = trainer.train_step(X_, B_)

    assert isinstance(dcc_loss, float)
    assert isinstance(behaviour_loss, float)
    assert isinstance(total_loss, float)

    assert dcc_loss >= 0
    assert behaviour_loss >= 0
    assert total_loss >= 0

    torch.testing.assert_close(total_loss, dcc_loss + behaviour_loss)
