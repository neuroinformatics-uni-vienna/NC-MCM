import torch
import functools
import numpy as np
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.bundlenet.bundlenet import GaussianNoise, BunDLeNet, BunDLeTrainer, train_model, project_into_latent_space
from ncmcm.bundlenet.initialisations import pca_initialisation, best_of_n_runs


assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)


def test_gaussian_noise_train():
    mean = 0.0
    stddev = 0.1
    X = torch.randn(50, 10)

    noise = GaussianNoise(mean=mean, stddev=stddev)

    noise.train()
    output = noise(X)

    torch.testing.assert_close((output - X).mean(), torch.tensor(mean), atol=0.01, rtol=0)
    torch.testing.assert_close((output - X).std(), torch.tensor(stddev), atol=0.01, rtol=0)


def test_gaussian_noise_eval():
    X = torch.randn(50, 10)

    noise = GaussianNoise(mean=0, stddev=0.1)

    noise.eval()
    output = noise(X)

    assert_equal(output, X)


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

    Yt1_upper, Yt1_lower, Bt1_upper = model(X)

    assert_equal(Yt1_upper.shape, (len(X), latent_dim))
    assert_equal(Yt1_lower.shape, (len(X), latent_dim))
    assert_equal(Bt1_upper.shape, (len(X), num_behaviour))


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
