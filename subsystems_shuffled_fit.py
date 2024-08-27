import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import tensorflow as tf

# Load Data (excluding behavioural neurons) and plot
worm_num = 0
algorithm = 'BunDLeNet'
b_neurons = [
    'AVAR',
    'AVAL',
    'SMDVR',
    'SMDVL',
    'SMDDR',
    'SMDDL',
    'RIBR',
    'RIBL'
]
data_path = 'datasets/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
data.exclude_neurons(b_neurons)
mask = data.categorise_neurons('datasets/raw/c_elegans')
X = data.neuron_traces.T
B = data.behaviour

# plotting_neuronal_behavioural(X, B, b_names=data.behaviour_names)

B = np.roll(B, shift=np.random.randint(500,B.shape[0]-500)) # shuffle B by circular permutation
# B = np.random.permutation(B)

### Preprocess and prepare data for BundLe Net
# time, X = preprocess_data(X, data.fps)
X_, B_ = prep_data(X, B, win=15)

Xs_ = X_[:, :, :, mask == 1]
Xi_ = X_[:, :, :, mask == 2]
Xm_ = X_[:, :, :, mask == 3]

# Xm_ = np.random.permutation(Xi_)

# X_train, X_test, B_train, B_test = timeseries_train_test_split(X_, B_)
# Xs_, Xs_train, Xs_test = X_[:, :, :, mask == 1], X_train[:, :, :, mask == 1], X_test[:, :, :, mask == 1]
# Xi_, Xi_train, Xi_test = X_[:, :, :, mask == 2], X_train[:, :, :, mask == 2], X_test[:, :, :, mask == 2]
# Xm_, Xm_train, Xm_test = X_[:, :, :, mask == 3], X_train[:, :, :, mask == 3], X_test[:, :, :, mask == 3]


def set_first_weights_to_zeros(module):
    for var in module.trainable_variables:
        var.assign(tf.zeros_like(var))
        print(var.name)
        break


# def _build_tau_network():
#     return tf.keras.Sequential([
#         layers.Flatten(),
#         layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(5e-4)),
#         layers.Dense(20, activation='relu'),
#         layers.Dense(10, activation='relu'),
#         layers.Dense(7, activation='relu'),
#         layers.Dense(3, activation='relu'),
#         layers.Dense(1, activation='linear'),
#     ])


for i in range(10):
    # Deploy BunDLe Net
    model = BunDLeNet(
        latent_dim=3,
        num_behaviour=len(data.behaviour_names),
        reg_coef=2e-4
    )
    # model.post_tau = tf.keras.Sequential([
    #     layers.Concatenate(axis=1),
    #     layers.GaussianNoise(0.01)
    # ])
    # model.tau_s, model.tau_i, model.tau_m = _build_tau_network(), _build_tau_network(), _build_tau_network()
    # _ = model((Xs_, Xi_, Xm_))

    # # Set the weights of tau_s, tau_i, and tau_m to zeros
    # for module in [model.tau_s, model.tau_i, model.tau_m]:
    #     set_first_weights_to_zeros(module)

    loss_array, _ = train_model(
        (Xs_, Xi_, Xm_),
        B_,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=500,
    )

    for i, label in enumerate([
        r"$\mathcal{L}_{\mathrm{Markov}}$",
        r"$\mathcal{L}_{\mathrm{Behavior}}$",
        r"$\mathcal{L}_{\mathrm{Reg}}$",
        r"Total loss $\mathcal{L}$"
    ]):
        plt.semilogy(loss_array[:, i], label=label)
    plt.legend()
    plt.show()

    # Projecting into latent space
    Y0s_ = model.tau_s(Xs_[:, 0])
    Y0i_ = model.tau_i(Xi_[:, 0])
    Y0m_ = model.tau_m(Xm_[:, 0])
    Y0_ = model.post_tau([Y0s_, Y0i_, Y0m_]).numpy()

    from sklearn.metrics import accuracy_score
    from scipy.stats import mode

    B_pred = model.predictor(Y0_).numpy().argmax(axis=1)
    print('accuracy learned embedding', accuracy_score(B_, B_pred))
    print('accuracy of mode predictor', accuracy_score(B_, mode(B_)[0] * np.ones_like(B_)))

    # Plotting latent space dynamics
    vis = LatentSpaceVisualiser(Y0_, B_, data.behaviour_names)
    vis.plot_latent_timeseries()

    # fig, ax = vis.plot_phase_space(show_fig=False, arrow_length_ratio=0.01)
    # ax.set_axis_on()
    # ax.set_xlabel('sensory neurons axis ')
    # ax.set_ylabel('inter neuron axis')
    # ax.set_zlabel('motor neuron axis')
#
    # ax.set_xlim3d(-3, 3)
    # ax.set_ylim3d(-3, 3)
    # ax.set_zlim3d(-3, 3)
    plt.show()
