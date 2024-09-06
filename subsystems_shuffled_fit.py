import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
import tensorflow as tf
import seaborn as sns

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

# B = np.roll(B, shift=np.random.randint(500,B.shape[0]-500)) # shuffle B by circular permutation
# B = np.random.permutation(B)

### Preprocess and prepare data for BundLe Net
X_, B_ = prep_data(X, B, win=15)

Xs_ = X_[:, :, :, mask == 1]
Xi_ = X_[:, :, :, mask == 2]
Xm_ = X_[:, :, :, mask == 3]

Xs_ = np.random.permutation(Xs_)
# Xi_ = np.random.permutation(Xi_)
# Xm_ = np.random.permutation(Xm_)

# Deploy BunDLe Net
model = BunDLeNet(
    latent_dim=3,
    num_behaviour=len(data.behaviour_names),
    reg_coef=0
)
for model_type in ['unreg_min_train_loss']:#['unreg_min_train_loss', 'reg_min_train_loss', 'reg_min_test_loss']:

    model.load_weights(
        f"temp/selected_models/model_{model_type}_worm_{worm_num}"
    )
    loss_array, _ = train_model(
        (Xs_, Xi_, Xm_),
        B_,
        model,
        b_type='discrete',
        gamma=0.9,
        learning_rate=0.001,
        n_epochs=500,
    )

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

    colors = {
        "Sensory": sns.color_palette("Set2")[0],  # sns.color_palette("dark", 3)[0],  # Dark blue-gray
        "Inter": sns.color_palette("Set2")[1],  # sns.color_palette("dark", 3)[1],     # Dark grayish-brown
        "Motor": sns.color_palette("Set2")[2]  # sns.color_palette("dark", 3)[2]     # Dark slate blue
    }
    fig, ax = vis.plot_phase_space(show_fig=False, arrow_length_ratio=0.1)
    ax.set_axis_on()
    ax.set_xlabel('sensory neurons axis', fontsize=14, color=colors["Sensory"])
    ax.set_ylabel('inter neuron axis', fontsize=14, color=colors["Inter"])
    ax.set_zlabel('motor neuron axis', fontsize=14, color=colors["Motor"])
    # ax.set_title(model_type)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)
    ax.legend().remove()
    plt.show()
