import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.metrics import accuracy_score

from ncmcm.bundlenet.losses import ScaleInvariantMSE
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
# from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

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

### Preprocess and prepare data for BundLe Net
# time, X = preprocess_data(X, float(data.fps))
X_, B_ = prep_data(X, B, win=15)
Xs_ = X_[:, :, :, mask == 1]
Xi_ = X_[:, :, :, mask == 2]
Xm_ = X_[:, :, :, mask == 3]
print(Xs_.shape, Xi_.shape, Xm_.shape)

X_train, X_test, B_train, B_test = timeseries_train_test_split(X_, B_)
Xs_, Xs_train, Xs_test = X_[:, :, :, mask == 1], X_train[:, :, :, mask == 1], X_test[:, :, :, mask == 1]
Xi_, Xi_train, Xi_test = X_[:, :, :, mask == 2], X_train[:, :, :, mask == 2], X_test[:, :, :, mask == 2]
Xm_, Xm_train, Xm_test = X_[:, :, :, mask == 3], X_train[:, :, :, mask == 3], X_test[:, :, :, mask == 3]

# Deploy BunDLe Net
model = BunDLeNet(
    latent_dim=3,
    num_behaviour=len(data.behaviour_names),
    reg_coef=0.0
)
# model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
train_loss, test_loss = train_model(
    (Xs_train, Xi_train, Xm_train),
    B_train,
    model,
    b_type='discrete',
    gamma=0.5,
    learning_rate=0.001,
    n_epochs=800,  # 800
    validation_data=((Xs_test, Xi_test, Xm_test), B_test,)
)

### Projecting into latent space
Y0s_ = model.tau_s(Xs_[:, 0])
Y0i_ = model.tau_i(Xi_[:, 0])
Y0m_ = model.tau_m(Xm_[:, 0])
Y0_ = model.post_tau([Y0s_, Y0i_, Y0m_]).numpy()

colors = ['blue', 'orange', 'green', 'red']
for i, label in enumerate([
    r"$\mathcal{L}_{\mathrm{Markov}}$",
    r"$\mathcal{L}_{\mathrm{Behavior}}$",
    r"regularisation loss",
    r"Total loss $\mathcal{L}$"
]):
    plt.plot(train_loss[:, i], c=colors[i], label=label)
    plt.plot(test_loss[:, i], c=colors[i], label=label + ' test', linestyle='--')
plt.hlines(xmin=0, xmax=train_loss.shape[0],
           y=0.9*ScaleInvariantMSE()(Y0_[1:], Y0_[:-1]),
           label='baseline dynamics', linestyle='-.', color='cyan')
plt.legend()
plt.show()


# Plotting latent space dynamics
vis = LatentSpaceVisualiser(Y0_, B_, data.behaviour_names)
vis.plot_latent_timeseries()

fig, ax = vis.plot_phase_space(show_fig=False)
ax.set_axis_on()
ax.set_xlabel('sensory neurons axis ')
ax.set_ylabel('inter neuron axis')
ax.set_zlabel('motor neuron axis')
plt.show()


