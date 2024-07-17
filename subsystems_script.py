import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import train_model
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import BunDLeNet
from ncmcm.bundlenet.utils import prep_data
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
# time, X = preprocess_data(X, data.fps)
X_, B_ = prep_data(X, B, win=15)
Xs_ = X_[:, :, :, mask == 1]
Xi_ = X_[:, :, :, mask == 2]
Xm_ = X_[:, :, :, mask == 3]
print(Xs_.shape, Xi_.shape, Xm_.shape)

# Deploy BunDLe Net
model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
loss_array, _ = train_model(
    (Xs_, Xi_, Xm_),
    B_,
    model,
    b_type='discrete',
    gamma=0.9,
    learning_rate=0.001,
    n_epochs=1500,
    #initialisation='best_of_5_init'
)

for i, label in enumerate([
    r"$\mathcal{L}_{\mathrm{Markov}}$",
    r"$\mathcal{L}_{\mathrm{Behavior}}$",
    r"Total loss $\mathcal{L}$"
]):
    plt.semilogy(loss_array[:, i], label=label)
plt.legend()
plt.show()

### Projecting into latent space
Y0s_ = model.tau_s(Xs_[:, 0])
Y0i_ = model.tau_i(Xi_[:, 0])
Y0m_ = model.tau_m(Xm_[:, 0])
Y0_ = model.post_tau([Y0s_, Y0i_, Y0m_]).numpy()

model.post_tau.get_weights()

# Plotting latent space dynamics
vis = LatentSpaceVisualiser(Y0_, B_, data.behaviour_names)
vis.plot_latent_timeseries()

fig, ax = vis.plot_phase_space(show_fig=False)
ax.set_axis_on()
ax.set_xlabel('sensory neurons axis ')
ax.set_ylabel('inter neuron axis')
ax.set_zlabel('motor neuron axis')
plt.show()

for angle in np.arange(0,180, 20):
    fig, ax = vis.plot_phase_space(show_fig=False, axis_view=(angle, 15))
    ax.set_axis_on()
    ax.set_xlabel('sensory neurons axis ')
    ax.set_ylabel('inter neuron axis')
    ax.set_zlabel('motor neuron axis')
    plt.show()

for angle in np.arange(0,180, 20):
    fig, ax = vis.plot_phase_space(show_fig=False, axis_view=(0, angle))
    ax.set_axis_on()
    ax.set_xlabel('sensory neurons axis ')
    ax.set_ylabel('inter neuron axis')
    ax.set_zlabel('motor neuron axis')
    plt.show()



# vis.rotating_plot(filename='figures/rotation_subsystems_' + algorithm + '_worm_' + str(worm_num) + '.gif')

# plot_phase_space_2d(Y0_[:,[1,2]], B_, state_names = state_names)
# plot_phase_space_2d(Y0_[:,[2,0]], B_, state_names = state_names)
# plot_phase_space_2d(Y0_[:,[0,1]], B_, state_names = state_names)
# rotating_plot(Y0_, B_,filename='figures/rotation_axis_decomp/rotation'+ algorithm + '_worm_'+str(worm_num) +'.gif', state_names=state_names, legend=False)


# Further experiments
# proportion of variance of a behaviour in latent space along a dimension
var_X_frac, var_Y_frac = [], []
for b_state in np.unique(B):
    var_Y_s = np.var(Y0_[B_ == b_state][:, 0])
    var_Y_i = np.var(Y0_[B_ == b_state][:, 1])
    var_Y_m = np.var(Y0_[B_ == b_state][:, 2])

    var_Y_s_frac = var_Y_s / (var_Y_s + var_Y_i + var_Y_m)
    var_Y_i_frac = var_Y_i / (var_Y_s + var_Y_i + var_Y_m)
    var_Y_m_frac = var_Y_m / (var_Y_s + var_Y_i + var_Y_m)

    var_X_s = np.var(Xs_[B_ == b_state, 0, -1, :])
    var_X_i = np.var(Xi_[B_ == b_state, 0, -1, :])
    var_X_m = np.var(Xm_[B_ == b_state, 0, -1, :])

    var_X_s_frac = var_X_s / (var_X_s + var_X_i + var_X_m)
    var_X_i_frac = var_X_i / (var_X_s + var_X_i + var_X_m)
    var_X_m_frac = var_X_m / (var_X_s + var_X_i + var_X_m)

    var_X_frac.append([var_X_s_frac, var_X_i_frac, var_X_m_frac])
    var_Y_frac.append([var_Y_s_frac, var_Y_i_frac, var_Y_m_frac])

plt.figure()
plt.matshow(var_X_frac)
plt.show()
plt.figure()
plt.matshow(var_Y_frac)
plt.show()

# visualisation - single dimension
import seaborn as sns

for dim, group_name in enumerate(['sensory', 'inter', 'motor']):
    plt.figure()
    sns.histplot([Y0_[B_ == i][:, dim] for i in range(8)])
    ax = plt.gca()
    ax.set_xlabel(group_name)
    plt.show()

# visualisation - pair of dimensions
axis_labels = ['sensory', 'inter', 'motor']
for pair in [[0, 1], [1, 2], [0, 2]]:
    plt.figure()
    [plt.scatter(Y0_[B_ == i][:, pair[0]], Y0_[B_ == i][:, pair[1]], alpha=0.3) for i in range(8)]

    plt.xlabel(axis_labels[pair[0]])
    plt.ylabel(axis_labels[pair[1]])
    plt.show()
