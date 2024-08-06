import os
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
'''
model_loss = []
for i in range(10):
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
    )
    os.makedirs(os.path.dirname('temp/subsystems_model_selection'), exist_ok=True)
    model.save_weights(f"temp/subsystems_model_selection/model_{i}")
    model_loss.append(loss_array[-1,-1])

plt.hist(model_loss)
plt.show()

np.save("temp/subsystems_model_selection/model_loss", model_loss)
'''
model_loss = np.load("temp/subsystems_model_selection/model_loss.npy")

# plotting
for i in np.argsort(model_loss):
    print(i, "loss:", model_loss[i])

    # Load model
    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
    model.load_weights(
        f"temp/subsystems_model_selection/model_{i}"
    )

    # Projecting into latent space
    Y0s_ = model.tau_s(Xs_[:, 0])
    Y0i_ = model.tau_i(Xi_[:, 0])
    Y0m_ = model.tau_m(Xm_[:, 0])
    Y0_ = model.post_tau([Y0s_, Y0i_, Y0m_]).numpy()

    model.post_tau.get_weights()

    # Plotting latent space dynamics
    vis = LatentSpaceVisualiser(Y0_, B_, data.behaviour_names)
    # vis.plot_latent_timeseries()

    fig, ax = vis.plot_phase_space(show_fig=False, arrow_length_ratio=0.01)
    ax.set_axis_on()
    ax.set_xlabel('sensory neurons axis ')
    ax.set_ylabel('inter neuron axis')
    ax.set_zlabel('motor neuron axis')
    plt.show()
