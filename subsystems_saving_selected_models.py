import os
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import train_model
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data
# from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

# Load Data (excluding behavioural neurons) and plot
for worm_num in [0]:
    print('worm_num ', worm_num)
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



    # model category 1: Unregularised model - train loss criterion
    model_loss = np.load(f"temp/subsystems_model_selection/model_loss_worm_{worm_num}.npy")
    idx = np.argmin(model_loss)
    print(idx, "loss:", model_loss[idx])

    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
    model.load_weights(
        f"temp/subsystems_model_selection/worm_{worm_num}_model_{idx}"
    )
    Y0s_ = model.tau_s(Xs_[:, 0])
    Y0i_ = model.tau_i(Xi_[:, 0])
    Y0m_ = model.tau_m(Xm_[:, 0])
    Y0_ = model.post_tau([Y0s_, Y0i_, Y0m_]).numpy()

    os.makedirs("temp/selected_models", exist_ok=True)
    model.save_weights(f"temp/selected_models/model_unreg_min_train_loss_worm_{worm_num}")
    np.save(f"temp/selected_models/Y0_unreg_min_train_loss_worm_{worm_num}", Y0_)


    # model category 2: regularised model - train loss criterion
    model_train_loss = np.load(f"temp/subsystems_model_selection_reg_test/model_train_loss_worm_{worm_num}.npy")
    idx = np.argmin(model_train_loss)
    print(idx, "loss:", model_train_loss[idx])

    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
    model.load_weights(
        f"temp/subsystems_model_selection_reg_test/worm_{worm_num}_model_{idx}"
    )
    # Projecting into latent space
    Y0s_ = model.tau_s(Xs_[:, 0])
    Y0i_ = model.tau_i(Xi_[:, 0])
    Y0m_ = model.tau_m(Xm_[:, 0])
    Y0_ = model.post_tau([Y0s_, Y0i_, Y0m_]).numpy()

    os.makedirs("temp/selected_models", exist_ok=True)
    model.save_weights(f"temp/selected_models/model_reg_min_train_loss_worm_{worm_num}")
    np.save(f"temp/selected_models/Y0_reg_min_train_loss_worm_{worm_num}", Y0_)


    # model category 3: regularised model - test loss criterion
    model_test_loss = np.load(f"temp/subsystems_model_selection_reg_test/model_test_loss_worm_{worm_num}.npy")
    idx = np.argmin(model_test_loss)
    print(idx, "loss:", model_test_loss[idx])

    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
    model.load_weights(
        f"temp/subsystems_model_selection_reg_test/worm_{worm_num}_model_{idx}"
    )
    # Projecting into latent space
    Y0s_ = model.tau_s(Xs_[:, 0])
    Y0i_ = model.tau_i(Xi_[:, 0])
    Y0m_ = model.tau_m(Xm_[:, 0])
    Y0_ = model.post_tau([Y0s_, Y0i_, Y0m_]).numpy()

    os.makedirs("temp/selected_models", exist_ok=True)
    model.save_weights(f"temp/selected_models/model_reg_min_test_loss_worm_{worm_num}")
    np.save(f"temp/selected_models/Y0_reg_min_test_loss_worm_{worm_num}", Y0_)


    # model category 4: unregularised model - test loss criterion
    model_test_loss = np.load(f"temp/subsystems_model_selection_unreg_test/model_test_loss_worm_{worm_num}.npy")
    idx = np.argmin(model_test_loss)
    print(idx, "loss:", model_test_loss[idx])

    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
    model.load_weights(
        f"temp/subsystems_model_selection_unreg_test/worm_{worm_num}_model_{idx}"
    )
    # Projecting into latent space
    Y0s_ = model.tau_s(Xs_[:, 0])
    Y0i_ = model.tau_i(Xi_[:, 0])
    Y0m_ = model.tau_m(Xm_[:, 0])
    Y0_ = model.post_tau([Y0s_, Y0i_, Y0m_]).numpy()

    os.makedirs("temp/selected_models", exist_ok=True)
    model.save_weights(f"temp/selected_models/model_unreg_min_test_loss_worm_{worm_num}")
    np.save(f"temp/selected_models/Y0_unreg_min_test_loss_worm_{worm_num}", Y0_)



worm_num = 0
data_path = 'datasets/raw/c_elegans/NoStim_Data.mat'
data = Database(data_path=data_path, dataset_no=worm_num)
X = data.neuron_traces.T
B = data.behaviour
X_, B_ = prep_data(X, B, win=15)
for model_type in [
    'unreg_min_train_loss',
    'unreg_min_test_loss',
    'reg_min_train_loss',
    'reg_min_test_loss'
]:
    print(model_type)
    Y0_ = np.load(f"temp/selected_models/Y0_{model_type}_worm_{worm_num}.npy")

    # Plotting latent space dynamics
    vis = LatentSpaceVisualiser(Y0_, B_, data.behaviour_names)
    # vis.plot_latent_timeseries()
    fig, ax = vis.plot_phase_space(show_fig=False, arrow_length_ratio=0.1)
    ax.set_axis_on()
    ax.set_xlabel('sensory neurons axis ')
    ax.set_ylabel('inter neuron axis')
    ax.set_zlabel('motor neuron axis')
    ax.set_title(model_type)
    plt.show()
