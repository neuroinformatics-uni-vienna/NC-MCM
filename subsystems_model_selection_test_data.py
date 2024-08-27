import os
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import train_model
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import BunDLeNet, train_model
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
# from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

# Load Data (excluding behavioural neurons) and plot
for worm_num in [0]:
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

    X_train, X_test, B_train, B_test = timeseries_train_test_split(X_, B_)
    Xs_, Xs_train, Xs_test = X_[:, :, :, mask == 1], X_train[:, :, :, mask == 1], X_test[:, :, :, mask == 1]
    Xi_, Xi_train, Xi_test = X_[:, :, :, mask == 2], X_train[:, :, :, mask == 2], X_test[:, :, :, mask == 2]
    Xm_, Xm_train, Xm_test = X_[:, :, :, mask == 3], X_train[:, :, :, mask == 3], X_test[:, :, :, mask == 3]

    model_loss = []
    for i in range(10):
        # Deploy BunDLe Net
        model = BunDLeNet(
            latent_dim=3,
            num_behaviour=len(data.behaviour_names),
            reg_coef=2e-4
        )
        # model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
        train_loss, test_loss = train_model(
            (Xs_train, Xi_train, Xm_train),
            B_train,
            model,
            b_type='discrete',
            gamma=0.9999,
            learning_rate=0.001,
            n_epochs=500,
            validation_data=((Xs_test, Xi_test, Xm_test), B_test,)
        )
        for i, label in enumerate([
            r"$\mathcal{L}_{\mathrm{Markov}}$",
            r"$\mathcal{L}_{\mathrm{Behavior}}$",
            r"regularisation loss",
            r"Total loss $\mathcal{L}$"
        ]):
            plt.plot(train_loss[:, i], label=label)
        plt.plot(test_loss[:, -1], label='test loss')
        plt.legend()
        plt.show()

        os.makedirs(os.path.dirname('temp/subsystems_model_selection_reg_test'), exist_ok=True)
        model.save_weights(f"temp/subsystems_model_selection_reg_test/worm_{worm_num}_model_{i}")
        print(worm_num, i)
        model_loss.append(test_loss[-1,-1])

   #plt.hist(model_loss)
   #plt.show()

    np.save(f"temp/subsystems_model_selection_reg_test/model_loss_worm_{worm_num}", model_loss)

model_loss = np.load(f"temp/subsystems_model_selection_reg_test/model_loss_worm_{worm_num}.npy")
print(model_loss.shape)
# plotting all the models to visualise
for i in np.argsort(model_loss):
    print(i, "loss:", model_loss[i])

    # Load model
    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
    model.load_weights(
        f"temp/subsystems_model_selection_reg_test/worm_{worm_num}_model_{i}"
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
