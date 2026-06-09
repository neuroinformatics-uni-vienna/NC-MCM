import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn

import os


from ncmcm.bundlenet.denoiser.denoiser import Denoiser, prepare_denoiser_data, DenoiserTrainer
from ncmcm.bundlenet.denoiser.denoiserlosses import *
from ncmcm.bundlenet.denoiser.denoiser_analysis import *
from ncmcm.bundlenet.denoiser.denoiser_data import DenoiserData
from ncmcm.bundlenet.neuronal_saliency.neuronal_saliency import NeuronalSaliencyAnalyzer, NeuronalSaliencyPlotter

def train_bundlenet(model, X_, B_, device=None, gamma=0.9):
    loss_array, _ = train_model(
        X_,
        B_,
        model,
        b_type='discrete',
        gamma=gamma,
        learning_rate=0.001,
        n_epochs=1000,
        device=device
    )
    return loss_array

def save_bundlenet(model: BunDLeNet, Y0_, B_, worm_num):
    """Saves the trained BunDLeNet model and the corresponding latent representations and behavioural labels."""
    algorithm = 'BunDLeNet'
    
    os.makedirs('data/generated/saved_Y', exist_ok=True)
    torch.save(model.state_dict(), 'data/generated/BunDLeNet_model_worm_' + str(worm_num) + '.pt')

    np.savetxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num), Y0_)
    np.savetxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num), B_)
    Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
    B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)
    

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    X = data.neuron_traces.T
    B = data.behaviour
    neuron_names = list(getattr(data, 'neuron_names', [f'Neuron {i}' for i in range(X.shape[1])]))

    # Ensure behaviour names are present
    if not hasattr(data, 'behaviour_names') or data.behaviour_names is None or len(getattr(data, 'behaviour_names', [])) == 0:
        # Default behaviour names for the dataset
        data.behaviour_names = ['Forward', 'Backward', 'Turn']

    # Prepare data for BunDLe Net
    label_encoder = LabelEncoder()
    B = label_encoder.fit_transform(B)
    X_, B_ = prep_data(X, B, win=1)
    model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names), input_shape=X_.shape)
        
    if not os.path.exists('data/generated/BunDLeNet_model_worm_' + str(worm_num) + '.pt'):
        print("BunDLeNet model not found. Training and saving model...")
   
   
        loss_array = train_bundlenet(model, X_, B_, device)

        for i, label in enumerate([
            r"$\mathcal{L}_{\mathrm{Markov}}$",
            r"$\mathcal{L}_{\mathrm{Behavior}}$",
            r"Total loss $\mathcal{L}$"
        ]):
            plt.plot(loss_array[:, i], label=label)

        plt.legend()
        plt.show()

        Y0_ = project_into_latent_space(X_, model)

        save_bundlenet(model, Y0_, B_, worm_num)

        
        print("Model trained and saved.")

    # From here we assume that the BunDLeNet model is trained and saved.


    #####################################################################################
    # 1 - Load the trained BunDLeNet model and compute saliency maps for each behaviour #
    #####################################################################################

    model.load_state_dict(torch.load('data/generated/BunDLeNet_model_worm_' + str(worm_num) + '.pt'))
    model = model.to(device)
    model.eval()

    saliency_analyzer = NeuronalSaliencyAnalyzer(model=model, batch_size=1)
    saliency_maps = saliency_analyzer.compute_behavioural_saliency(
        neuronal_data=torch.tensor(X_, dtype=torch.float32),
        behavioral_data=torch.tensor(B_, dtype=torch.int64),
        behavioral_labels=np.unique(B_)
    )
    saliency_plotter = NeuronalSaliencyPlotter(saliency_analyzer=saliency_analyzer, path_to_save='data/generated/saliency_maps_worm_' + str(worm_num) + '.png', dataset=data)
    saliency_plotter.plot_saliency_maps()

    #####################################################################################
    # 2 - Prepare data for training the Denoiser using the trained BunDLeNet            #
    #####################################################################################

    train_loader, test_loader = prepare_denoiser_data(X, B, model, device, batch_size=256)

    #####################################################################################
    # 3 - Initialize the Denoiser and its loss functions                                #
    #####################################################################################

    model = model.to(device)
    denoiser = Denoiser(bundlenet_model=model, window_size=1).to(device)
    denoiser_loss = CompositeLoss(
        MSELatentLoss(weight=0.4),
        LInftyNeuronalLoss(weight=0.6),
        L1NeuronalRegularization(weight=0.2),
        record_losses=True
    )

    statistical_loss = ConditionedNeuronalMomentMatching(
        weight=0.7,
        moments_to_match=4,
        record_losses=True,
        standardized_moments=False
    )

    # 8000, 0.8, 0.15
    print("Initialized Denoiser and loss functions. .")
    )

        optimizer=torch.optim.Adam(denoiser.parameters(), lr=0.01, weight_decay=1e-5),
        loss_fn=denoiser_loss,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=9000,
        statistical_fit=True,
        statistical_loss_fn=statistical_loss,
        statistical_epochs=0.2,
        denoiser_test_loader=test_loader,
        denoiser_num_epochs=1000,
        device=device,
    )


    PATH = f"data/generated/{trainer.summarize()}"
    os.makedirs(f'{PATH}/', exist_ok=True)
    os.makedirs(f'{PATH}/denoiser_comparison', exist_ok=True)
    os.makedirs(f'{PATH}/neuronal_plots', exist_ok=True)

    #####################################################################################
    # 4 - Train the Denoiser and save the trained model and loss curves                 #
    #####################################################################################

    trainer.train()
    torch.save(denoiser.state_dict(), f'{PATH}/denoiser_model_worm_{worm_num}.pt')

    train_losses, test_losses = denoiser_loss.get_loss_recordings()
    
    #####################################################################################
    # 5 - Plot the loss curves for each loss component and the total loss               #
    #####################################################################################

    loss_names = list(train_losses.keys())
    fig, axes = plt.subplots(len(loss_names) + 3, 2, figsize=(24, 4 * (len(loss_names) + 3)), sharex='col')
    axes = np.atleast_2d(axes)
    
    # Linear scale plots
    for idx, loss_name in enumerate(loss_names):
        ax = axes[idx, 0]
        if loss_name in train_losses:
            ax.plot(train_losses[loss_name], label=f"Train {loss_name}", linestyle='-')
        if loss_name in test_losses:
            ax.plot(test_losses[loss_name], label=f"Test {loss_name}", linestyle='--')
        ax.set_title(f'{loss_name} Loss')
        ax.set_ylabel('Loss')
        ax.legend()
    
    ax_train = axes[-3, 0]
    for loss_name, loss_values in train_losses.items():
        ax_train.plot(loss_values, label=f"Train {loss_name}", linestyle='-')
    ax_train.set_title('All Training Losses')
    ax_train.set_ylabel('Loss')
    ax_train.legend()

    ax_test = axes[-2, 0]
    for loss_name, loss_values in test_losses.items():
        ax_test.plot(loss_values, label=f"Test {loss_name}", linestyle='--')
    ax_test.set_title('All Testing Losses')
    ax_test.set_ylabel('Loss')
    ax_test.legend()

    ax_all = axes[-1, 0]
    for loss_name, loss_values in train_losses.items():
        ax_all.plot(loss_values, label=f"Train {loss_name}", linestyle='-')
    for loss_name, loss_values in test_losses.items():
        ax_all.plot(loss_values, label=f"Test {loss_name}", linestyle='--')
    ax_all.set_title('All Losses Combined')
    ax_all.set_xlabel('Epoch')
    ax_all.set_ylabel('Loss')
    ax_all.legend()
    
    # Logarithmic scale plots
    for idx, loss_name in enumerate(loss_names):
        ax = axes[idx, 1]
        if loss_name in train_losses:
            ax.semilogy(train_losses[loss_name], label=f"Train {loss_name}", linestyle='-')
        if loss_name in test_losses:
            ax.semilogy(test_losses[loss_name], label=f"Test {loss_name}", linestyle='--')
        ax.set_title(f'{loss_name} Loss (Log Scale)')
        ax.set_ylabel('Loss (log)')
        ax.legend()
    
    ax_train_log = axes[-3, 1]
    for loss_name, loss_values in train_losses.items():
        ax_train_log.semilogy(loss_values, label=f"Train {loss_name}", linestyle='-')
    ax_train_log.set_title('All Training Losses (Log Scale)')
    ax_train_log.set_ylabel('Loss (log)')
    ax_train_log.legend()

    ax_test_log = axes[-2, 1]
    for loss_name, loss_values in test_losses.items():
        ax_test_log.semilogy(loss_values, label=f"Test {loss_name}", linestyle='--')
    ax_test_log.set_title('All Testing Losses (Log Scale)')
    ax_test_log.set_ylabel('Loss (log)')
    ax_test_log.legend()

    ax_all_log = axes[-1, 1]
    for loss_name, loss_values in train_losses.items():
        ax_all_log.semilogy(loss_values, label=f"Train {loss_name}", linestyle='-')
    for loss_name, loss_values in test_losses.items():
        ax_all_log.semilogy(loss_values, label=f"Test {loss_name}", linestyle='--')
    ax_all_log.set_title('All Losses Combined (Log Scale)')
    ax_all_log.set_xlabel('Epoch')
    ax_all_log.set_ylabel('Loss (log)')
    ax_all_log.legend()

    fig.tight_layout()
    fig.savefig(f'{PATH}/denoiser_loss_components_worm_{worm_num}.png')

    plt.close()


    denoiser.load_state_dict(torch.load(f'{PATH}/denoiser_model_worm_{worm_num}.pt'))
    denoiser = denoiser.to(device)
    
    #####################################################################################
    # 6 - Load the trained denoiser and evaluate it                                     #
    #####################################################################################

    denoiser.eval()
    denoised_states, re_abstracted_states = denoiser.pipeline(X_)
    Y0_ = project_into_latent_space(X_, model)


    print("Plotting behaviorally relevant neurons...")
    os.makedirs(f'{PATH}/denoiser_top_neurons', exist_ok=True)
    plot_top_neurons(denoised_states, B_, data, worm_num, f'{PATH}/denoiser_top_neurons', num_neurons=3)
    plot_top_neurons_hists(denoised_states, B_, data, worm_num, f'{PATH}/denoiser_top_neurons', num_neurons=3)

    os.makedirs(f'{PATH}/original_top_neurons', exist_ok=True)
    plot_top_neurons(X_[:, 0].squeeze(axis=1), B_, data, worm_num, f'{PATH}/original_top_neurons', num_neurons=3)
    plot_top_neurons_hists(X_[:, 0].squeeze(axis=1), B_, data, worm_num, f'{PATH}/original_top_neurons', num_neurons=3)


    print("Rendering original and denoised neuronal activity plots...")
    fig, axes = plotting_neuronal_behavioural(
        X_[:, 0].squeeze(axis=1),
        b=B, b_names=data.behaviour_names, show_fig=False)
    fig.suptitle("Original Neuronal Activity")
    plt.savefig(f'{PATH}/neuronal_plots/original_neuronal_behavioural_worm_{worm_num}.png')
    plt.close()

    fig, axes = plotting_neuronal_behavioural(
        denoised_states,
        b=B, b_names=data.behaviour_names, show_fig=False)
    fig.suptitle("Denoised Neuronal Activity")
    plt.savefig(f'{PATH}/neuronal_plots/denoised_neuronal_behavioural_worm_{worm_num}.png')
    plt.close('all')


    print("Rendering original and denoised neural activity plots for test set only...")
    test_X = torch.cat([batch[0] for batch in test_loader], dim=0).cpu().numpy()
    test_B = torch.cat([batch[2] for batch in test_loader], dim=0).cpu().numpy()

    test_denoised_states, test_re_abstracted_states = None, None

    with torch.no_grad():
        latent_states = denoiser.bundlenet_tau(torch.from_numpy(test_X).float().to(device))
        test_denoised_states, test_re_abstracted_states = denoiser.forward(latent_states)

    fig, axes = plotting_neuronal_behavioural(
        test_X,
        b=test_B, b_names=data.behaviour_names, show_fig=False,
    )
    fig.suptitle("Original Neuronal Activity (Test Data Only)")
    plt.savefig(f'{PATH}/neuronal_plots/original_neuronal_behavioural_test_data_worm_{worm_num}.png')
    plt.close()

    fig, axes = plotting_neuronal_behavioural(
        test_denoised_states.cpu().numpy(),
        b=test_B, b_names=data.behaviour_names, show_fig=False,
    )
    fig.suptitle("Denoised Neuronal Activity (Test Data Only)")
    plt.savefig(f'{PATH}/neuronal_plots/denoised_neuronal_behavioural_test_data_worm_{worm_num}.png')
    plt.close()

    fig, axes = plotting_neuronal_behavioural(
        np.abs(test_X - test_denoised_states.cpu().numpy()) / (np.abs(test_X) + 1e-8),
        b=test_B, b_names=data.behaviour_names, show_fig=False, 
    )

    fig.suptitle("Relative Difference between Original and Denoised Neuronal Activity (Test Data Only)")
    plt.savefig(f'{PATH}/neuronal_plots/relative_difference_neuronal_behavioural_test_data_worm_{worm_num}.png')
    plt.close()

    print("Rendering original and re-abstracted latent space plots")
    # Visualize the original latent space, denoised latent space, and re-abstracted latent space
    visualizer = LatentSpaceVisualiser(Y0_, B_, data.behaviour_names)
    visualizer.rotating_plot(filename=f'{PATH}/denoiser_comparison/original_rotation_{algorithm}_worm_{worm_num}.gif', show_fig=False)

    # Visualize the denoised latent space by projecting the denoised states through the original BunDLeNet encoder
    visualizer = LatentSpaceVisualiser(re_abstracted_states, B_, data.behaviour_names)
    visualizer.rotating_plot(filename=f'{PATH}/denoiser_comparison/reconstructed_rotation_{algorithm}_worm_{worm_num}.gif', show_fig=False)

   
    print("Original and denoised neuronal activity plots saved.")

    ##################################################################################################
    # 7 - Retrain BunDLeNet on the denoised neuronal activity, visualize latent space and saliency   #
    ##################################################################################################

    print("Retraining BunDLeNet on denoised neuronal activity and visualizing new latent space...")
    X_new, B_new = prep_data(denoised_states, B_, win=1)
    bundlenet_model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names), input_shape=X_new.shape).to(device)
    loss_array = train_bundlenet(bundlenet_model, X_new, B_new, device, gamma=0.9)

    print("Computing saliency maps for the retrained BunDLeNet model...")
    saliency_analyzer = NeuronalSaliencyAnalyzer(model=bundlenet_model, batch_size=1)
    saliency_maps = saliency_analyzer.compute_behavioural_saliency(
        neuronal_data=torch.tensor(X_, dtype=torch.float32),
        behavioral_data=torch.tensor(B_, dtype=torch.int64),
        behavioral_labels=np.unique(B_)
    )

    print("Plotting saliency maps for the retrained BunDLeNet model...")
    saliency_plotter = NeuronalSaliencyPlotter(saliency_analyzer=saliency_analyzer, path_to_save=f'{PATH}/refitted_saliency_maps_worm_{worm_num}.png', dataset=data)
    saliency_plotter.plot_saliency_maps()

    print("Visualizing the new latent space of the retrained BunDLeNet model...")
    Y0_ = project_into_latent_space(X_new, bundlenet_model)
    visualizer = LatentSpaceVisualiser(Y0_, B_new, data.behaviour_names)
    visualizer.rotating_plot(filename=f'{PATH}/denoiser_comparison/new_latent_rotation_{algorithm}_worm_{worm_num}.gif', show_fig=False)
    
    print("Refitting BunDLeNet with gamma=0")    
    loss_array = train_bundlenet(bundlenet_model, X_new, B_new, device, gamma=0)
    Y0_ = project_into_latent_space(X_new, bundlenet_model)
    print("Visualizing the new latent space of the retrained BunDLeNet model with gamma=0...")
    visualizer = LatentSpaceVisualiser(Y0_, B_new, data.behaviour_names)
    visualizer.rotating_plot(filename=f'{PATH}/denoiser_comparison/new_gamma=0_latent_rotation_{algorithm}_worm_{worm_num}.gif', show_fig=False)
    
    print("Refitting BunDLeNet with gamma=1")
    loss_array = train_bundlenet(bundlenet_model, X_new, B_new, device, gamma=1)
    Y0_ = project_into_latent_space(X_new, bundlenet_model)
    print("Visualizing the new latent space of the retrained BunDLeNet model with gamma=1...")
    visualizer = LatentSpaceVisualiser(Y0_, B_new, data.behaviour_names)
    visualizer.rotating_plot(filename=f'{PATH}/denoiser_comparison/new_gamma=1_latent_rotation_{algorithm}_worm_{worm_num}.gif', show_fig=False)   

