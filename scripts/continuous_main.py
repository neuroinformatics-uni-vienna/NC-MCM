import torch
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from ncmcm.bundlenet.utils import prep_data
from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser

rat_names = ['achilles', 'gatsby','cicero', 'buddy']
rat_name = rat_names[0]
data = np.load(f'datasets/raw/rat_hippocampus/{rat_name}.npz')
x, b = data['x'], data['b']
x_, b_ = prep_data(x, b, win=20)

### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3, num_behaviour=b_.shape[1], input_shape=x_.shape)

loss_array, _ = train_model(
    x_,
    b_,
    model,
    b_type='continuous',
    gamma=0.9,
    learning_rate=0.001,
    n_epochs=100,
    initialisation=None,
)

plt.figure()
for i, label in enumerate([
    r"$\mathcal{L}_{\mathrm{Markov}}$",
    r"$\mathcal{L}_{\mathrm{Behavior}}$",
    r"Total loss $\mathcal{L}$"
]):
    plt.semilogy(loss_array[:, i], label=label)
plt.legend()
plt.show()

y0_ = project_into_latent_space(x_[:,0], model)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
true_y_line = ax.scatter(y0_[:, 0], y0_[:, 1], y0_[:, 2], c=b_[:,0])
plt.colorbar(true_y_line)
plt.show()
