import os
import numpy as np
import matplotlib.pyplot as plt
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.bundlenet import train_model
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import BunDLeNet
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
# from ncmcm.visualisers.neuronal_behavioural import plotting_neuronal_behavioural
from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

X_, B_ = prep_data(X, B, win=15)
Xs_ = X_[:, :, :, mask == 1]
Xi_ = X_[:, :, :, mask == 2]
Xm_ = X_[:, :, :, mask == 3]

# loading embedding
model_loss = np.load("temp/subsystems_model_selection/model_loss.npy")
model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names))
model.load_weights(
    f"temp/subsystems_model_selection/model_{np.argmin(model_loss)}"
)
Y0s_ = model.tau_s(Xs_[:, 0])
Y0i_ = model.tau_i(Xi_[:, 0])
Y0m_ = model.tau_m(Xm_[:, 0])
Y0_ = model.post_tau([Y0s_, Y0i_, Y0m_]).numpy()

# decoding accuracy estimation
X_ = X_.reshape(X_.shape[0],-1)
Xs_ = Xs_.reshape(X_.shape[0],-1)
Xi_ = Xi_.reshape(X_.shape[0],-1)
Xm_ = Xm_.reshape(X_.shape[0],-1)

score = []
for population_ in [X_, Xs_, Xi_, Xm_, Y0_, Y0_[:,[0]], Y0_[:,[1]], Y0_[:,[2]]]:
    population_train, population_test, B_train, B_test = timeseries_train_test_split(population_, B_)

    clf = LogisticRegression(max_iter=1000).fit(population_train, B_train)
    # score.append([clf.score(population_test, B_test), clf.score(population_train, B_train)])

    # single behaviour accuracy
    B_pred = clf.predict(population_test)
    b_acc = []
    for b in np.unique(B_):
        b_acc.append( np.mean(B_pred[B_test == b] == b) )
    b_acc.append(clf.score(population_test, B_test))
    score.append(b_acc)
score = np.array(score)
print(score)

# plotting
plt.figure(figsize=(8,8))
plt.imshow(score, cmap='cividis')
plt.yticks(np.arange(score.shape[0]), ['whole brain','sensory', 'interneuron', 'motor', 'Y', 'Y_s', 'Y_i', 'Y_m'])
b_names = [data.behaviour_names[i] for i in data.behaviour_names] + ['total']
plt.xticks(np.arange(score.shape[1]), b_names, rotation=40)
plt.colorbar()
plt.show()
