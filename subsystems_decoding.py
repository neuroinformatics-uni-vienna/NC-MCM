import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.subsystem_fit.bundlenet_subsystem import BunDLeNet
from ncmcm.bundlenet.utils import prep_data, timeseries_train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
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

X_, B_ = prep_data(X, B, win=15)
Xs_ = X_[:, :, :, mask == 1]
Xi_ = X_[:, :, :, mask == 2]
Xm_ = X_[:, :, :, mask == 3]
Y0_ = np.load(f"temp/selected_models/Y0_reg_min_test_loss_worm_0.npy")


# Preparing neuronal data
X_win_15, B_win_15 = prep_data(X, B, win=15)
Xs_win_15 = X_win_15[:, :, :, mask == 1]
Xi_win_15 = X_win_15[:, :, :, mask == 2]
Xm_win_15 = X_win_15[:, :, :, mask == 3]

X_win_15 = X_win_15.reshape(X_win_15.shape[0],-1)
Xs_win_15 = Xs_win_15.reshape(X_win_15.shape[0],-1)
Xi_win_15 = Xi_win_15.reshape(X_win_15.shape[0],-1)
Xm_win_15 = Xm_win_15.reshape(X_win_15.shape[0],-1)

X_win_1, B_win_1 = prep_data(X, B, win=1)
Xs_win_1 = X_win_1[:, :, :, mask == 1]
Xi_win_1 = X_win_1[:, :, :, mask == 2]
Xm_win_1 = X_win_1[:, :, :, mask == 3]

X_win_1 = X_win_1.reshape(X_win_1.shape[0],-1)
Xs_win_1 = Xs_win_1.reshape(X_win_1.shape[0],-1)
Xi_win_1 = Xi_win_1.reshape(X_win_1.shape[0],-1)
Xm_win_1 = Xm_win_1.reshape(X_win_1.shape[0],-1)


score = []
for population_ in [X_win_1[14:], Xs_win_1[14:], Xi_win_1[14:], Xm_win_1[14:], X_win_15, Xs_win_15, Xi_win_15, Xm_win_15, Y0_]:
    print(population_.shape, B_.shape)
    # Cross validation
    b_acc = []
    kf = KFold(n_splits=7)
    for i, (train_index, test_index) in enumerate(kf.split(population_)):
        population_train, population_test = population_[train_index], population_[test_index]
        B_train, B_test = B_[train_index], B_[test_index]

        clf = LogisticRegression(max_iter=1000).fit(population_train, B_train)
        B_pred = clf.predict(population_test)
        b_acc.append(clf.score(population_test, B_test))
    score.append(b_acc)
score = np.array(score)

# plotting
plt.figure(figsize=(16,8))
sns.boxenplot(score.T, cmap='cividis')
plt.xticks(np.arange(score.shape[0]), ['whole brain','sensory', 'interneuron', 'motor', 'whole brain (win)', 'sensory (win)', 'interneuron (win)', 'motor (win)', 'Y'])
plt.show()


'''
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

'''