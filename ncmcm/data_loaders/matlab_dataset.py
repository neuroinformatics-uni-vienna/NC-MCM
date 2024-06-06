"""
@authors:
Akshey Kumar
"""

import numpy as np
import mat73
from dataclasses import dataclass
import scipy


@dataclass
class Database:
    """
    Loading neuronal and behavioural data from matlab files 

    Attributes:
        dataset_no (int): The number of the data set.
        behaviour (numpy.ndarray): Array of discrete behaviour of shape (time_steps,).
        behaviour_names (list): List of discrete behaviour names.
        neuron_traces (numpy.ndarray): Array of neuron traces.
        neuron_names (numpy.ndarray): Array of neuron names.
        fps (float): Frames per second.

    Methods:
        exclude_neurons: Excludes specified neurons from the database.
        categorise_neurons: Categorises neurons based on whether it is sensory,
                            inter or motor neuron. 

    """

    def __init__(self, data_path, dataset_no):
        self.dataset_no = dataset_no
        try:
            data_dict = mat73.loadmat(data_path)
        except Exception as e:
            raise ValueError(f"Error loading MATLAB data from {data_path}: {e}")

        data = data_dict['NoStim_Data']

        delta_f_over_fbc = data['deltaFOverF_bc'][self.dataset_no]
        neuron_names = data['NeuronNames'][self.dataset_no]
        fps = data['fps'][self.dataset_no]
        states = data['States'][self.dataset_no]

        self.behaviour = np.sum([n * states[s] for n, s in enumerate(states)], axis=0).astype(
            int)  # making a single array in which each number corresponds to a behaviour
        behaviour_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal',
                           'Slowing', 'Ventral turn']
        self.behaviour_names = {i: name for i, name in enumerate(behaviour_names)}
        self.neuron_traces = np.array(delta_f_over_fbc).T
        self.neuron_names = np.array(neuron_names, dtype=object)
        self.fps = fps

        # To handle bug in dataset 3 where in neuron_names the last entry is a list. 
        # we replace the list with the contents of the list
        self.neuron_names = np.array([x if not isinstance(x, list) else x[0] for x in self.neuron_names])

    def exclude_neurons(self, exclude_neurons):
        """
        Excludes specified neurons from the database.

        Args:
            exclude_neurons (list): List of neuron names to exclude.

        Returns:
            None

        """
        neuron_names = self.neuron_names
        mask = np.zeros_like(self.neuron_names, dtype='bool')
        for exclude_neuron in exclude_neurons:
            mask = np.logical_or(mask, neuron_names == exclude_neuron)
        mask = ~mask
        self.neuron_traces = self.neuron_traces[mask]
        self.neuron_names = self.neuron_names[mask]

    def _only_identified_neurons(self):
        mask = np.logical_not([x.isnumeric() for x in self.neuron_names])
        self.neuron_traces = self.neuron_traces[mask]
        self.neuron_names = self.neuron_names[mask]

    def categorise_neurons(self):
        self._only_identified_neurons()
        neuron_list = mat73.loadmat('data/raw/Order279.mat')['Order279']
        neuron_category = mat73.loadmat('data/raw/ClassIDs_279.mat')['ClassIDs_279']
        category_dict = {neuron: int(category) for neuron, category in zip(neuron_list, neuron_category)}

        mask = np.array([category_dict[neuron] for neuron in self.neuron_names])
        mask_s = mask == 1
        mask_i = mask == 2
        mask_m = mask == 3

        self.neuron_names_s = self.neuron_names[mask_s]
        self.neuron_names_i = self.neuron_names[mask_i]
        self.neuron_names_m = self.neuron_names[mask_m]

        self.neuron_traces_s = self.neuron_traces[mask_s]
        self.neuron_traces_i = self.neuron_traces[mask_i]
        self.neuron_traces_m = self.neuron_traces[mask_m]

        return mask


@dataclass
class DatabaseStimuli(Database):
    """
    Loading neuronal, behavioural and stimuli data from matlab files
    Inherits from Database class (without stimulus)

    The raw data was curated according to the following correspondence:
    data[0] is name
    data[1] is fps
    data[2] is Bleachcorrected_traces
    data[3] is deltaFOverF
    data[4] is deltaFOverF_bc
    data[5] is deriv_traces
    data[6] is zscored_traces
    data[7] is neuron names
    data[8] is 0,1,2 Response
    data[9] is 11, 21 Stimulus
    data[10] is 1,2,3,4 behaviour

    """

    def __init__(self, data_path, dataset_no):
        self.dataset_no = dataset_no
        try:
            data_dict = scipy.io.loadmat(data_path)
        except Exception as e:
            raise ValueError(f"Error loading MATLAB data from {data_path}: {e}")

        data = data_dict['All_neuron_data'][0, self.dataset_no]

        self.dataset_name = data[0][0]
        self.fps = data[1].flatten()[0]
        self.neuron_traces = data[3].T
        self.response = data[8].flatten()
        self.stimulus = data[9].flatten()
        self.behaviour = data[10].flatten()
        self.response_names = {0: 'miss', 1: 'hit', 2: 'NaN (excluded trials)'}
        self.stimulus_names = {11: '11% oxygen', 21: '21% oxygen'}
        self.behaviour_names = {1: 'forward', 2: 'reverse', 3: 'sustained reversal', 4: 'turn'}

        nn = data[7]
        neuron_names = []
        for i in nn.flat:
            if i[0][0].size > 0:
                neuron_names.append(i[0][0][0])
            else:
                neuron_names.append('0')
        self.neuron_names = np.array(neuron_names)


def merge_discrete_variables(B, R, behaviour_names, response_names):
    """
    Create new target variable combining both discrete variables (B and R)

    parameters:

    B : np.array of shape (m,)
    R : np.array of shape (m,)

    returns:

    BR: np.array of shape (m,)
    """
    br = []
    for i in zip(B, R):
        br.append(i)

    br_states = np.unique(br, axis=0)
    br_dict = {tuple(state): i for i, state in enumerate(br_states)}
    br_names = {n: ' '.join((behaviour_names[state[0]], response_names[state[1]])) for n, state in enumerate(br_states)}
    br = [br_dict[x] for x in br]

    return np.array(br), br_names