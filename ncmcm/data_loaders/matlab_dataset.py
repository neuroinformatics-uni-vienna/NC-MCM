"""
@authors:
Akshey Kumar
"""

import numpy as np
import mat73

class Database:
    """
    Loading neuronal and behavioural data from matlab files 

    Attributes:
        dataset_no (int): The number of the data set.
        behaviour (numpy.ndarray): Array of discrete behaviour of shape (time_steps,).
        behaviour_names (list): List of discretebehaviour names.
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

        data  = data_dict['NoStim_Data']

        deltaFOverF_bc = data['deltaFOverF_bc'][self.dataset_no]
        derivatives = data['derivs'][self.dataset_no]
        NeuronNames = data['NeuronNames'][self.dataset_no]
        fps = data['fps'][self.dataset_no]
        States = data['States'][self.dataset_no]

        self.behaviour = np.sum([n*States[s] for n, s in enumerate(States)], axis = 0).astype(int) # making a single array in which each number corresponds to a behaviour
        behaviour_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
        self.behaviour_names = {i:name for i, name in enumerate(behaviour_names)}
        print(np.unique(self.behaviour))
        print(self.behaviour_names)
        self.neuron_traces = np.array(deltaFOverF_bc).T
        self.neuron_names = np.array(NeuronNames, dtype=object)
        self.fps = fps

        ### To handle bug in dataset 3 where in neuron_names the last entry is a list. we replace the list with the contents of the list
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
            mask = np.logical_or(mask, neuron_names==exclude_neuron)
        mask = ~mask
        self.neuron_traces = self.neuron_traces[mask] 
        #self.derivative_traces = self.derivative_traces[mask] 
        self.neuron_names = self.neuron_names[mask]

    def _only_identified_neurons(self):
        mask = np.logical_not([x.isnumeric() for x in self.neuron_names])
        self.neuron_traces = self.neuron_traces[mask] 
        #self.derivative_traces = self.derivative_traces[mask] 
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
