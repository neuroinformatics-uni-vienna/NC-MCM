"""
@authors 
Kerim Atak
"""

import json
import pandas as pd
import os
import numpy as np
from scipy import sparse
import warnings

class BanditTaskNeuroPixelsDataset:
    def __init__(self, data_path, downsample_num_samples=None):
        """
        Initialize dataset with flexible spike representation options.

        Args:
            data_path: Path to the dataset directory
            downsample_num_samples: If provided, downsample the data to this number of samples.
        """
        self.data_path = data_path
        self.downsample_num_samples = downsample_num_samples
        self.x = None  # neuronal time-series data
        self.b = None  # behavioral time-series data
        self.b_labels_dict = None # behavioral lables as dict
        self.fs = None  # sampling frequency
        
        # load data
        self.load_data()
        

    def load_data(self):
        # Load parameters
        with open(os.path.join(self.data_path, "params.py"), "r") as f:
            params_content = f.read()
            sample_rate = float(params_content.split("sample_rate = ")[1].split("\n")[0])
        
        # Load spike data
        cluster_info = self._load_cluster_info(self.data_path)
        spike_times_in_neuronal_time = np.load(os.path.join(self.data_path, "spike_times.npy"))
        max_spike_times_in_neuronal_time = spike_times_in_neuronal_time.max()
        spike_clusters = np.load(os.path.join(self.data_path, "spike_clusters.npy"))
        spike_times_in_behavioral_time = np.load(os.path.join(self.data_path, "spike_times_milliseconds_sync_to_behav.npy")) # in ms
        translation_indices_neuronal_to_behavioral = np.interp(np.arange(max_spike_times_in_neuronal_time + 1),
                                                               spike_times_in_neuronal_time,
                                                               spike_times_in_behavioral_time)
        
        # Load behavioral data
        with open(os.path.join(self.data_path, "metrics.json"), "r") as metrics_file:
            metrics = json.load(metrics_file)

        # Saving sampling frequency
        self.fs = sample_rate

        # Create neuronal data representation based on chosen method
        self.x = self._create_sparse_neuronal_data_matrix(spike_times_in_neuronal_time, spike_clusters, cluster_info)

        # Downsample data if requested
        if self.downsample_num_samples is not None:
            original_samples = self.x.shape[1]
            self.x = self._downsample_spike_data(self.x, self.downsample_num_samples)
            # Update sampling frequency based on new number of samples
            self.fs = self.fs * (self.downsample_num_samples / original_samples)
            # Downsample translation indices to match downsampled neuronal data
            translation_indices_neuronal_to_behavioral = self._downsample_translation_indices(
                translation_indices_neuronal_to_behavioral, self.downsample_num_samples
            )
            neuronal_length = self.downsample_num_samples
        else:
            neuronal_length = max_spike_times_in_neuronal_time + 1

        # Create behavioral data
        self.b, self.b_labels_dict = self._create_behavioral_data_matrix(metrics, neuronal_length, translation_indices_neuronal_to_behavioral)
        
        # Post processing: Trim waiting periods from start and end
        self._trim_waiting_periods(self.x, self.b, self.b_labels_dict)
        self._relabel_behavioral_states()
        
    def _relabel_behavioral_states(self):        
        # relabel states to start from 0 in case some states are missing
        unique_states = np.unique(self.b)
        state_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_states)}

        # Apply the mapping to state_array_neuronal
        relabeled_state_array = np.zeros_like(self.b)
        for old_label, new_label in state_mapping.items():
            relabeled_state_array[self.b == old_label] = new_label

        # Update state_labels to reflect the new labeling
        relabeled_state_labels = {new_label: self.b_labels_dict[old_label] for old_label, new_label in state_mapping.items()}

        self.b = relabeled_state_array
        self.b_labels_dict = relabeled_state_labels

    def _trim_waiting_periods(self, x, b, b_labels_dict):
        waiting_state_id = next((k for k, v in b_labels_dict.items() if v == 'waiting'), None)
        # from the start to the first non-waiting state
        first_non_waiting_idx = np.argmax(b != waiting_state_id)
        # last non-waiting state to the end
        last_non_waiting_idx = len(b) - np.argmax(b[::-1] != waiting_state_id)
        self.x = x[:, first_non_waiting_idx:last_non_waiting_idx]
        self.b = b[first_non_waiting_idx:last_non_waiting_idx]

    def _downsample_translation_indices(self, translation_indices, target_num_samples):
        """
        Downsample translation indices to match downsampled neuronal data.
        Samples at the center of each bin.

        Args:
            translation_indices: array mapping neuronal time to behavioral time
            target_num_samples: target number of samples after downsampling

        Returns:
            Downsampled translation indices array
        """
        original_samples = len(translation_indices)
        bin_size = original_samples / target_num_samples

        # Sample at the center of each bin
        sample_indices = np.array([int((i + 0.5) * bin_size) for i in range(target_num_samples)])

        # Ensure indices are within bounds
        sample_indices = np.clip(sample_indices, 0, original_samples - 1)

        return translation_indices[sample_indices]

    def _downsample_spike_data(self, spike_matrix, target_num_samples):
        """
        Downsample spike data to a specific number of samples while maintaining binary rasterization.
        If any spike occurs in a time bin, mark the bin as 1.

        Args:
            spike_matrix: scipy.sparse.csr_matrix or np.ndarray (n_neurons, n_timepoints)
            target_num_samples: target number of samples after downsampling

        Returns:
            Downsampled spike matrix with same type as input
        """
        n_neurons, n_samples = spike_matrix.shape

        if target_num_samples >= n_samples:
            raise ValueError(f"Target number of samples ({target_num_samples}) must be lower than original number of samples ({n_samples}) for downsampling")

        # Calculate bin size (how many original samples per new sample)
        bin_size = n_samples / target_num_samples

        downsampled = self._downsample_sparse(spike_matrix, bin_size, n_neurons, target_num_samples)

        return downsampled

    def _downsample_sparse(self, spike_matrix, bin_size, n_neurons, n_downsampled):
        """Downsample sparse spike matrix efficiently"""
        # Convert to COO format for easier manipulation
        coo = spike_matrix.tocoo() # Get row (neuron) and col (time) indices

        # Bin the time indices
        new_time_indices = (coo.col / bin_size).astype(int) # Map original time indices to new binned indices

        # Keep only valid bins
        valid_mask = new_time_indices < n_downsampled
        neuron_indices = coo.row[valid_mask]
        new_time_indices = new_time_indices[valid_mask]

        # Create binary matrix (OR operation: if spike in bin, mark as 1)
        # Remove duplicates by converting to set of (neuron, time) tuples
        unique_spikes = np.unique(np.column_stack([neuron_indices, new_time_indices]), axis=0) # Get unique spikes

        data = np.ones(len(unique_spikes), dtype=np.uint8) # Use uint8 to save memory
        downsampled = sparse.csr_matrix(
            (data, (unique_spikes[:, 0], unique_spikes[:, 1])),
            shape=(n_neurons, n_downsampled),
            dtype=np.uint8
        )

        return downsampled

    def _create_sparse_neuronal_data_matrix(self, spike_times, spike_clusters, cluster_info):
        """
        Create sparse CSR spike matrix (n_neurons, max_time) using scipy.sparse.csr_matrix.
        Rows follow cluster_info['cluster_id']; columns are integer time bins.
        Entries are uint8 counts (1 indicates spike). spike_times must be ints.
        """
        n_neurons = len(cluster_info)
        max_time = int(np.max(spike_times)) + 1
                
        # Prepare data for sparse matrix construction
        neuron_indices = []
        time_indices = []
        
        neuron_id_to_index = {neuron_id: idx for idx, neuron_id in enumerate(cluster_info['cluster_id'])}
        
        for spike_time, cluster_id in zip(spike_times, spike_clusters):
            if cluster_id in neuron_id_to_index:
                neuron_indices.append(neuron_id_to_index[cluster_id])
                time_indices.append(int(spike_time))
        
        # Create sparse matrix
        data = np.ones(len(neuron_indices), dtype=np.uint8)  # Use uint8 to save memory
        sparse_matrix = sparse.csr_matrix(
            (data, (neuron_indices, time_indices)), 
            shape=(n_neurons, max_time), 
            dtype=np.uint8
        )
        
        return sparse_matrix

    def _create_behavioral_data_matrix(self, metrics, neuronal_length, translation_indices_neuronal_to_behavioral):
        """
        Create a behavioral state array aligned with neuronal data.
        Uses translation_indices_neuronal_to_behavioral to map each neuronal time index to behavioral time.
        The num of samples in the behavioral data will exactly match the num of samples in the neuronal data.

        Args:
            metrics: behavioral metrics from JSON
            neuronal_length: number of samples in neuronal data (potentially downsampled)
            translation_indices_neuronal_to_behavioral: mapping from neuronal time to behavioral time (potentially downsampled)
        """
        trials = metrics['metrics']['trials']
        states = metrics['metrics']['states']

        # Find the last timestamp in behavioral time (ms)
        max_trial_time = max([t.get('t chosen', 0) for t in trials if 't chosen' in t], default=0)
        max_state_time = max([s[0] for s in states], default=0)
        last_timestamp_ms = int(max(max_trial_time, max_state_time))

        # Build state_labels dictionary from unique state names in JSON
        unique_state_names = []
        for state in states:
            state_name = state[1]
            if state_name not in unique_state_names:
                unique_state_names.append(state_name)

        state_name_to_id = {name: idx for idx, name in enumerate(unique_state_names)}
        state_labels = {idx: name for name, idx in state_name_to_id.items()}

        # Initialize state array in behavioral time (milliseconds)
        state_array_ms = np.zeros(last_timestamp_ms + 1, dtype=np.int8)

        # Apply states from metrics.json to behavioral time array
        for i in range(len(states)):
            state_time = states[i][0]
            state_name = states[i][1]
            state_id = state_name_to_id[state_name]

            # Find end time (next state or end of recording)
            if i < len(states) - 1:
                next_state_time = states[i + 1][0]
            else:
                next_state_time = last_timestamp_ms + 1

            state_array_ms[state_time:next_state_time] = state_id

        # Map neuronal time to behavioral states using translation indices
        # For each neuronal time index, look up the corresponding behavioral time and get the state
        state_array_neuronal = np.zeros(neuronal_length, dtype=np.int8)

        for neuronal_idx in range(neuronal_length):
            behavioral_ms = int(translation_indices_neuronal_to_behavioral[neuronal_idx])
            # Ensure we don't go out of bounds
            behavioral_ms = min(behavioral_ms, last_timestamp_ms)
            state_array_neuronal[neuronal_idx] = state_array_ms[behavioral_ms]


        return state_array_neuronal, state_labels
        
    def _load_cluster_info(self, data_path):
        """Load cluster information and filter for good neurons"""
        cluster_info = pd.read_csv(os.path.join(data_path, "cluster_info.tsv"), sep="\t")
        cluster_info = cluster_info[cluster_info["group"] == "good"]
        return cluster_info