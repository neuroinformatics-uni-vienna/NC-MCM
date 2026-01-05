"""
@authors 
Kerim Atak
"""

import json
import pandas as pd
import os
import numpy as np
from scipy import sparse

class BanditTaskNeuroPixelsDataset:
    # Common state transition combinations
    HOLD_TO_CHOOSING_TRANSITIONS = {
        ("hold", "choosing left"): "hold --> choosing left",
        ("hold", "choosing right"): "hold --> choosing right"
    }
    
    # Default color map for behavioral state visualization
    # Maps state names to color strings (compatible with Plotly)
    DEFAULT_COLOR_MAP = {
        "waiting": "#ffffff",           # White - waiting period
        "intertrial": "#95a5a6",        # Gray - between trials
        "hold": "#f39d12",              # Orange/Gold - preparatory hold state
        "choosing left": "#e74c3c",     # Red - left choice
        "choosing right": "#3498db",    # Blue - right choice
        "reward": "#2ecc71",            # Green - positive outcome
        "no reward": "#000000",         # Black - negative outcome
        "hold --> choosing left": "#e74c3c",   # Red - transition to left
        "hold --> choosing right": "#3498db",  # Blue - transition to right
    }
    
    def __init__(self, data_path, downsample_fs=None, downsample_method='binary', good_neurons_only=True, state_transitions=None):
        """
        Initialize dataset with flexible spike representation options.

        Args:
            data_path: Path to the dataset directory
            downsample_fs: If provided, downsample the data to this sampling frequency (Hz).
            downsample_method: Method for aggregating spikes during downsampling.
                             'binary': Use OR operation (any spike -> 1)
                             'count': Sum the number of spikes in each bin
            good_neurons_only: If True, only include neurons labeled as 'good' in cluster_info.tsv (default: True)
            state_transitions: Dict mapping state transition tuples to combined state names.
                             Example: {("hold", "choosing left"): "hold --> choosing left",
                                      ("hold", "choosing right"): "hold --> choosing right"}
                             When consecutive states match a transition, they are merged into one combined state.
            
        Attributes after loading:
            x: Neuronal time-series data (scipy.sparse.csr_matrix) of shape (num_neurons, num_timepoints). Do .toarray() to convert to dense.
            b: Behavioral time-series data (scipy.sparse.csr_matrix) of shape (num_behaviors, num_timepoints). Do .toarray() to convert to dense.
            b_labels_dict: Behavioral labels as a dictionary, mapping state IDs to state names.
            b_continuous: Continuous behavioral data (np.ndarray) of shape (num_timepoints,) with running average of last 10 decisions (-1 for left, 1 for right)
            trial_indices: Trial indices (np.ndarray) of shape (num_timepoints,) indicating which trial each timepoint belongs to (starting from 0, -1 for timepoints outside trials)
            block_indices: Block indices (np.ndarray) of shape (num_timepoints,) indicating which block each timepoint belongs to (starting from 0, -1 for timepoints before first block)
            block_labels: Block labels (np.ndarray) of shape (num_timepoints,) indicating the block name for each timepoint
            behavioral_time: Behavioral time (np.ndarray) of shape (num_timepoints,) with the behavioral time in milliseconds at the start of each timepoint
            fs: Sampling frequency
        
        """
        self.data_path = data_path
        self.downsample_fs = downsample_fs
        self.downsample_method = downsample_method
        self.good_neurons_only = good_neurons_only
        self.state_transitions = state_transitions if state_transitions is not None else {}
        self.x = None  # neuronal time-series data (scipy.sparse.csr_matrix)
        self.b = None  # behavioral time-series data (scipy.sparse.csr_matrix)
        self.b_labels_dict = None # behavioral lables as dict, mapping state id to state name
        self.b_labels = None # behavioral labels as list
        self.b_continuous = None  # continuous behavioral data (running avg of last 10 decisions)
        self.trial_indices = None  # the trial indices for each timepoint
        self.block_indices = None  # the block indices for each timepoint
        self.block_labels = None  # the block labels/names for each timepoint
        self.behavioral_time = None  # behavioral time array (in ms)
        self.fs = None  # sampling frequency
        
        # load data and assign to attributes
        self.load_data()
        
        # Print information about the loaded dataset
        print(f"Loaded BanditTaskNeuroPixelsDataset from {data_path}")
        print(f"Neuronal data shape: {self.x.shape}, Behavioral data shape: {self.b.shape}, Sampling frequency: {self.fs} Hz")
        print(f"Behavioral labels: {self.b_labels_dict}")
        

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

        # Downsample neuronal data if requested
        if self.downsample_fs is not None:
            original_samples = self.x.shape[1]
            # Calculate target number of samples based on desired sampling frequency
            target_num_samples = int(original_samples * self.downsample_fs / self.fs)
            self.x = self._downsample_spike_data(self.x, target_num_samples)
            # Update sampling frequency to the new target
            self.fs = self.downsample_fs
            # Downsample translation indices to match downsampled neuronal data
            translation_indices_neuronal_to_behavioral = self._downsample_translation_indices(
                translation_indices_neuronal_to_behavioral, target_num_samples
            )
            neuronal_length = target_num_samples
        else:
            neuronal_length = max_spike_times_in_neuronal_time + 1

        # Create behavioral data
        self.b, self.b_labels_dict = self._create_behavioral_data_matrix(metrics, neuronal_length, translation_indices_neuronal_to_behavioral)
        
        # Create continuous behavioral data (running average of last 10 decisions)
        self.b_continuous = self._create_continuous_behavioral_data(metrics, neuronal_length, translation_indices_neuronal_to_behavioral)
        
        # Create trial indices for each timepoint
        self.trial_indices = self._create_trial_indices(metrics, neuronal_length, translation_indices_neuronal_to_behavioral)
        
        # Create block indices for each timepoint
        self.block_indices = self._create_block_indices(metrics, neuronal_length, translation_indices_neuronal_to_behavioral)
        
        # Create block labels for each timepoint
        self.block_labels = self._create_block_labels(metrics, neuronal_length, translation_indices_neuronal_to_behavioral)
        
        # Create behavioral time array (start time of each timepoint in ms)
        self.behavioral_time = self._create_behavioral_time_array(neuronal_length, translation_indices_neuronal_to_behavioral)
        
        # Post processing: Trim waiting periods from start and end
        self._trim_waiting_periods(self.x, self.b, self.b_labels_dict)
        
        # Apply state transitions if specified
        if self.state_transitions:
            self._apply_state_transitions()
        
        self._relabel_behavioral_states()
        
        # Final check to ensure lengths match
        assert self.x.shape[1] == self.b.shape[1], "Final neuronal and behavioral data length mismatch after processing."
        assert len(self.b_continuous) == self.b.shape[1], "Continuous behavioral data length mismatch after processing."
        assert len(self.trial_indices) == self.b.shape[1], "Trial indices length mismatch after processing."
        assert len(self.block_indices) == self.b.shape[1], "Block indices length mismatch after processing."
        assert len(self.block_labels) == self.b.shape[1], "Block labels length mismatch after processing."
        assert len(self.behavioral_time) == self.b.shape[1], "Behavioral time array length mismatch after processing."
        
        # Sort behavioral labels by their state id (dict keys) and store as list
        self.b_labels = [self.b_labels_dict[k] for k in sorted(self.b_labels_dict.keys())]
    
    def _apply_state_transitions(self):
        """
        Apply state transitions to merge consecutive states into combined states.
        This modifies self.b and self.b_labels_dict in place.
        """
        b_dense = self.b.toarray().flatten()
        
        # Create reverse mapping from state name to state id
        state_name_to_id = {name: state_id for state_id, name in self.b_labels_dict.items()}
        
        # Find the next available state id for new combined states
        next_state_id = max(self.b_labels_dict.keys()) + 1
        
        # Create a mapping for transition tuples (state_id1, state_id2) -> combined_state_id
        transition_id_mapping = {}
        combined_state_labels = {}
        
        for (state1_name, state2_name), combined_name in self.state_transitions.items():
            if state1_name in state_name_to_id and state2_name in state_name_to_id:
                state1_id = state_name_to_id[state1_name]
                state2_id = state_name_to_id[state2_name]
                
                # Check if we already have a combined state for this transition
                transition_key = (state1_id, state2_id)
                if transition_key not in transition_id_mapping:
                    combined_state_id = next_state_id
                    transition_id_mapping[transition_key] = combined_state_id
                    combined_state_labels[combined_state_id] = combined_name
                    next_state_id += 1
        
        # Process the state array to identify and merge transitions
        new_state_array = b_dense.copy()
        i = 0
        while i < len(b_dense) - 1:
            current_state = b_dense[i]
            next_state = b_dense[i + 1]
            transition_key = (current_state, next_state)
            
            if transition_key in transition_id_mapping:
                # Found a transition to merge
                combined_state_id = transition_id_mapping[transition_key]
                
                # Find the extent of current_state followed by next_state
                # Start of current_state segment
                start_idx = i
                while start_idx > 0 and b_dense[start_idx - 1] == current_state:
                    start_idx -= 1
                
                # End of next_state segment
                end_idx = i + 1
                while end_idx < len(b_dense) - 1 and b_dense[end_idx + 1] == next_state:
                    end_idx += 1
                
                # Mark this entire segment as the combined state
                new_state_array[start_idx:end_idx + 1] = combined_state_id
                
                # Skip past this merged segment
                i = end_idx + 1
            else:
                i += 1
        
        # Update the behavioral state array and labels
        self.b = sparse.csr_matrix(new_state_array, shape=(1, len(new_state_array)))
        self.b_labels_dict.update(combined_state_labels)
        
    def _relabel_behavioral_states(self):        
        # relabel states to start from 0 in case some states are missing
        b_dense = self.b.toarray().flatten()
        unique_states = np.unique(b_dense)
        state_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_states)}

        # Apply the mapping to state_array_neuronal
        relabeled_state_array = np.zeros_like(b_dense)
        for old_label, new_label in state_mapping.items():
            relabeled_state_array[b_dense == old_label] = new_label

        # Update state_labels to reflect the new labeling
        relabeled_state_labels = {new_label: self.b_labels_dict[old_label] for old_label, new_label in state_mapping.items()}

        self.b = sparse.csr_matrix(relabeled_state_array, shape=(1, len(relabeled_state_array)))
        self.b_labels_dict = relabeled_state_labels

    def _trim_waiting_periods(self, x, b, b_labels_dict):
        waiting_state_id = next((k for k, v in b_labels_dict.items() if v == 'waiting'), None)
        
        # Extract dense array from sparse matrix for comparison
        b_dense = b.toarray().flatten()
        
        # from the start to the first non-waiting state
        first_non_waiting_idx = np.argmax(b_dense != waiting_state_id)
        # last non-waiting state to the end
        last_non_waiting_idx = len(b_dense) - np.argmax(b_dense[::-1] != waiting_state_id)
        
        self.x = x[:, first_non_waiting_idx:last_non_waiting_idx]
        self.b = b[:, first_non_waiting_idx:last_non_waiting_idx]
        self.b_continuous = self.b_continuous[first_non_waiting_idx:last_non_waiting_idx]
        self.trial_indices = self.trial_indices[first_non_waiting_idx:last_non_waiting_idx]
        self.block_indices = self.block_indices[first_non_waiting_idx:last_non_waiting_idx]
        self.block_labels = self.block_labels[first_non_waiting_idx:last_non_waiting_idx]
        self.behavioral_time = self.behavioral_time[first_non_waiting_idx:last_non_waiting_idx]

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
        Downsample spike data to a specific number of samples.
        Binary method: If any spike occurs in a time bin, mark the bin as 1.
        Count method: Sum the number of spikes in each bin.

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
        coo = spike_matrix.tocoo()

        # Bin the time indices
        new_time_indices = (coo.col / bin_size).astype(int)

        # Keep only valid bins
        valid_mask = new_time_indices < n_downsampled
        neuron_indices = coo.row[valid_mask]
        new_time_indices = new_time_indices[valid_mask]

        if self.downsample_method == 'binary':
            # Binary method: OR operation (if spike in bin, mark as 1)
            unique_spikes = np.unique(np.column_stack([neuron_indices, new_time_indices]), axis=0)
            data = np.ones(len(unique_spikes), dtype=np.uint8)
            downsampled = sparse.csr_matrix(
                (data, (unique_spikes[:, 0], unique_spikes[:, 1])),
                shape=(n_neurons, n_downsampled),
                dtype=np.uint8
            )
        elif self.downsample_method == 'count':
            # Count method: sum spikes in each bin
            # Count occurrences of each (neuron, time) pair
            spike_coords = np.column_stack([neuron_indices, new_time_indices])
            unique_coords, counts = np.unique(spike_coords, axis=0, return_counts=True)
            
            downsampled = sparse.csr_matrix(
                (counts, (unique_coords[:, 0], unique_coords[:, 1])),
                shape=(n_neurons, n_downsampled),
                dtype=np.uint16  # Use uint16 to allow counts > 255
            )
        else:
            raise ValueError(f"Invalid downsample_method: {self.downsample_method}. Must be 'binary' or 'count'.")

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

        # Handle "choosing" state specially - split into "choosing left" and "choosing right"
        if "choosing" in unique_state_names:
            unique_state_names.remove("choosing")
            unique_state_names.append("choosing left")
            unique_state_names.append("choosing right")

        state_name_to_id = {name: idx for idx, name in enumerate(unique_state_names)}
        state_labels = {idx: name for name, idx in state_name_to_id.items()}

        # Initialize state array in behavioral time (milliseconds)
        state_array_ms = np.zeros(last_timestamp_ms + 1, dtype=np.int8)

        # Create a mapping from trial start time to choice (l or r)
        trial_choice_map = {}
        for trial in trials:
            start_time = trial.get('start')
            choice = trial.get('choice', '').lower()
            if start_time is not None:
                trial_choice_map[start_time] = choice

        # Apply states from metrics.json to behavioral time array
        for i in range(len(states)):
            state_time = states[i][0]
            state_name = states[i][1]
            
            # Find end time (next state or end of recording)
            if i < len(states) - 1:
                next_state_time = states[i + 1][0]
            else:
                next_state_time = last_timestamp_ms + 1

            # Handle "choosing" state: determine if left or right based on trial choice
            if state_name == "choosing":
                # Find the trial that contains this state time
                trial_choice = None
                for trial in trials:
                    trial_start = trial.get('start')
                    trial_end = trial.get('t chosen')
                    if trial_start is not None and trial_end is not None:
                        if trial_start <= state_time < trial_end:
                            trial_choice = trial.get('choice', '').lower()
                            break
                
                # Determine state name based on choice
                if trial_choice == 'r':
                    state_name = "choosing right"
                elif trial_choice == 'l':
                    state_name = "choosing left"
                else:
                    state_name = "choosing left"  # Default fallback

            state_id = state_name_to_id[state_name]
            state_array_ms[state_time:next_state_time] = state_id

        # Map neuronal time to behavioral states using translation indices
        # For each neuronal time index, look up the corresponding behavioral time and get the state
        state_array_neuronal = np.zeros(neuronal_length, dtype=np.int8)

        for neuronal_idx in range(neuronal_length):
            behavioral_ms = int(translation_indices_neuronal_to_behavioral[neuronal_idx])
            # Ensure we don't go out of bounds
            behavioral_ms = min(behavioral_ms, last_timestamp_ms)
            state_array_neuronal[neuronal_idx] = state_array_ms[behavioral_ms]

        # Convert to sparse matrix (1, neuronal_length) for consistency with x
        state_array_sparse = sparse.csr_matrix(state_array_neuronal, shape=(1, neuronal_length))

        return state_array_sparse, state_labels
        
    def _create_continuous_behavioral_data(self, metrics, neuronal_length, translation_indices_neuronal_to_behavioral):
        """
        Create continuous behavioral data representing the running average of the last 10 trial decisions.
        Values range from -1 (all left) to 1 (all right), with 0 being half left, half right.

        Args:
            metrics: behavioral metrics from JSON
            neuronal_length: number of samples in neuronal data (potentially downsampled)
            translation_indices_neuronal_to_behavioral: mapping from neuronal time to behavioral time (potentially downsampled)

        Returns:
            np.ndarray of shape (neuronal_length,) with continuous values
        """
        trials = metrics['metrics']['trials']
        
        # Find the last timestamp in behavioral time (ms)
        max_trial_time = max([t.get('t chosen', 0) for t in trials if 't chosen' in t], default=0)
        last_timestamp_ms = int(max_trial_time)
        
        # Initialize continuous array in behavioral time (milliseconds)
        continuous_array_ms = np.zeros(last_timestamp_ms + 1, dtype=np.float32)
        
        # Sort trials by start time to process chronologically
        sorted_trials = sorted(trials, key=lambda t: t.get('start', 0))
        
        # Track last 10 choices (use a rolling window)
        recent_choices = []  # Will store 1 for right, -1 for left
        
        for trial in sorted_trials:
            start_time = trial.get('start')
            end_time = trial.get('t chosen')
            choice = trial.get('choice', '').lower()
            
            if start_time is None or end_time is None or choice not in ['l', 'r']:
                continue
            
            # Add current choice to the window
            choice_value = 1.0 if choice == 'r' else -1.0
            recent_choices.append(choice_value)
            
            # Keep only last 10 choices
            if len(recent_choices) > 10:
                recent_choices.pop(0)
            
            # Calculate running average
            running_avg = np.mean(recent_choices)
            
            # Apply this value to the entire trial period
            continuous_array_ms[start_time:end_time + 1] = running_avg
            
        # After processing all trials, forward-fill to handle gaps
        last_value = 0.0
        for i in range(len(continuous_array_ms)):
            if continuous_array_ms[i] != 0.0 or i == 0:
                last_value = continuous_array_ms[i]
            else:
                continuous_array_ms[i] = last_value
    
        # Map neuronal time to continuous behavioral data using translation indices
        continuous_array_neuronal = np.zeros(neuronal_length, dtype=np.float32)
        
        for neuronal_idx in range(neuronal_length):
            behavioral_ms = int(translation_indices_neuronal_to_behavioral[neuronal_idx])
            behavioral_ms = min(behavioral_ms, last_timestamp_ms)
            continuous_array_neuronal[neuronal_idx] = continuous_array_ms[behavioral_ms]
        
        return continuous_array_neuronal

    def _load_cluster_info(self, data_path):
        """Load cluster information and filter for good neurons if specified"""
        cluster_info = pd.read_csv(os.path.join(data_path, "cluster_info.tsv"), sep="\t")
        if self.good_neurons_only:
            cluster_info = cluster_info[cluster_info["group"] == "good"]
        return cluster_info
    
    def _create_trial_indices(self, metrics, neuronal_length, translation_indices_neuronal_to_behavioral):
        """
        Create trial index array aligned with neuronal data.
        Each timepoint is assigned the index of the trial it belongs to (starting from 0).
        Timepoints outside of trials are assigned -1.

        Args:
            metrics: behavioral metrics from JSON
            neuronal_length: number of samples in neuronal data (potentially downsampled)
            translation_indices_neuronal_to_behavioral: mapping from neuronal time to behavioral time (potentially downsampled)

        Returns:
            np.ndarray of shape (neuronal_length,) with trial indices
        """
        trials = metrics['metrics']['trials']
        
        # Find the last timestamp in behavioral time (ms)
        max_trial_time = max([t.get('t chosen', 0) for t in trials if 't chosen' in t], default=0)
        last_timestamp_ms = int(max_trial_time)
        
        # Initialize trial index array in behavioral time (milliseconds)
        # Use -1 to indicate timepoints not in any trial
        trial_indices_ms = np.full(last_timestamp_ms + 1, -1, dtype=np.int32)
        
        # Sort trials by start time to process chronologically
        sorted_trials = sorted(trials, key=lambda t: t.get('start', 0))
        
        # Assign trial indices to each time period
        for trial_idx, trial in enumerate(sorted_trials):
            start_time = trial.get('start')
            end_time = trial.get('t chosen')
            
            if start_time is None or end_time is None:
                continue
            
            # Assign this trial index to all timepoints in the trial
            trial_indices_ms[start_time:end_time + 1] = trial_idx
        
        # Map neuronal time to trial indices using translation indices
        trial_indices_neuronal = np.full(neuronal_length, -1, dtype=np.int32)
        
        for neuronal_idx in range(neuronal_length):
            behavioral_ms = int(translation_indices_neuronal_to_behavioral[neuronal_idx])
            behavioral_ms = min(behavioral_ms, last_timestamp_ms)
            trial_indices_neuronal[neuronal_idx] = trial_indices_ms[behavioral_ms]
        
        return trial_indices_neuronal
    
    def _create_block_indices(self, metrics, neuronal_length, translation_indices_neuronal_to_behavioral):
        """
        Create block index array aligned with neuronal data.
        Each timepoint is assigned the index of the block it belongs to (starting from 0).
        Timepoints before the first block are assigned -1.

        Args:
            metrics: behavioral metrics from JSON
            neuronal_length: number of samples in neuronal data (potentially downsampled)
            translation_indices_neuronal_to_behavioral: mapping from neuronal time to behavioral time (potentially downsampled)

        Returns:
            np.ndarray of shape (neuronal_length,) with block indices
        """
        blocks = metrics['metrics']['blocks']
        
        # Find the last timestamp in behavioral time (ms)
        # Use the last block's start time or last trial time as reference
        trials = metrics['metrics']['trials']
        max_trial_time = max([t.get('t chosen', 0) for t in trials if 't chosen' in t], default=0)
        max_block_time = max([b.get('t', 0) for b in blocks], default=0)
        last_timestamp_ms = int(max(max_trial_time, max_block_time))
        
        # Initialize block index array in behavioral time (milliseconds)
        # Use -1 to indicate timepoints before any block
        block_indices_ms = np.full(last_timestamp_ms + 1, -1, dtype=np.int32)
        
        # Sort blocks by start time to process chronologically
        sorted_blocks = sorted(blocks, key=lambda b: b.get('t', 0))
        
        # Assign block indices to each time period
        for block_idx, block in enumerate(sorted_blocks):
            start_time = block.get('t')
            
            if start_time is None:
                continue
            
            # Find the end time (start of next block or end of recording)
            if block_idx < len(sorted_blocks) - 1:
                end_time = sorted_blocks[block_idx + 1].get('t', last_timestamp_ms)
            else:
                end_time = last_timestamp_ms
            
            # Assign this block index to all timepoints in the block
            block_indices_ms[start_time:end_time + 1] = block_idx
        
        # Map neuronal time to block indices using translation indices
        block_indices_neuronal = np.full(neuronal_length, -1, dtype=np.int32)
        
        for neuronal_idx in range(neuronal_length):
            behavioral_ms = int(translation_indices_neuronal_to_behavioral[neuronal_idx])
            behavioral_ms = min(behavioral_ms, last_timestamp_ms)
            block_indices_neuronal[neuronal_idx] = block_indices_ms[behavioral_ms]
        
        return block_indices_neuronal
    
    def _create_block_labels(self, metrics, neuronal_length, translation_indices_neuronal_to_behavioral):
        """
        Create block label array aligned with neuronal data.
        Each timepoint is assigned the label/name of the block it belongs to.
        Timepoints before the first block are assigned None.

        Args:
            metrics: behavioral metrics from JSON
            neuronal_length: number of samples in neuronal data (potentially downsampled)
            translation_indices_neuronal_to_behavioral: mapping from neuronal time to behavioral time (potentially downsampled)

        Returns:
            np.ndarray of shape (neuronal_length,) with block labels (strings)
        """
        blocks = metrics['metrics']['blocks']
        
        # Find the last timestamp in behavioral time (ms)
        trials = metrics['metrics']['trials']
        max_trial_time = max([t.get('t chosen', 0) for t in trials if 't chosen' in t], default=0)
        max_block_time = max([b.get('t', 0) for b in blocks], default=0)
        last_timestamp_ms = int(max(max_trial_time, max_block_time))
        
        # Initialize block label array in behavioral time (milliseconds)
        # Use None to indicate timepoints before any block
        block_labels_ms = np.full(last_timestamp_ms + 1, None, dtype=object)
        
        # Sort blocks by start time to process chronologically
        sorted_blocks = sorted(blocks, key=lambda b: b.get('t', 0))
        
        # Assign block labels to each time period
        for block_idx, block in enumerate(sorted_blocks):
            start_time = block.get('t')
            block_label = block.get('block')
            
            if start_time is None or block_label is None:
                continue
            
            # Find the end time (start of next block or end of recording)
            if block_idx < len(sorted_blocks) - 1:
                end_time = sorted_blocks[block_idx + 1].get('t', last_timestamp_ms)
            else:
                end_time = last_timestamp_ms
            
            # Assign this block label to all timepoints in the block
            block_labels_ms[start_time:end_time + 1] = block_label
        
        # Map neuronal time to block labels using translation indices
        block_labels_neuronal = np.full(neuronal_length, None, dtype=object)
        
        for neuronal_idx in range(neuronal_length):
            behavioral_ms = int(translation_indices_neuronal_to_behavioral[neuronal_idx])
            behavioral_ms = min(behavioral_ms, last_timestamp_ms)
            block_labels_neuronal[neuronal_idx] = block_labels_ms[behavioral_ms]
        
        return block_labels_neuronal
    
    def _create_behavioral_time_array(self, neuronal_length, translation_indices_neuronal_to_behavioral):
        """
        Create behavioral time array aligned with neuronal data.
        Each timepoint is assigned its corresponding behavioral time in milliseconds (at the start of the timepoint).

        Args:
            neuronal_length: number of samples in neuronal data (potentially downsampled)
            translation_indices_neuronal_to_behavioral: mapping from neuronal time to behavioral time (potentially downsampled)

        Returns:
            np.ndarray of shape (neuronal_length,) with behavioral times in milliseconds
        """
        behavioral_time = np.zeros(neuronal_length, dtype=np.float64)
        
        for neuronal_idx in range(neuronal_length):
            behavioral_time[neuronal_idx] = translation_indices_neuronal_to_behavioral[neuronal_idx]
        
        return behavioral_time
    
    # ------------------ Additional methods can be added here ----------------- #
    
    def get_color_map_for_plotting(self):
        """
        Get a color map suitable for plotting functions that maps state IDs to colors.
        
        This method converts the DEFAULT_COLOR_MAP (which maps state names to colors)
        into a dictionary mapping state IDs to colors using the current dataset's
        b_labels_dict.
        
        Returns:
            dict: Dictionary mapping state IDs (int) to color strings (hex codes).
                  Example: {0: '#ffffff', 1: '#e74c3c', 2: '#3498db'}
        
        Example:
            >>> dataset = BanditTaskNeuroPixelsDataset(data_path)
            >>> color_map = dataset.get_color_map_for_plotting()
            >>> from ncmcm.visualisers import behavioural_discrete
            >>> fig = behavioural_discrete.plot_behavior_state_frequencies_barchart(
            ...     dataset.b, dataset.b_labels_dict, color_map=color_map
            ... )
        """
        color_map = {}
        for state_id, state_name in self.b_labels_dict.items():
            if state_name in self.DEFAULT_COLOR_MAP:
                color_map[state_id] = self.DEFAULT_COLOR_MAP[state_name]
            else:
                # Fallback to a default color if state name not in DEFAULT_COLOR_MAP
                import plotly.express as px
                colors = px.colors.qualitative.Plotly
                color_map[state_id] = colors[state_id % len(colors)]
        
        return color_map
    
    def get_rgb_colors_for_visualizer(self):
        """
        Get RGB colors suitable for LatentSpaceVisualiser.
        
        This method converts the DEFAULT_COLOR_MAP (which maps state names to hex colors)
        into a list of RGB tuples ordered by state IDs. RGB values are normalized to [0, 1].
        
        Returns:
            list: List of RGB tuples with values in [0, 1].
                  Example: [(1.0, 1.0, 1.0), (0.91, 0.61, 0.07), (0.21, 0.60, 0.86)]
        
        Example:
            >>> dataset = BanditTaskNeuroPixelsDataset(data_path)
            >>> colors = dataset.get_rgb_colors_for_visualizer()
            >>> from ncmcm.visualisers.latent_space import LatentSpaceVisualiser
            >>> vis = LatentSpaceVisualiser(y, b, dataset.b_labels_dict, colors=colors)
        """
        import matplotlib.colors as mcolors
        
        rgb_colors = []
        # Sort by state ID to maintain consistent ordering
        for state_id in sorted(self.b_labels_dict.keys()):
            state_name = self.b_labels_dict[state_id]
            
            if state_name in self.DEFAULT_COLOR_MAP:
                hex_color = self.DEFAULT_COLOR_MAP[state_name]
                # Convert hex to RGB in [0, 1] range
                rgb = mcolors.to_rgb(hex_color)
                rgb_colors.append(rgb)
            else:
                # Fallback to plotly colors if state name not in DEFAULT_COLOR_MAP
                import plotly.express as px
                colors = px.colors.qualitative.Plotly
                hex_color = colors[state_id % len(colors)]
                rgb = mcolors.to_rgb(hex_color)
                rgb_colors.append(rgb)
        
        return rgb_colors
    
    def get_recording_length_mins(self):
        """
        Get the length of the recording in minutes.
        
        Returns:
            float: Recording length in minutes
        """
        return self.x.shape[1] / self.fs / 60
    
    