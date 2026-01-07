"""
@authors 
Kerim Atak
"""

import json
import pandas as pd
import os
import numpy as np
from scipy import sparse
import pickle
import hashlib

class BanditTaskNeuroPixelsDataset:
    # Cache directory name
    CACHE_DIR = "BanditTaskNeuroPixelsDataset"
    
    # Common state transition combinations
    HOLD_TO_CHOOSING_TRANSITIONS = {
        ("hold", "choosing left"): "hold --> choosing left",
        ("hold", "choosing right"): "hold --> choosing right"
    }
    
    # Default valid state transition map for the bandit task
    # Maps each state to a list of valid next states
    DEFAULT_TRANSITION_MAP = {
        "intertrial": ["hold", "hold --> choosing left", "hold --> choosing right"],
        "hold": ["choosing left", "choosing right"],  # intertrial for aborted trials
        "choosing left": ["reward", "no reward"],
        "choosing right": ["reward", "no reward"],
        "reward": ["intertrial"],
        "no reward": ["intertrial"],
        # Combined states (when using HOLD_TO_CHOOSING_TRANSITIONS)
        "hold --> choosing left": ["reward", "no reward"],
        "hold --> choosing right": ["reward", "no reward"],
    }
    
    # Default color map for behavioral state visualization
    # Maps state names to color strings (compatible with Plotly)
    DEFAULT_COLOR_MAP = {
        "intertrial": "#c7c7c7",        # Light gray (more white-like) - between trials
        "hold": "#f39d12",              # Orange/Gold - preparatory hold state
        "choosing left": "#e74c3c",     # Red - left choice
        "choosing right": "#3498db",    # Blue - right choice
        "reward": "#2ecc71",            # Green - positive outcome
        "no reward": "#6b8e7f",         # Greyish lifeless green - negative outcome
        "hold --> choosing left": "#e74c3c",   # Red - transition to left
        "hold --> choosing right": "#3498db",  # Blue - transition to right
    }
      
    
    def __init__(self, data_path, downsample_fs=None, downsample_method='binary', good_neurons_only=True, state_transitions=None, gaussian_sigma_ms=25.0, normalize_method=None):
        """
        Initialize dataset with flexible spike representation options.

        Args:
            data_path: Path to the dataset directory
            downsample_fs: If provided, downsample the data to this sampling frequency (Hz).
            downsample_method: Method for aggregating spikes during downsampling.
                             'binary': Use OR operation (any spike -> 1)
                             'count': Sum the number of spikes in each bin
                             'rate': Firing rate in Hz (spikes per second)
                             'mean': Average of binary values (spike proportion, 0-1)
                             'gaussian': Gaussian kernel smoothing (firing rate in Hz)
            good_neurons_only: If True, only include neurons labeled as 'good' in cluster_info.tsv (default: True)
            state_transitions: Dict mapping state transition tuples to combined state names.
                             Example: {("hold", "choosing left"): "hold --> choosing left",
                                      ("hold", "choosing right"): "hold --> choosing right"}
                             When consecutive states match a transition, they are merged into one combined state.
            gaussian_sigma_ms: Standard deviation of Gaussian kernel in milliseconds (default: 25.0).
                             Only used when downsample_method='gaussian'.
            normalize_method: Method for normalizing neuronal data (default: None).
                             None: No normalization
                             'None': No normalization
                             'minmax': Min-max scaling to [0, 1] per neuron (each neuron scaled independently)
                             'minmax_global': Min-max scaling to [0, 1] using global min/max across all neurons
            
        Attributes after loading:
            data_path: Path to the dataset directory
            downsample_fs: Target sampling frequency for downsampling (Hz), or None if no downsampling
            downsample_method: Method used for spike aggregation ('binary', 'count', 'rate', 'mean', or 'gaussian')
            good_neurons_only: Whether only 'good' neurons are included
            state_transitions: Dictionary of state transition combinations
            gaussian_sigma_ms: Standard deviation of Gaussian kernel in milliseconds
            normalize_method: Method used for neuronal data normalization
            x: Neuronal time-series data (scipy.sparse.csr_matrix) of shape (num_neurons, num_timepoints). Do .toarray() to convert to dense.
            b: Behavioral time-series data (scipy.sparse.csr_matrix) of shape (num_behaviors, num_timepoints). Do .toarray() to convert to dense.
            b_labels_dict: Behavioral labels as a dictionary, mapping state IDs to state names.
            b_labels: Behavioral labels as a list, ordered by state ID.
            b_continuous: Continuous behavioral data (np.ndarray) of shape (num_timepoints,) with running average of last 10 decisions (-1 for left, 1 for right)
            trial_indices: Trial indices (np.ndarray) of shape (num_timepoints,) indicating which trial each timepoint belongs to (starting from 0, -1 for timepoints outside trials)
            block_indices: Block indices (np.ndarray) of shape (num_timepoints,) indicating which block each timepoint belongs to (starting from 0, -1 for timepoints before first block)
            block_labels: Block labels (np.ndarray) of shape (num_timepoints,) indicating the block name for each timepoint
            behavioral_time: Behavioral time (np.ndarray) of shape (num_timepoints,) with the behavioral time in milliseconds at the start of each timepoint
            fs: Sampling frequency (Hz) after any downsampling
        
        """
        self.data_path = data_path
        self.downsample_fs = downsample_fs
        self.downsample_method = downsample_method
        self.good_neurons_only = good_neurons_only
        self.state_transitions = state_transitions if state_transitions is not None else {}
        self.gaussian_sigma_ms = gaussian_sigma_ms
        self.normalize_method = normalize_method
        # Store original parameters for cache key (before any adjustments)
        self._original_downsample_fs = downsample_fs
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
        
        # Try to load from cache, otherwise process data and save to cache
        if not self._load_from_cache():
            # load data and assign to attributes
            self.load_data()
            # Save to cache for future use
            self._save_to_cache()
            # Print information about the loaded dataset
            print(f"Loaded BanditTaskNeuroPixelsDataset from {data_path}")
            print(f"Neuronal data shape: {self.x.shape}, Behavioral data shape: {self.b.shape}, Sampling frequency: {self.fs} Hz")
            print(f"Behavioral labels: {self.b_labels_dict}")
        
    def load_data(self):
        """
        Load and process all neuronal and behavioral data from the dataset directory.
        
        This method orchestrates the complete data loading pipeline:
        1. Loads spike times, cluster info, and behavioral metrics
        2. Creates sparse neuronal data matrix
        3. Downsamples data if requested
        4. Creates behavioral state, continuous, trial, and block arrays
        5. Trims waiting periods from start/end
        6. Applies state transitions if specified
        7. Relabels states to ensure continuous indexing from 0
        
        All loaded data is stored in instance attributes (x, b, b_labels_dict, etc.).
        """
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
        
        # Free memory from large spike arrays no longer needed
        del spike_times_in_neuronal_time, spike_clusters, cluster_info, spike_times_in_behavioral_time

        # Downsample neuronal data if requested
        if self.downsample_fs is not None:
            original_samples_count = self.x.shape[1]
            # Calculate target number of samples based on desired sampling frequency
            target_num_samples = int(original_samples_count * self.downsample_fs / self.fs)
            self.downsample_fs = target_num_samples * self.fs / original_samples_count  # Adjusted downsample_fs due to rounding 
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
        
        # Free memory from large temporary arrays no longer needed
        del translation_indices_neuronal_to_behavioral, metrics
        
        # Post processing: Trim waiting periods from start and end
        self._trim_waiting_periods(self.x, self.b, self.b_labels_dict)
        
        # Apply normalization if specified
        self._normalize_neuronal_data()
        
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
        
        # Free memory from dense copies
        del b_dense, new_state_array
        
    def _relabel_behavioral_states(self):
        """
        Relabel behavioral states to have continuous IDs starting from 0.
        
        This method ensures that state IDs are sequential (0, 1, 2, ...) even if
        some intermediate states were removed during processing (e.g., state transitions
        or filtering). Updates both self.b and self.b_labels_dict in place.
        """
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
        
        # Free memory from dense copies
        del b_dense, relabeled_state_array

    def _trim_waiting_periods(self, x, b, b_labels_dict):
        """
        Remove waiting periods from the start and end of the recording.
        
        Trims all timepoints in the 'waiting' state from both the beginning and end
        of the data arrays. Updates self.x, self.b, self.b_continuous, self.trial_indices,
        self.block_indices, self.block_labels, and self.behavioral_time in place.
        
        Args:
            x: Neuronal data sparse matrix
            b: Behavioral state sparse matrix
            b_labels_dict: Dictionary mapping state IDs to state names
        """
        waiting_state_id = next((k for k, v in b_labels_dict.items() if v == 'waiting'), None)
        
        # Extract dense array from sparse matrix for comparison
        b_dense = b.toarray().flatten()
        
        # from the start to the first non-waiting state
        first_non_waiting_idx = np.argmax(b_dense != waiting_state_id)
        # last non-waiting state to the end
        last_non_waiting_idx = len(b_dense) - np.argmax(b_dense[::-1] != waiting_state_id)
        
        # Free memory from dense copy
        del b_dense
        
        self.x = x[:, first_non_waiting_idx:last_non_waiting_idx]
        self.b = b[:, first_non_waiting_idx:last_non_waiting_idx]
        self.b_continuous = self.b_continuous[first_non_waiting_idx:last_non_waiting_idx]
        self.trial_indices = self.trial_indices[first_non_waiting_idx:last_non_waiting_idx]
        self.block_indices = self.block_indices[first_non_waiting_idx:last_non_waiting_idx]
        self.block_labels = self.block_labels[first_non_waiting_idx:last_non_waiting_idx]
        self.behavioral_time = self.behavioral_time[first_non_waiting_idx:last_non_waiting_idx]

    def _normalize_neuronal_data(self):
        """
        Normalize neuronal data according to the specified normalization method.

        Methods:
            - None or 'None': No normalization
            - 'minmax': Per-neuron scaling. Each neuron's activity is scaled to [0, 1]
                       via division by that neuron's max (assumes baseline 0).
            - 'minmax_global': Global scaling. All neurons scaled to [0, 1] using the
                              global maximum across all neurons and time bins.

        Notes:
            - Both methods keep the matrix sparse by scaling nonzeros only.
            - 'minmax' results in each neuron having different scaling factors.
            - 'minmax_global' uses a single scaling factor for all neurons.
        """
        if self.normalize_method is None or self.normalize_method == 'None':
            return

        if self.x is None:
            return

        method = self.normalize_method.lower()
        if method not in ("minmax", "minmax_global"):
            raise ValueError(f"Invalid normalize_method: {self.normalize_method}. Use None, 'minmax', or 'minmax_global'.")

        # Ensure CSR for efficient row ops
        x_csr = self.x.tocsr()

        if method == "minmax":
            # Per-neuron min-max scaling
            # Row-wise max (includes zeros). Convert to dense first to avoid sparse type issues.
            row_max = x_csr.max(axis=1)
            # Some SciPy versions return a sparse matrix here (e.g., coo_matrix).
            # Convert to ndarray safely, then flatten.
            if hasattr(row_max, "toarray"):
                row_max = row_max.toarray()
            row_max = np.asarray(row_max).ravel().astype(np.float32)
            # If a row has all zeros, keep zeros
            with np.errstate(divide='ignore'):
                inv = np.where(row_max > 0, 1.0 / row_max, 0.0).astype(np.float32)
            # Scale rows: diag(inv) @ X
            self.x = sparse.diags(inv, offsets=0, dtype=np.float32) @ x_csr.astype(np.float32)
        
        elif method == "minmax_global":
            # Global min-max scaling using the same scaler for all neurons
            # Find global max across all neurons and timepoints
            global_max = x_csr.max()
            # Convert to scalar if needed
            if hasattr(global_max, "toarray"):
                global_max = global_max.toarray().item()
            global_max = float(global_max)
            
            # Scale all values by global max (assumes baseline 0)
            if global_max > 0:
                self.x = x_csr.astype(np.float32) / global_max
            else:
                # All zeros, keep as is
                self.x = x_csr.astype(np.float32)

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
        Rate method: Firing rate in Hz (spikes per second).
        Mean method: Average of binary values (spike proportion, 0-1).
        Gaussian method: Gaussian kernel smoothing (firing rate in Hz).

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
        bin_size_samples = n_samples / target_num_samples

        downsampled = self._downsample_sparse(spike_matrix, bin_size_samples, n_neurons, target_num_samples)

        return downsampled

    def _downsample_sparse(self, spike_matrix, bin_size_samples, num_neurons, n_downsampled):
        """Downsample a sparse spike matrix using multiple aggregation schemes.

        Converts the input spike matrix to COO so each non-zero entry is represented
        by its row (`coo.row` -> neuron index), column (`coo.col` -> original time
        sample), and value (`coo.data`). Time columns are binned by dividing by
        `bin_size_samples` to obtain `new_time_indices`, filtered to bins
        `[0, n_downsampled)`, and then aggregated according to
        `self.downsample_method`:

        - binary: OR within each (neuron, bin), output dtype uint8
        - count: sum of spikes per bin, output dtype uint16
        - rate: firing rate Hz = counts * fs / bin_size_samples, output float32
        - mean: spike proportion per bin = counts / bin_size_samples, output float32
        - gaussian: sparse Gaussian smoothing (kernel truncated at +-3*sigma),
          output float32 firing rates

        Args:
            spike_matrix: scipy.sparse.csr_matrix with shape (num_neurons, n_samples).
            bin_size_samples: float bin width in original samples.
            num_neurons: int number of neurons (rows) in the output.
            n_downsampled: int number of time bins in the output.

        Returns:
            scipy.sparse.csr_matrix of shape (num_neurons, n_downsampled) with dtype
            depending on the chosen method (see above).
        """
        # Convert to COO format for easier manipulation
        # coo.row: neuron indices, coo.col: time indices, coo.data: spike counts (1s) https://scipy-lectures.org/advanced/scipy_sparse/coo_matrix.html
        coo = spike_matrix.tocoo() 
        
        # Bin the time indices
        new_time_indices = (coo.col / bin_size_samples).astype(int) # e.g. 0,1,16,2,10 --> 0,0,7,1,5 for bin_size=2

        # Keep only valid bins
        valid_mask = new_time_indices < n_downsampled # 0,0,7,1,5 --> True,True,False,True,True if n_downsampled=6
        new_time_indices = new_time_indices[valid_mask] # 0,0,7,1,5 --> 0,0,1,5 (7 is dropped as its bin is out of range)
        neuron_indices = coo.row[valid_mask] # n0,n0,n8,n3,n10 --> n0,n0,n3,n10 (n8 is dropped as its bin is out of range)

        if self.downsample_method == 'binary':
            # Binary method: OR operation (if spike in bin, mark as 1)
            unique_spikes = np.unique(np.column_stack([neuron_indices, new_time_indices]), axis=0) # n0,n0,n3,n10; 0,0,1,5 --> n0,0; n3,1; n10,5 (duplicate n0,0 removed)
            data = np.ones(len(unique_spikes), dtype=np.uint8)
            downsampled = sparse.csr_matrix(
                (data, (unique_spikes[:, 0], unique_spikes[:, 1])), 
                shape=(num_neurons, n_downsampled),
                dtype=np.uint8
            )
        elif self.downsample_method == 'count':
            # Count method: sum spikes in each bin
            # Count occurrences of each (neuron, time) pair
            spike_coords = np.column_stack([neuron_indices, new_time_indices])
            unique_coords, counts = np.unique(spike_coords, axis=0, return_counts=True)
            
            downsampled = sparse.csr_matrix(
                (counts, (unique_coords[:, 0], unique_coords[:, 1])),
                shape=(num_neurons, n_downsampled),
                dtype=np.uint16  # Use uint16 to allow counts > 255
            )
        elif self.downsample_method == 'rate':
            # Rate method: firing rate in Hz (spikes per second)
            # Count occurrences of each (neuron, time) pair
            spike_coords = np.column_stack([neuron_indices, new_time_indices])
            unique_coords, counts = np.unique(spike_coords, axis=0, return_counts=True)
            
            # Convert counts to firing rate in Hz
            # bin_size is in original samples, self.fs is original sampling frequency
            # bin_duration_seconds = bin_size / self.fs
            # rate = counts / bin_duration_seconds = counts * self.fs / bin_size
            rates = (counts / (bin_size_samples / self.fs)).astype(np.float32)
            
            downsampled = sparse.csr_matrix(
                (rates, (unique_coords[:, 0], unique_coords[:, 1])),
                shape=(num_neurons, n_downsampled),
                dtype=np.float32
            )
        elif self.downsample_method == 'mean':
            # Mean method: average of binary values (spike proportion per bin)
            # Count occurrences of each (neuron, time) pair
            spike_coords = np.column_stack([neuron_indices, new_time_indices])
            unique_coords, counts = np.unique(spike_coords, axis=0, return_counts=True)
            
            # Convert counts to mean (proportion of bin samples with spikes)
            # mean = counts / bin_size (values between 0 and 1)
            means = (counts / bin_size_samples).astype(np.float32)
            
            downsampled = sparse.csr_matrix(
                (means, (unique_coords[:, 0], unique_coords[:, 1])),
                shape=(num_neurons, n_downsampled),
                dtype=np.float32
            )
        elif self.downsample_method == 'gaussian':
            # Gaussian smoothing with direct sparse computation
            # Apply Gaussian kernel to each spike and accumulate to downsampled bins
            # Avoids materializing any dense arrays
            
            # Convert sigma from milliseconds to original sample units
            gaussian_sigma_samples = (self.gaussian_sigma_ms / 1000.0) * self.fs
            
            # Precompute Gaussian kernel truncation radius (±3σ covers 99.7%)
            kernel_radius_samples = int(np.ceil(3 * gaussian_sigma_samples))
            
            # Precompute normalization constant for Gaussian kernel
            # The Gaussian PDF: (1 / (sigma * sqrt(2π))) * exp(-0.5 * (x/sigma)^2)
            gaussian_norm = 1.0 / (gaussian_sigma_samples * np.sqrt(2 * np.pi))
            
            # Dictionary to accumulate Gaussian contributions: {(neuron, time_bin): value}
            contributions = {}
            
            # Get spike times (in original sample coordinates) and neuron IDs (already filtered by valid_mask)
            spike_times_orig = coo.col[valid_mask]
            spike_neuron_ids = neuron_indices
            
            # For each spike, add Gaussian kernel contributions to nearby downsampled bins
            for spike_idx in range(len(spike_times_orig)):
                spike_time_orig = spike_times_orig[spike_idx]  # Original sample coordinate
                neuron_id = spike_neuron_ids[spike_idx]
                
                # Determine range of downsampled bins affected by this spike
                # Find min/max time coordinates (in original samples) within ±3σ of spike
                min_time_orig = max(0, spike_time_orig - kernel_radius_samples)
                max_time_orig = min(spike_matrix.shape[1] - 1, spike_time_orig + kernel_radius_samples)
                
                # Convert original sample coordinates to downsampled bin indices
                min_bin_down = int(min_time_orig / bin_size_samples)
                max_bin_down = min(n_downsampled - 1, int(max_time_orig / bin_size_samples))
                
                # Add Gaussian contributions to affected downsampled bins
                for bin_idx_down in range(min_bin_down, max_bin_down + 1):
                    # Calculate bin center in original sample coordinates
                    bin_center_orig = (bin_idx_down + 0.5) * bin_size_samples
                    
                    # Distance from spike to bin center (in original samples)
                    distance_orig = bin_center_orig - spike_time_orig
                    
                    # Only add contribution if within kernel radius
                    if abs(distance_orig) <= kernel_radius_samples:
                        # Compute Gaussian kernel value
                        kernel_value = gaussian_norm * np.exp(-0.5 * (distance_orig / gaussian_sigma_samples) ** 2)
                        
                        # Accumulate contribution (will convert to firing rate at the end)
                        key = (neuron_id, bin_idx_down)
                        if key in contributions:
                            contributions[key] += kernel_value
                        else:
                            contributions[key] = kernel_value
            
            # Convert contributions dictionary to sparse matrix
            if contributions:
                keys = list(contributions.keys())
                neuron_ids = np.array([k[0] for k in keys], dtype=np.int32)
                time_bins = np.array([k[1] for k in keys], dtype=np.int32)
                values = np.array([contributions[k] for k in keys], dtype=np.float32)
                
                # Convert to firing rate in Hz (multiply by sampling frequency)
                values = values * self.fs
                
                downsampled = sparse.csr_matrix(
                    (values, (neuron_ids, time_bins)),
                    shape=(num_neurons, n_downsampled),
                    dtype=np.float32
                )
            else:
                # No spikes, return empty sparse matrix
                downsampled = sparse.csr_matrix(
                    (num_neurons, n_downsampled),
                    dtype=np.float32
                )
        else:
            raise ValueError(f"Invalid downsample_method: {self.downsample_method}. Must be 'binary', 'count', 'rate', 'mean', or 'gaussian'.")

        return downsampled

    def _create_sparse_neuronal_data_matrix(self, spike_times, spike_clusters, cluster_info):
        """
        Create a sparse binary matrix representing neuronal spike data.
        
        Constructs a sparse CSR matrix where rows represent neurons and columns represent
        time bins. Each cell contains 1 if the neuron fired at that time, 0 otherwise.
        
        Args:
            spike_times: Array of spike times in milliseconds (integer values)
            spike_clusters: Array of cluster IDs corresponding to each spike
            cluster_info: pandas DataFrame with 'cluster_id' column mapping neurons to cluster IDs
            
        Returns:
            scipy.sparse.csr_matrix: Binary spike matrix of shape (num_neurons, num_time_bins)
                with dtype uint8, where 1 indicates a spike occurred
        """
        num_neurons = len(cluster_info) # number of neurons to include
        num_time_bins = int(np.max(spike_times)) + 1 # max time bin (assuming spike_times are in integer ms)
                
        # Prepare data for sparse matrix construction
        neuron_indices = [] # row indices
        time_indices = [] # column indices
        
        # Map cluster IDs to neuron indices
        neuron_id_to_index = {neuron_id: idx for idx, neuron_id in enumerate(cluster_info['cluster_id'])}
        
        for spike_time, cluster_id in zip(spike_times, spike_clusters):
            if cluster_id in neuron_id_to_index:
                neuron_indices.append(neuron_id_to_index[cluster_id])
                time_indices.append(int(spike_time))
        
        # Create sparse matrix
        data = np.ones(len(neuron_indices), dtype=np.uint8)  # Use uint8 to save memory
        sparse_matrix = sparse.csr_matrix(
            (data, (neuron_indices, time_indices)), 
            shape=(num_neurons, num_time_bins), 
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
        
        Returns:
            tuple: (state_array_sparse, state_labels) where:
                - state_array_sparse: scipy.sparse.csr_matrix of shape (1, neuronal_length)
                - state_labels: dict mapping state IDs (int) to state names (str)
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
        """
        Load cluster information from TSV file and optionally filter for good neurons.
        
        Args:
            data_path: Path to the dataset directory containing cluster_info.tsv
        
        Returns:
            pandas.DataFrame: Cluster information with 'cluster_id' and 'group' columns,
                            filtered to only 'good' neurons if self.good_neurons_only is True
        """
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
    
    def _get_cache_filename(self):
        """
        Generate a unique cache filename based on dataset parameters.
        
        Returns:
            str: Filename for caching this dataset configuration
        """
        # Create a dictionary of parameters that affect the processed data
        # Use original downsample_fs (before adjustment) for consistent cache key
        params = {
            'downsample_fs': self._original_downsample_fs,
            'downsample_method': self.downsample_method,
            'good_neurons_only': self.good_neurons_only,
            'state_transitions': str(sorted(self.state_transitions.items())) if self.state_transitions else 'None',
            'gaussian_sigma_ms': self.gaussian_sigma_ms,
            'normalize_method': self.normalize_method
        }
        
        # Create a string representation of parameters
        param_str = '_'.join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Create a hash of the parameter string to keep filename manageable
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        # Create readable filename with key parameters and hash
        filename_parts = [
            f"fs{self._original_downsample_fs}" if self._original_downsample_fs else "fsNone",
            self.downsample_method,
            "good" if self.good_neurons_only else "all",
            f"norm{self.normalize_method}" if self.normalize_method else "normNone",
            param_hash
        ]
        
        filename = '_'.join(filename_parts) + '.pkl'
        return filename
    
    def _get_cache_dir(self):
        """
        Get or create the cache directory path.
        
        Returns:
            str: Path to cache directory
        """
        cache_dir = os.path.join(self.data_path, self.CACHE_DIR)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _save_to_cache(self):
        """
        Save the processed dataset to cache file.
        """
        try:
            cache_dir = self._get_cache_dir()
            cache_filename = self._get_cache_filename()
            cache_path = os.path.join(cache_dir, cache_filename)
            
            # Prepare data dictionary to cache
            cache_data = {
                'x': self.x,
                'b': self.b,
                'b_labels_dict': self.b_labels_dict,
                'b_labels': self.b_labels,
                'b_continuous': self.b_continuous,
                'trial_indices': self.trial_indices,
                'block_indices': self.block_indices,
                'block_labels': self.block_labels,
                'behavioral_time': self.behavioral_time,
                'fs': self.fs,
                # Save parameters for verification
                'params': {
                    'downsample_fs': self._original_downsample_fs,  # Use original for cache key
                    'actual_downsample_fs': self.downsample_fs,  # Save actual adjusted value
                    'downsample_method': self.downsample_method,
                    'good_neurons_only': self.good_neurons_only,
                    'state_transitions': self.state_transitions,
                    'gaussian_sigma_ms': self.gaussian_sigma_ms,
                    'normalize_method': self.normalize_method
                }
            }
            
            # Save to pickle file
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Dataset cached to: {cache_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save dataset to cache: {e}")
    
    def _load_from_cache(self):
        """
        Try to load processed dataset from cache.
        
        Returns:
            bool: True if successfully loaded from cache, False otherwise
        """
        try:
            cache_dir = self._get_cache_dir()
            cache_filename = self._get_cache_filename()
            cache_path = os.path.join(cache_dir, cache_filename)
            
            # Check if cache file exists
            if not os.path.exists(cache_path):
                return False
            
            print(f"Loading dataset from cache: {cache_path}")
            
            # Load from pickle file
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify parameters match (safety check)
            # Use original downsample_fs for comparison
            cached_params = cache_data.get('params', {})
            if cached_params.get('downsample_fs') != self._original_downsample_fs or \
               cached_params.get('downsample_method') != self.downsample_method or \
               cached_params.get('good_neurons_only') != self.good_neurons_only or \
               cached_params.get('normalize_method') != self.normalize_method:
                print("Warning: Cached parameters mismatch, reprocessing data...")
                return False
            
            # Restore data from cache
            self.x = cache_data['x']
            self.b = cache_data['b']
            self.b_labels_dict = cache_data['b_labels_dict']
            self.b_labels = cache_data['b_labels']
            self.b_continuous = cache_data['b_continuous']
            self.trial_indices = cache_data['trial_indices']
            self.block_indices = cache_data['block_indices']
            self.block_labels = cache_data['block_labels']
            self.behavioral_time = cache_data['behavioral_time']
            self.fs = cache_data['fs']
            # Update downsample_fs to the actual value (may be slightly adjusted)
            self.downsample_fs = cache_data['params'].get('actual_downsample_fs', self.fs)
            
            print(f"Successfully loaded cached dataset")
            print(f"Neuronal data shape: {self.x.shape}, Behavioral data shape: {self.b.shape}, Sampling frequency: {self.fs} Hz")
            print(f"Behavioral labels: {self.b_labels_dict}")
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load from cache: {e}")
            print("Reprocessing data...")
            return False
    
    def get_recording_length_mins(self):
        """
        Get the length of the recording in minutes.
        
        Returns:
            float: Recording length in minutes
        """
        return self.x.shape[1] / self.fs / 60
    
    def check_state_transitions(self, transition_map=None):
        """
        Check if state transitions in the behavioral data follow a valid transition map.
        
        Args:
            transition_map: Dictionary mapping state names to lists of valid next states.
                          Example: {"hold": ["choosing left", "choosing right"],
                                   "choosing left": ["reward", "no reward"],
                                   "choosing right": ["reward", "no reward"]}
                          If None, returns all observed transitions without validation.
        
        Returns:
            dict: Dictionary with:
                - 'valid': bool, True if all transitions are valid (or if no map provided)
                - 'observed_transitions': dict mapping (from_state, to_state) tuples to counts
                - 'invalid_transitions': list of dicts with 'from', 'to', 'count', 'indices'
                                        for each invalid transition (empty if no map provided)
        
        Example:
            >>> dataset = BanditTaskNeuroPixelsDataset(data_path)
            >>> transition_map = {
            ...     "hold": ["choosing left", "choosing right"],
            ...     "choosing left": ["reward", "no reward"],
            ...     "choosing right": ["reward", "no reward"],
            ...     "reward": ["intertrial"],
            ...     "no reward": ["intertrial"],
            ...     "intertrial": ["hold", "waiting"],
            ... }
            >>> result = dataset.check_state_transitions(transition_map)
            >>> if not result['valid']:
            ...     print(f"Found {len(result['invalid_transitions'])} invalid transition types")
        """
        b_dense = self.b.toarray().flatten()
        
        # Find all transitions (where state changes)
        state_changes = np.where(np.diff(b_dense) != 0)[0]
        
        # Count observed transitions
        observed_transitions = {}
        transition_indices = {}  # Store indices for each transition type
        
        for idx in state_changes:
            from_state_id = b_dense[idx]
            to_state_id = b_dense[idx + 1]
            from_state = self.b_labels_dict[from_state_id]
            to_state = self.b_labels_dict[to_state_id]
            
            key = (from_state, to_state)
            observed_transitions[key] = observed_transitions.get(key, 0) + 1
            
            if key not in transition_indices:
                transition_indices[key] = []
            transition_indices[key].append(idx)
        
        # If no transition map provided, just return observed transitions
        if transition_map is None:
            return {
                'valid': True,
                'observed_transitions': observed_transitions,
                'invalid_transitions': []
            }
        
        # Validate transitions against the map
        invalid_transitions = []
        
        for (from_state, to_state), count in observed_transitions.items():
            # Check if the from_state has defined transitions
            if from_state in transition_map:
                valid_next_states = transition_map[from_state]
                if to_state not in valid_next_states:
                    invalid_transitions.append({
                        'from': from_state,
                        'to': to_state,
                        'count': count,
                        'indices': transition_indices[(from_state, to_state)]
                    })
            else:
                # State not in map - could be considered invalid or just undefined
                # Here we treat undefined source states as potentially invalid
                invalid_transitions.append({
                    'from': from_state,
                    'to': to_state,
                    'count': count,
                    'indices': transition_indices[(from_state, to_state)],
                    'reason': 'source state not in transition map'
                })
        
        return {
            'valid': len(invalid_transitions) == 0,
            'observed_transitions': observed_transitions,
            'invalid_transitions': invalid_transitions
        }
    