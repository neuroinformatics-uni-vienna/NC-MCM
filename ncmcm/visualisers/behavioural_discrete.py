"""
Visualisers for discrete behavioral states.

@authors
Kerim Atak
"""

import numpy as np
import plotly.graph_objects as go
from scipy import sparse


def plot_behavior_state_lengths_boxplot(
    behavior_data,
    state_labels_dict,
    sampling_frequency=None,
    show_fig=True,
    convert_to_seconds=False,
    title="Behavioral State Segment Length Distributions"
):
    """
    Create a Plotly boxplot showing the distribution of segment lengths for each discrete behavioral state.
    
    This function works with any discrete behavioral data where consecutive timepoints 
    can be grouped into segments of the same state.
    
    Args:
        behavior_data: Behavioral state time-series data. Can be:
                      - scipy.sparse matrix of shape (1, num_timepoints) or (num_timepoints,)
                      - numpy array of shape (num_timepoints,)
                      Values should be integer state IDs.
        state_labels_dict: Dictionary mapping state IDs (int) to state names (str).
                          Example: {0: 'waiting', 1: 'choosing', 2: 'reward'}
        sampling_frequency: Sampling frequency in Hz. Required if convert_to_seconds=True.
        show_fig: If True, display the figure. Default is True.
        convert_to_seconds: If True, convert time steps to seconds using the sampling frequency.
                           Requires sampling_frequency to be provided.
        title: Title for the plot. Default is "Behavioral State Segment Length Distributions".
    
    Returns:
        plotly.graph_objects.Figure: The boxplot figure
        
    Raises:
        ValueError: If convert_to_seconds=True but sampling_frequency is not provided.
        
    Example:
        >>> import numpy as np
        >>> from ncmcm.visualisers import behavioural_discrete
        >>> 
        >>> # Create sample behavioral data
        >>> behavior = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 1])
        >>> labels = {0: 'rest', 1: 'active', 2: 'reward'}
        >>> 
        >>> # Plot boxplot
        >>> fig = behavioural_discrete.plot_behavior_state_lengths_boxplot(
        ...     behavior, labels, sampling_frequency=100, convert_to_seconds=True
        ... )
    """
    # Validate inputs
    if convert_to_seconds and sampling_frequency is None:
        raise ValueError("sampling_frequency must be provided when convert_to_seconds=True")
    
    # Convert sparse matrix to dense array if needed
    if sparse.issparse(behavior_data):
        b_dense = behavior_data.toarray().flatten()
    else:
        b_dense = np.asarray(behavior_data).flatten()
    
    # Find where the state changes to identify segment boundaries
    state_changes = np.where(np.diff(b_dense) != 0)[0] + 1
    segment_boundaries = np.concatenate([[0], state_changes, [len(b_dense)]])
    
    # Calculate segment lengths and identify the state of each segment
    segment_lengths = np.diff(segment_boundaries)
    segment_states = b_dense[segment_boundaries[:-1]]  # State at start of each segment
    
    # Convert to seconds if requested
    if convert_to_seconds:
        segment_lengths = segment_lengths / sampling_frequency
        y_label = "Segment Length (seconds)"
    else:
        y_label = "Segment Length (time steps)"
    
    # Create box plot data for each state
    fig = go.Figure()
    
    for state_id in sorted(state_labels_dict.keys()):
        state_name = state_labels_dict[state_id]
        mask = segment_states == state_id
        state_segments = segment_lengths[mask]
        
        fig.add_trace(go.Box(
            y=state_segments,
            name=state_name,
            boxpoints='outliers',  # Show only outliers as points
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        xaxis_title="Behavioral State",
        showlegend=False,
        height=500
    )
    
    if show_fig:
        fig.show()
    
    return fig


def get_behavior_state_segments(behavior_data, state=None, state_labels_dict=None):
    """
    Get all consecutive segment lengths for discrete behavioral states.
    
    Args:
        behavior_data: Behavioral state time-series data. Can be:
                      - scipy.sparse matrix of shape (1, num_timepoints) or (num_timepoints,)
                      - numpy array of shape (num_timepoints,)
                      Values should be integer state IDs.
        state: Optional. If provided, only return segments for this specific state.
               Can be an int (state id) or str (state name, requires state_labels_dict).
               If None, return segments for all states.
        state_labels_dict: Optional. Dictionary mapping state IDs (int) to state names (str).
                          Required if state is provided as a string.
    
    Returns:
        np.ndarray: Array of segment lengths (in time steps)
        
    Example:
        >>> import numpy as np
        >>> from ncmcm.visualisers import behavioural_discrete
        >>> 
        >>> behavior = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        >>> labels = {0: 'rest', 1: 'active', 2: 'reward'}
        >>> 
        >>> # Get all segment lengths
        >>> all_segments = behavioural_discrete.get_behavior_state_segments(behavior)
        >>> print(all_segments)  # [3, 2, 4]
        >>> 
        >>> # Get segments for specific state
        >>> rest_segments = behavioural_discrete.get_behavior_state_segments(
        ...     behavior, state='rest', state_labels_dict=labels
        ... )
        >>> print(rest_segments)  # [3]
    """
    # Convert sparse matrix to dense array if needed
    if sparse.issparse(behavior_data):
        b_dense = behavior_data.toarray().flatten()
    else:
        b_dense = np.asarray(behavior_data).flatten()
    
    # Convert state name to id if necessary
    state_id = None
    if isinstance(state, str):
        if state_labels_dict is None:
            raise ValueError("state_labels_dict must be provided when state is a string")
        state_id = next((k for k, v in state_labels_dict.items() if v == state), None)
        if state_id is None:
            raise ValueError(f"State '{state}' not found in state_labels_dict")
    elif state is not None:
        state_id = state
    
    # Find where the state changes
    state_changes = np.where(np.diff(b_dense) != 0)[0] + 1
    
    # Add start and end indices
    segment_boundaries = np.concatenate([[0], state_changes, [len(b_dense)]])
    
    # Calculate segment lengths and get corresponding states
    segment_lengths = np.diff(segment_boundaries)
    segment_states = b_dense[segment_boundaries[:-1]]  # State at start of each segment
    
    if state_id is not None:
        # Filter for specific state
        mask = segment_states == state_id
        segment_lengths = segment_lengths[mask]
    
    return segment_lengths


def get_behavior_state_length_statistics(behavior_data, state=None, state_labels_dict=None):
    """
    Get comprehensive statistics on behavioral state segment durations.
    
    Args:
        behavior_data: Behavioral state time-series data. Can be:
                      - scipy.sparse matrix of shape (1, num_timepoints) or (num_timepoints,)
                      - numpy array of shape (num_timepoints,)
                      Values should be integer state IDs.
        state: Optional. If provided, only consider segments for this specific state.
               Can be an int (state id) or str (state name, requires state_labels_dict).
        state_labels_dict: Optional. Dictionary mapping state IDs (int) to state names (str).
                          Required if state is provided as a string.
    
    Returns:
        dict: Dictionary containing:
            - 'state': The state name (or 'all' if state is None)
            - 'count': Number of segments
            - 'min': Minimum segment length
            - 'max': Maximum segment length
            - 'mean': Mean segment length
            - 'median': Median segment length
            - 'std': Standard deviation of segment lengths
            - 'percentile_25': 25th percentile
            - 'percentile_75': 75th percentile
            - 'percentile_90': 90th percentile
            - 'percentile_95': 95th percentile
            - 'percentile_99': 99th percentile
            
    Example:
        >>> import numpy as np
        >>> from ncmcm.visualisers import behavioural_discrete
        >>> 
        >>> behavior = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        >>> labels = {0: 'rest', 1: 'active', 2: 'reward'}
        >>> 
        >>> # Get statistics for all states
        >>> stats = behavioural_discrete.get_behavior_state_length_statistics(behavior)
        >>> 
        >>> # Get statistics for specific state
        >>> rest_stats = behavioural_discrete.get_behavior_state_length_statistics(
        ...     behavior, state='rest', state_labels_dict=labels
        ... )
    """
    segments = get_behavior_state_segments(behavior_data, state, state_labels_dict)
    
    if len(segments) == 0:
        raise ValueError("No segments found for the specified state")
    
    # Determine state name for reporting
    if state is None:
        state_name = 'all'
    elif isinstance(state, str):
        state_name = state
    else:
        state_name = state_labels_dict[state] if state_labels_dict else f'state_{state}'
    
    return {
        'state': state_name,
        'count': len(segments),
        'min': int(np.min(segments)),
        'max': int(np.max(segments)),
        'mean': float(np.mean(segments)),
        'median': float(np.median(segments)),
        'std': float(np.std(segments)),
        'percentile_25': float(np.percentile(segments, 25)),
        'percentile_75': float(np.percentile(segments, 75)),
        'percentile_90': float(np.percentile(segments, 90)),
        'percentile_95': float(np.percentile(segments, 95)),
        'percentile_99': float(np.percentile(segments, 99)),
    }


def get_all_behavior_states_length_statistics(behavior_data, state_labels_dict):
    """
    Get comprehensive statistics on segment durations for all behavioral states.
    
    Args:
        behavior_data: Behavioral state time-series data. Can be:
                      - scipy.sparse matrix of shape (1, num_timepoints) or (num_timepoints,)
                      - numpy array of shape (num_timepoints,)
                      Values should be integer state IDs.
        state_labels_dict: Dictionary mapping state IDs (int) to state names (str).
    
    Returns:
        dict: Dictionary mapping state names to their statistics dictionaries
        
    Example:
        >>> import numpy as np
        >>> from ncmcm.visualisers import behavioural_discrete
        >>> 
        >>> behavior = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        >>> labels = {0: 'rest', 1: 'active', 2: 'reward'}
        >>> 
        >>> all_stats = behavioural_discrete.get_all_behavior_states_length_statistics(
        ...     behavior, labels
        ... )
        >>> for state_name, stats in all_stats.items():
        ...     print(f"{state_name}: {stats['count']} segments, mean={stats['mean']:.2f}")
    """
    stats = {}
    for state_id, state_name in state_labels_dict.items():
        stats[state_name] = get_behavior_state_length_statistics(
            behavior_data, state=state_id, state_labels_dict=state_labels_dict
        )
    return stats
