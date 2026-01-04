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
    color_map=None,
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
        color_map: Optional dictionary mapping state IDs to color strings.
                  Example: {0: 'blue', 1: 'red', 2: 'green'}
                  If None, uses a default color palette.
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
    
    # Generate default color map if not provided
    if color_map is None:
        import plotly.express as px
        colors = px.colors.qualitative.Plotly
        color_map = {state_id: colors[i % len(colors)] 
                    for i, state_id in enumerate(sorted(state_labels_dict.keys()))}
    
    # Create box plot data for each state
    fig = go.Figure()
    
    for state_id in sorted(state_labels_dict.keys()):
        state_name = state_labels_dict[state_id]
        mask = segment_states == state_id
        state_segments = segment_lengths[mask]
        
        fig.add_trace(go.Box(
            y=state_segments,
            name=state_name,
            marker_color=color_map.get(state_id),
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


def plot_behavior_state_frequencies_barchart(
    behavior_data,
    state_labels_dict,
    show_fig=True,
    show_percentages=True,
    show_counts=True,
    y_range=None,
    color_map=None,
    title="Behavioral State Frequency Distribution"
):
    """
    Create a Plotly bar chart showing the frequency distribution of each discrete behavioral state.
    
    This function displays how often each behavioral state occurs as a proportion of total timepoints,
    with optional display of both percentages and absolute counts.
    
    Args:
        behavior_data: Behavioral state time-series data. Can be:
                      - scipy.sparse matrix of shape (1, num_timepoints) or (num_timepoints,)
                      - numpy array of shape (num_timepoints,)
                      Values should be integer state IDs.
        state_labels_dict: Dictionary mapping state IDs (int) to state names (str).
                          Example: {0: 'waiting', 1: 'choosing', 2: 'reward'}
        show_fig: If True, display the figure. Default is True.
        show_percentages: If True, show percentages in the bar text labels. Default is True.
        show_counts: If True, show absolute counts in the bar text labels. Default is True.
        y_range: Optional tuple (min, max) to set the y-axis range. If None, auto-scale.
        color_map: Optional dictionary mapping state IDs to color strings.
                  Example: {0: 'blue', 1: 'red', 2: 'green'}
                  If None, uses a default color palette.
        title: Title for the plot. Default is "Behavioral State Frequency Distribution".
    
    Returns:
        plotly.graph_objects.Figure: The bar chart figure
        
    Example:
        >>> import numpy as np
        >>> from ncmcm.visualisers import behavioural_discrete
        >>> 
        >>> # Create sample behavioral data
        >>> behavior = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 1])
        >>> labels = {0: 'rest', 1: 'active', 2: 'reward'}
        >>> 
        >>> # Plot frequency distribution
        >>> fig = behavioural_discrete.plot_behavior_state_frequencies_barchart(
        ...     behavior, labels, y_range=(0, 0.8)
        ... )
    """
    # Convert sparse matrix to dense array if needed
    if sparse.issparse(behavior_data):
        b_dense = behavior_data.toarray().flatten()
    else:
        b_dense = np.asarray(behavior_data).flatten()
    
    # Get unique elements and their counts
    unique_elements, counts = np.unique(b_dense, return_counts=True)
    
    # Map unique elements to their labels using state_labels_dict
    labels = [state_labels_dict.get(element, str(element)) for element in unique_elements]
    
    # Calculate frequencies
    total_timepoints = len(b_dense)
    frequencies = counts / total_timepoints
    
    # Generate default color map if not provided
    if color_map is None:
        import plotly.express as px
        colors = px.colors.qualitative.Plotly
        color_map = {state_id: colors[i % len(colors)] 
                    for i, state_id in enumerate(sorted(state_labels_dict.keys()))}
    
    # Get colors for each bar in the same order as unique_elements
    bar_colors = [color_map.get(element) for element in unique_elements]
    
    # Create text labels based on user preferences
    text_labels = []
    for count, freq in zip(counts, frequencies):
        parts = []
        if show_percentages:
            parts.append(f"{freq:.2%}")
        if show_counts:
            parts.append(f"({count})")
        text_labels.append(" ".join(parts))
    
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=frequencies,
            text=text_labels,
            textposition='auto',
            marker_color=bar_colors
        )
    ])
    
    # Update layout
    layout_updates = {
        'title': title,
        'xaxis_title': "Behavioral State",
        'yaxis_title': "Frequency",
        'template': "plotly_white"
    }
    
    if y_range is not None:
        layout_updates['yaxis'] = dict(range=y_range)
    
    fig.update_layout(**layout_updates)
    
    if show_fig:
        fig.show()
    
    return fig


def plot_behavior_state_timeline(
    behavior_data,
    state_labels_dict,
    sampling_frequency=None,
    time_range=None,
    show_fig=True,
    color_map=None,
    convert_to_seconds=False,
    title="Behavioral State Timeline"
):
    """
    Create a Plotly timeline visualization showing behavioral states over time.
    
    This function creates a horizontal timeline heatmap showing how behavioral states
    change over time, useful for visualizing behavioral sequences and patterns.
    
    Args:
        behavior_data: Behavioral state time-series data. Can be:
                      - scipy.sparse matrix of shape (1, num_timepoints) or (num_timepoints,)
                      - numpy array of shape (num_timepoints,)
                      Values should be integer state IDs.
        state_labels_dict: Dictionary mapping state IDs (int) to state names (str).
                          Example: {0: 'waiting', 1: 'choosing', 2: 'reward'}
        sampling_frequency: Sampling frequency in Hz. Required if convert_to_seconds=True.
        time_range: Optional tuple (start, end) specifying the time range to display.
                   Units are samples unless convert_to_seconds=True, then in seconds.
                   If None, display the entire timeline.
        show_fig: If True, display the figure. Default is True.
        color_map: Optional dictionary mapping state IDs to color strings.
                  Example: {0: 'blue', 1: 'red', 2: 'green'}
                  If None, uses a default color palette.
        convert_to_seconds: If True, convert time axis to seconds using sampling_frequency.
                           Requires sampling_frequency to be provided.
        title: Title for the plot. Default is "Behavioral State Timeline".
    
    Returns:
        plotly.graph_objects.Figure: The timeline figure
        
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
        >>> # Plot full timeline
        >>> fig = behavioural_discrete.plot_behavior_state_timeline(
        ...     behavior, labels, sampling_frequency=100, convert_to_seconds=True
        ... )
        >>> 
        >>> # Plot specific time range
        >>> fig = behavioural_discrete.plot_behavior_state_timeline(
        ...     behavior, labels, time_range=(0, 6)
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
    
    # Apply time range if specified
    if time_range is not None:
        start_idx = int(time_range[0] * sampling_frequency) if convert_to_seconds else int(time_range[0])
        end_idx = int(time_range[1] * sampling_frequency) if convert_to_seconds else int(time_range[1])
        start_idx = max(0, start_idx)
        end_idx = min(len(b_dense), end_idx)
        b_dense = b_dense[start_idx:end_idx]
        time_offset = start_idx
    else:
        time_offset = 0
    
    # Create time axis
    time_axis = np.arange(len(b_dense)) + time_offset
    if convert_to_seconds:
        time_axis = time_axis / sampling_frequency
        x_label = "Time (seconds)"
    else:
        x_label = "Time (samples)"
    
    # Generate default color map if not provided
    if color_map is None:
        import plotly.express as px
        colors = px.colors.qualitative.Plotly
        color_map = {state_id: colors[i % len(colors)] 
                    for i, state_id in enumerate(sorted(state_labels_dict.keys()))}
    
    # Create custom discrete colorscale for the heatmap
    unique_states = np.unique(b_dense)
    n_states = len(unique_states)
    
    if n_states > 1:
        colorscale = []
        sorted_states = sorted(unique_states)
        for i, state_id in enumerate(sorted_states):
            color = color_map.get(state_id, f'rgb({i*50}, {i*100}, {200-i*50})')
            # Create discrete color blocks by duplicating colors at boundaries
            # Each state gets its color from its lower bound to upper bound
            lower_bound = i / n_states
            upper_bound = (i + 1) / n_states
            
            # Add the color at the start of this state's range
            colorscale.append([lower_bound, color])
            # Add the same color at the end of this state's range (creates discrete blocks)
            colorscale.append([upper_bound, color])
    else:
        # Single state case
        state_id = unique_states[0]
        color = color_map.get(state_id, 'rgb(100, 100, 200)')
        colorscale = [[0, color], [1, color]]
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap for behavioral states
    fig.add_trace(go.Heatmap(
        z=[b_dense],
        x=time_axis,
        y=['Behavior'],
        colorscale=colorscale,
        zmin=np.min(b_dense) - 0.5,
        zmax=np.max(b_dense) + 0.5,
        colorbar=dict(
            tickvals=sorted(unique_states),
            ticktext=[state_labels_dict.get(int(s), str(s)) for s in sorted(unique_states)],
            title="State"
        ),
        hovertemplate='Time: %{x}<br>State: %{z}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="",
        template="plotly_white",
        yaxis=dict(showticklabels=False)
    )
    
    if show_fig:
        fig.show()
    
    return fig
