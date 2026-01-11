"""
Visualization functions for Two-Arm Bandit task data.

Author: Kerim Atak (kerim.atak@univie.ac.at)
"""

import json
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ncmcm.data_loaders.bandit_task import BanditTaskNeuroPixelsDataset
import argparse


def generate_interactive_session_plot(data_session_path, output_filename='interactive_session_plot.html', window=10, state_color_map=None):
    # Use default color map from BanditTaskNeuroPixelsDataset if not provided
    if state_color_map is None:
        state_color_map = BanditTaskNeuroPixelsDataset.DEFAULT_COLOR_MAP
    """
    Generate an interactive HTML plot for a Two-Arm Bandit session.
    
    Creates a 3-subplot visualization showing:
    1. Trial-by-trial choices with timing information
    2. Behavioral states over time
    3. Moving average of left choice rate
    
    Parameters
    ----------
    data_session_path : str
        Path to the session directory containing metrics.json
    output_filename : str, optional
        Name of the output HTML file (default: 'interactive_session_plot.html')
    window : int, optional
        Window size for moving average calculation (default: 10)
    state_color_map : dict, optional
        Dictionary mapping state names to color strings (hex codes or named colors).
        Defaults to BanditTaskNeuroPixelsDataset.DEFAULT_COLOR_MAP.
        Example: {'hold': '#f39d12', 'choosing left': '#e74c3c', ...}
    
    Returns
    -------
    dict
        Dictionary containing summary statistics about the session
    """
    # Load data
    with open(os.path.join(data_session_path, 'metrics.json'), 'r') as f:
        data = json.load(f)

    trials = data['metrics']['trials']
    blocks = data['metrics']['blocks']
    states = data['metrics'].get('states', [])  # Get states, default to empty list if not present

    print("\nNumber of trials:", len(trials))
    print("Number of states:", len(states))

    # Check for missing 'choice' keys
    missing_choice = [i for i, t in enumerate(trials) if 'choice' not in t]
    print(f"\nTrials missing 'choice': {len(missing_choice)}")

    # Analyze states
    if states:
        unique_states = list(set([state[1] for state in states]))
        print(f"Unique states: {unique_states}")
    else:
        unique_states = []
        print("No states data found in the file")

    # Create figure with 3 subplots (adding states plot in middle)
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.2, 0.4],  # Adjusted heights
        vertical_spacing=0.08,
        subplot_titles=('Trial-by-Trial Choices (Click on a trial to see timing)', 
                        'States Over Time',
                        'Moving Average: Left Choice Rate')
    )

    # ========================================================================
    # PLOT 1: Trial-by-trial choices (top)
    # ========================================================================
    for i, trial in enumerate(trials):
        x_time = trial.get('start', i)
        
        if 'choice' not in trial:
            marker_symbol = 'circle'
            y = 0.5
            color = 'gray'
            hover_text = f'<b>Trial: {i}</b><br>Time: {x_time}<br>No choice recorded'
        else:
            if trial['choice'] == 'r':
                marker_symbol = 'triangle-up'
                y = 0
            else:
                marker_symbol = 'triangle-down'
                y = 1
            
            # Use colors from state_color_map for rewarded/unrewarded outcomes
            if trial.get('rewarded', False):
                # Try to get reward state color, fallback to blue
                color = state_color_map.get('reward', state_color_map.get('rewarded', '#3498db'))
            else:
                # Try to get no reward/unrewarded/ITI state color, fallback to red
                color = state_color_map.get('no reward', state_color_map.get('ITI', state_color_map.get('unrewarded', '#e74c3c')))
            
            prob_lr = trial.get('set reward probabs l/r', [None, None])
            hover_text = f"""<b>Trial: {i} - Click to view timing</b><br>
<br><b>Timing:</b><br>
Start: {trial.get('start', 'N/A')}<br>
Hold start: {trial.get('hold_start', 'N/A')}<br>
T choosing: {trial.get('t choosing', 'N/A')}<br>
T chosen: {trial.get('t chosen', 'N/A')}<br>
<br><b>Choice & Outcome:</b><br>
Choice: {trial.get('choice', 'N/A')}<br>
Rewarded: {trial.get('rewarded', 'N/A')}<br>
<br><b>Block Info:</b><br>
Reward prob L/R: [{prob_lr[0]:.3f}, {prob_lr[1]:.3f}]<br>
Achievement reached: {trial.get('Achievement reached', 'N/A')}<br>
<br><b>Recent Performance:</b><br>
Reward rate recent: {trial.get('reward rate recent', 'N/A')}<br>
Rew rate recent rights: {trial.get('rew rate in recent rights', 'N/A')}<br>
Rew rate recent lefts: {trial.get('rew rate in recent lefts', 'N/A')}
"""
        
        fig.add_trace(go.Scatter(
            x=[x_time],
            y=[y],
            mode='markers',
            marker=dict(
                symbol=marker_symbol,
                size=8,
                color=color,
                line=dict(width=0.5, color='white')
            ),
            customdata=[[i, trial.get('start', 0), trial.get('hold_start', 0), 
                         trial.get('t choosing', 0), trial.get('t chosen', 0),
                         trial.get('choice', ''), trial.get('rewarded', False)]],
            showlegend=False,
            hovertemplate=hover_text + '<extra></extra>',
            name=f'trial_{i}'
        ), row=1, col=1)

    # Add placeholder for connecting line
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(color='darkgray', width=3, dash='solid'),
        showlegend=False,
        name='timing_line'
    ), row=1, col=1)

    # Add placeholders for timing markers (will be updated via JavaScript)
    for timing_type in ['start_marker', 'hold_marker', 'choosing_marker', 'chosen_marker']:
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='markers+text',
            marker=dict(size=15, line=dict(width=2, color='white')),
            textposition='top center',
            showlegend=False,
            name=timing_type
        ), row=1, col=1)

    # ========================================================================
    # PLOT 2: States visualization (middle)
    # ========================================================================
    if states and len(states) > 0:
        # Use the provided color map (defaults to BanditTaskNeuroPixelsDataset.DEFAULT_COLOR_MAP)
        state_colors = state_color_map.copy()
        
        # Add a color for any unknown states
        for state in unique_states:
            if state not in state_colors:
                state_colors[state] = '#CCCCCC'  # Gray for unknown states
        
        # Sort states by timestamp to ensure correct ordering
        states_sorted = sorted(states, key=lambda x: x[0])
        
        # Create state timeline visualization
        for i, (timestamp, state_name) in enumerate(states_sorted):
            # Determine the duration of this state
            if i < len(states) - 1:
                next_timestamp = states_sorted[i + 1][0]
                duration = next_timestamp - timestamp
            else:
                # For the last state, use a default duration or find the session end
                duration = 1
            
            # Determine border color - use darker version of fill color, or black for white
            fill_color = state_colors.get(state_name, '#CCCCCC')
            if fill_color == 'white':
                border_color = 'darkgray'
            else:
                border_color = fill_color
            
            # Add rectangle for state duration with visible border
            fig.add_trace(go.Scatter(
                x=[timestamp, timestamp + duration, timestamp + duration, timestamp, timestamp],
                y=[0, 0, 1, 1, 0],
                fill='toself',
                fillcolor=fill_color,
                line=dict(color=border_color, width=1),
                mode='lines',
                showlegend=False,
                hoverinfo='text',
                text=f"<b>State: {state_name}</b><br>Start: {timestamp}<br>Duration: {duration} samples <br>End: {timestamp + duration}<br>Index: {i} of {len(states_sorted)}",
                name=f'state_{i}'
            ), row=2, col=1)
        
        # Add legend for states
        legend_added = set()
        for state in unique_states:
            if state not in legend_added:
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=15, color=state_colors.get(state, '#CCCCCC'), symbol='square'),
                    name=f'State: {state}',
                    legendgroup='states',
                    showlegend=True
                ), row=2, col=1)
                legend_added.add(state)

    else:
        # Add placeholder text when no states data
        fig.add_annotation(
            text="No states data available",
            x=0.5,
            y=0.5,
            xref="x2",
            yref="y2",
            showarrow=False,
            font=dict(size=16, color="gray"),
            row=2, col=1
        )

    # ========================================================================
    # PLOT 3: Moving average of left choices (bottom)
    # ========================================================================
    valid_trials_with_choice = [t for t in trials if 'choice' in t]
    times = [t.get('start', 0) for t in valid_trials_with_choice]
    choices = [1 if t['choice'] == 'l' else 0 for t in valid_trials_with_choice]

    left_choice_rate = np.convolve(choices, np.ones(window)/window, mode='valid')
    times_ma = times[window-1:]

    fig.add_trace(go.Scatter(
        x=times_ma,
        y=left_choice_rate,
        mode='lines',
        line=dict(color='purple', width=2),
        name=f'Left choice rate (window={window})',
        hovertemplate='Time: %{x}<br>Left choice rate: %{y:.2f}<extra></extra>',
        legendgroup='behavior'
    ), row=3, col=1)

    fig.add_hline(y=0.5, line=dict(color='gray', dash='dash'), row=3, col=1)

    # ========================================================================
    # Add block boundaries to all plots
    # ========================================================================
    for j, block in enumerate(blocks):
        block_time = block.get('t', 0)
        
        if j < len(blocks) - 1:
            next_block_time = blocks[j + 1].get('t', float('inf'))
        else:
            next_block_time = float('inf')
        
        trials_in_block = [t for t in trials if block_time <= t.get('start', 0) < next_block_time]
        n_trials = len(trials_in_block)
        
        if 'probabilities l/r' in block:
            prob_lr = block.get('probabilities l/r', [None, None])
            block_hover = f"""<b>Block {j}: {block.get('block', 'Unknown')}</b><br>
<br><b>Block Start:</b><br>
Time: {block_time}<br>
<br><b>Trials in Block:</b><br>
Number of trials: {n_trials}<br>
<br><b>Reward Probabilities:</b><br>
Left: {prob_lr[0]:.3f}<br>
Right: {prob_lr[1]:.3f}<br>
<br><b>Transition Criteria:</b><br>
Performance: {block.get('block transition perf criteria', 'N/A')}"""
            
            for row in [1, 2, 3]:
                fig.add_trace(go.Scatter(
                    x=[block_time],
                    y=[0.5],
                    mode='markers',
                    marker=dict(size=20, color='green', opacity=0, line=dict(width=0)),
                    showlegend=False,
                    hovertemplate=block_hover + '<extra></extra>',
                    legendgroup='blocks'
                ), row=row, col=1)
            
            for row in [1, 2, 3]:
                fig.add_vline(x=block_time, line=dict(color='green', dash='dash', width=2), opacity=0.5, row=row, col=1)
            
        else:
            block_hover = f"""<b>Block {j}: {block.get('block', 'Unknown')}</b><br>
<br><b>Block Start:</b><br>
Time: {block_time}<br>
<br><b>Trials in Block:</b><br>
Number of trials: {n_trials}<br>
<br>No reward probabilities (non-task block)"""
            
            for row in [1, 2, 3]:
                fig.add_trace(go.Scatter(
                    x=[block_time],
                    y=[0.5],
                    mode='markers',
                    marker=dict(size=20, color='orange', opacity=0, line=dict(width=0)),
                    showlegend=False,
                    hovertemplate=block_hover + '<extra></extra>',
                    legendgroup='blocks'
                ), row=row, col=1)
            
            for row in [1, 2, 3]:
                fig.add_vline(x=block_time, line=dict(color='orange', dash='dash', width=2), opacity=0.5, row=row, col=1)

    # ========================================================================
    # Update layout
    # ========================================================================
    fig.update_xaxes(title_text="Time (samples)", matches='x', row=3, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', matches='x', row=1, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', matches='x', row=2, col=1)

    # Plot 1 (Trials)
    fig.update_yaxes(
        title_text="Choice",
        tickmode='array',
        tickvals=[0, 0.5, 1],
        ticktext=['Right', 'No Choice', 'Left'],
        range=[-0.1, 1.1],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=1, col=1
    )

    # Plot 2 (States) 
    fig.update_yaxes(
        title_text="States",
        tickmode='array',
        tickvals=[0, 0.5, 1],
        ticktext=['', 'Behavioral State', ''],
        range=[-0.1, 1.1],
        showgrid=False,
        row=2, col=1
    )

    # Plot 3 (Choice rate)
    fig.update_yaxes(
        title_text="Choice Rate",
        tickmode='array',
        tickvals=[0, 0.5, 1],
        ticktext=['Right', 'Equal', 'Left'],
        range=[0, 1],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=3, col=1
    )

    fig.update_layout(
        title=f'Session Overview: {len(trials)} Trials, {len(states)} States',
        width=1400,
        height=1000,  # Increased height for 3 plots
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Write to HTML with custom JavaScript (updated for 3 plots)
    html_string = fig.to_html(include_plotlyjs='cdn')

    # Insert custom JavaScript before </body> (updated trace indices for 3 plots)
    custom_js = """
<script>
var plot = document.getElementsByClassName('plotly-graph-div')[0];
var lastSelectedTrial = null;

plot.on('plotly_click', function(data){
    var point = data.points[0];
    
    // Check if click is from a trial point (has customdata)
    if (point.data.customdata && point.data.customdata[point.pointIndex]) {
        var customData = point.data.customdata[point.pointIndex];
        var trialIdx = customData[0];
        var start = customData[1];
        var holdStart = customData[2];
        var tChoosing = customData[3];
        var tChosen = customData[4];
        var choice = customData[5];
        var rewarded = customData[6];
        
        // If clicking the same trial, remove markers
        if (lastSelectedTrial === trialIdx) {
            // Clear connecting line
            var lineUpdate = {
                x: [[]],
                y: [[]]
            };
            var numTrials = """ + str(len(trials)) + """;
            Plotly.restyle(plot, lineUpdate, [numTrials]);
            
            // Clear all timing markers
            var update = {
                x: [[], [], [], []],
                y: [[], [], [], []],
                text: [[], [], [], []],
                'marker.color': [[], [], [], []],
                'marker.symbol': [[], [], [], []]
            };
            
            var markerIndices = [numTrials+1, numTrials+2, numTrials+3, numTrials+4];
            Plotly.restyle(plot, update, markerIndices);
            
            lastSelectedTrial = null;
            return;
        }
        
        lastSelectedTrial = trialIdx;
        
        // Determine y position based on choice
        var yPos = (choice === 'r') ? 0 : 1;
        
        // Update connecting line first
        var lineUpdate = {
            x: [[start, holdStart, tChoosing, tChosen]],
            y: [[yPos, yPos, yPos, yPos]]
        };
        var numTrials = """ + str(len(trials)) + """;
        Plotly.restyle(plot, lineUpdate, [numTrials]);
        
        // Prepare timing markers data
        var timingData = [
            {x: [start], y: [yPos], text: ['Start'], color: 'green', symbol: 'square'},
            {x: [holdStart], y: [yPos], text: ['Hold'], color: 'orange', symbol: 'diamond'},
            {x: [tChoosing], y: [yPos], text: ['Go!'], color: 'cyan', symbol: 'star'},
            {x: [tChosen], y: [yPos], text: ['Done'], color: 'magenta', symbol: 'x'}
        ];
        
        // Update each timing marker trace
        for (var i = 0; i < 4; i++) {
            var update = {
                x: [timingData[i].x],
                y: [timingData[i].y],
                text: [timingData[i].text],
                'marker.color': [[timingData[i].color]],
                'marker.symbol': [[timingData[i].symbol]]
            };
            Plotly.restyle(plot, update, [numTrials + 1 + i]);
        }
        
        // Zoom to trial region
        var margin = (tChosen - start) * 1.5;
        var layoutUpdate = {
            'xaxis.range': [start - margin, tChosen + margin]
        };
        Plotly.relayout(plot, layoutUpdate);
    }
});

// Double-click to reset zoom
plot.on('plotly_doubleclick', function(){
    Plotly.relayout(plot, {
        'xaxis.autorange': true,
        'xaxis2.autorange': true,
        'xaxis3.autorange': true
    });
    
    var numTrials = """ + str(len(trials)) + """;
    
    // Clear connecting line
    var lineUpdate = {
        x: [[]],
        y: [[]]
    };
    Plotly.restyle(plot, lineUpdate, [numTrials]);
    
    // Clear timing markers
    var update = {
        x: [[], [], [], []],
        y: [[], [], [], []],
        text: [[], [], [], []],
        'marker.color': [[], [], [], []],
        'marker.symbol': [[], [], [], []]
    };
    var markerIndices = [numTrials+1, numTrials+2, numTrials+3, numTrials+4];
    Plotly.restyle(plot, update, markerIndices);
    
    lastSelectedTrial = null;
});
</script>
"""

    # Insert the custom JS before </body>
    html_string = html_string.replace('</body>', custom_js + '</body>')

    # Save to file
    output_path = os.path.join(data_session_path, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_string)

    print(f"\n✓ Interactive plot saved to '{output_filename}'")
    print("Instructions:")
    print("  - Click on any trial to see timing markers with connecting line and zoom")
    print("  - Click the same trial again to hide markers")
    print("  - Double-click anywhere to reset zoom")
    print("  - States are visualized as colored rectangles in the middle plot")
    print("  - Hover over states to see detailed information")

    # Calculate and print summary statistics
    valid_trials = [t for t in trials if 'choice' in t and 'rewarded' in t]
    choices_summary = [0 if t['choice'] == 'r' else 1 for t in valid_trials]
    rewards = [1 if t['rewarded'] else 0 for t in valid_trials]

    print("\n" + "="*50)
    print("SESSION SUMMARY")
    print("="*50)
    print(f"Total trials: {len(trials)}")
    print(f"Trials with no choice: {len(missing_choice)}")
    print(f"Valid trials with choice: {len(valid_trials)}")
    print(f"Right choices: {choices_summary.count(0)} ({choices_summary.count(0)/len(choices_summary)*100:.1f}%)")
    print(f"Left choices: {choices_summary.count(1)} ({choices_summary.count(1)/len(choices_summary)*100:.1f}%)")
    print(f"Rewarded trials: {rewards.count(1)} ({rewards.count(1)/len(rewards)*100:.1f}%)")
    print(f"Unrewarded trials: {rewards.count(0)} ({rewards.count(0)/len(rewards)*100:.1f}%)")
    print(f"Total states: {len(states)}")
    if unique_states:
        print(f"Unique states: {', '.join(unique_states)}")
    
    # Return summary statistics as a dictionary
    summary = {
        'total_trials': len(trials),
        'missing_choice': len(missing_choice),
        'valid_trials': len(valid_trials),
        'right_choices': choices_summary.count(0),
        'left_choices': choices_summary.count(1),
        'rewarded_trials': rewards.count(1),
        'unrewarded_trials': rewards.count(0),
        'total_states': len(states),
        'unique_states': unique_states
    }
    
    return summary


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description='Generate interactive session visualization for Two-Arm Bandit task data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_session_path',
        type=str,
        help='Path to the session directory containing metrics.json'
    )
    
    parser.add_argument(
        '--output_filename',
        type=str,
        default='interactive_session_plot.html',
        help='Name of the output HTML file'
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=10,
        help='Window size for moving average calculation'
    )
    
    args = parser.parse_args()
    
    # Generate the interactive plot
    summary = generate_interactive_session_plot(
        data_session_path=args.data_session_path,
        output_filename=args.output_filename,
        window=args.window
    )
    
    return summary


if __name__ == '__main__':
    main()
