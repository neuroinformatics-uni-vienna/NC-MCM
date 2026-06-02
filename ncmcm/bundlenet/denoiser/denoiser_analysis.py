import numpy as np
import matplotlib.pyplot as plt

def plot_top_neurons(denoised_states, behaviors, database, id, path, num_neurons=1):
    """Given the denoised neuronal time series, identifies the top n neurons contributing to each behavior
    and plots their contributions."""

    # for each time step, identify the top n neurons contributing to the denoised state
    top_neurons = np.argsort(np.abs(denoised_states), axis=1)[:, -num_neurons:]
    top_neurons = top_neurons[:,::-1]

    top_neurons_names = database.neuron_names[top_neurons]
    top_neurons_values = np.take_along_axis(denoised_states, top_neurons, axis=1)

    # Build a categorical mapping so present neurons are evenly spaced on y-axis
    unique_neurons = np.unique(top_neurons)
    unique_neurons.sort()
    neuron_to_pos = {neuron: pos for pos, neuron in enumerate(unique_neurons)}
    top_neurons_pos = np.vectorize(neuron_to_pos.get)(top_neurons)
    unique_behaviors = np.unique(behaviors)
    cmap = plt.cm.get_cmap('tab20', len(unique_behaviors))
    behavior_to_color = {b: cmap(i) for i, b in enumerate(unique_behaviors)}

    # One large figure with multiple subplots: one per top-neuron trace plus a combined panel.
    n_subplots = num_neurons + 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=(15, 5 * n_subplots), squeeze=False)

    behavior_handles = [
        plt.Rectangle((0, 0), 1, 1, color=behavior_to_color[b], alpha=0.25, label=f'Behavior {b}')
        for b in unique_behaviors
    ]

    def add_behavior_background(ax):
        start = 0
        for t in range(1, len(behaviors) + 1):
            if t == len(behaviors) or behaviors[t] != behaviors[start]:
                ax.axvspan(start, t, color=behavior_to_color[behaviors[start]], alpha=0.15, lw=0)
                start = t

    for i, ax in enumerate(axes[:num_neurons, 0]):
        add_behavior_background(ax)
        ax.plot(top_neurons_pos[:, i], label=f"Top {i+1} neuron", linewidth=1.5)
        ax.set_ylabel('Denoised activity')
        ax.set_yticks(np.arange(len(unique_neurons)))
        ax.set_yticklabels(database.neuron_names[unique_neurons])
        ax.set_title(f'Top {i+1} neuron contributions with behavior background')
        neuron_legend = ax.legend(loc='upper left', fontsize=9)
        ax.add_artist(neuron_legend)
        ax.legend(handles=behavior_handles, loc='upper right', fontsize=9)

    ax = axes[num_neurons, 0]
    add_behavior_background(ax)
    for i in range(num_neurons):
        ax.plot(top_neurons_pos[:, i], label=f"Top {i+1} neuron", linewidth=1.2)

    ax.set_xlabel('Time')
    ax.set_ylabel('Denoised activity')
    ax.set_yticks(np.arange(len(unique_neurons)))
    ax.set_yticklabels(database.neuron_names[unique_neurons])
    ax.set_title('Top neuron contributions with behavior background (all top-n)')
    neuron_legend = ax.legend(loc='upper left', fontsize=9)
    ax.add_artist(neuron_legend)
    ax.legend(handles=behavior_handles, loc='upper right', fontsize=9)

    fig.tight_layout()

    if path is not None:
        fig.savefig(f"{path}/top_{num_neurons}_neurons_{id}_all.pdf", dpi=200)

    plt.close(fig)

    return top_neurons, top_neurons_names, top_neurons_values

def plot_top_neurons_hists(denoised_states, behaviors, database, id, path, num_neurons=1):
    """Plots the percentage of time each neuron is in the top n contributing neurons for each behavior as a bar plot."""
    unique_behaviors = database.behaviour_names

    plots = []
    for (behavior_id, behavior_label) in unique_behaviors.items():
        behavior_mask = behaviors == behavior_id
        behavior_states = denoised_states[behavior_mask]

        if behavior_states.size == 0:
            continue

        plots.append((behavior_label, behavior_states))

    if not plots:
        return

    fig, axes = plt.subplots(len(plots), 1, figsize=(20, 6 * len(plots)), squeeze=False)


    for ax, (behavior_label, behavior_states) in zip(axes[:, 0], plots):
        top_neurons = np.argsort(np.abs(behavior_states), axis=1)[:, -num_neurons:][:, ::-1]
        count_per_neuron = np.array([
            np.any(top_neurons == neuron, axis=1)
            for neuron in range(denoised_states.shape[1])
        ])

        ax.bar(
            np.arange(len(count_per_neuron)),
            count_per_neuron.sum(axis=1) / len(behavior_states),
            label=f'Behavior {behavior_label}'
        )
        ax.set_xlabel('Neuron')
        ax.set_ylabel(f'Percentage time in top {num_neurons} neurons')
        ax.set_title(f'Neurons vs Time in top {num_neurons} neurons ({behavior_label})')
        ax.set_xticks(np.arange(denoised_states.shape[1]))
        ax.set_xticklabels(database.neuron_names[:denoised_states.shape[1]], rotation=90)
        ax.legend()

    fig.tight_layout()
    fig.savefig(f"{path}/top_{num_neurons}_neurons_hist_{id}_all_behaviors.pdf", dpi=200)
    plt.close(fig)


def plot_top_neurons_hists_stacked(denoised_states, behaviors, database, id, path, num_neurons=1):
    """Plots stacked bars of the probability for each neuron to be exactly the i-th top contributor for each behavior."""
    unique_behaviors = database.behaviour_names

    plots = []
    for behavior_id, behavior_label in unique_behaviors.items():
        behavior_mask = behaviors == behavior_id
        behavior_states = denoised_states[behavior_mask]

        if behavior_states.size == 0:
            continue

        plots.append((behavior_label, behavior_states))

    if not plots:
        return

    n_neurons = denoised_states.shape[1]
    fig, axes = plt.subplots(len(plots), 1, figsize=(20, 6 * len(plots)), squeeze=False)

    for ax, (behavior_label, behavior_states) in zip(axes[:, 0], plots):
        top_neurons = np.argsort(np.abs(behavior_states), axis=1)[:, -num_neurons:][:, ::-1]

        x = np.arange(n_neurons)
        bottom = np.zeros(n_neurons, dtype=float)

        # probs_per_rank[i, j] = P(neuron j is exactly rank i+1 top neuron | behavior)
        probs_per_rank = np.zeros((num_neurons, n_neurons), dtype=float)
        for i in range(num_neurons):
            rank_i = top_neurons[:, i]
            counts = np.bincount(rank_i, minlength=n_neurons)
            probs_per_rank[i] = counts / len(behavior_states)

        for i in range(num_neurons):
            ax.bar(
                x,
                probs_per_rank[i],
                bottom=bottom,
                label=f'Top {i+1}'
            )
            bottom += probs_per_rank[i]

        ax.set_xlabel('Neuron')
        ax.set_ylabel('Probability')
        ax.set_title(
            f'Neuron probability by exact top rank (1..{num_neurons}) for behavior {behavior_label}'
        )

        non_null_neurons = np.where(bottom > 0)[0]
        ax.set_xticks(non_null_neurons)
        ax.set_xticklabels(database.neuron_names[non_null_neurons], rotation=90)
        ax.legend()

    fig.tight_layout()
    fig.savefig(f"{path}/top_{num_neurons}_neurons_hist_stacked_{id}_all_behaviors.pdf", dpi=200)
    plt.close(fig)


    
