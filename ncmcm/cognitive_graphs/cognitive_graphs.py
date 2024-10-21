"""
@authors:
Michael Hofer
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx
from pyvis.network import Network
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.linear_model import LogisticRegression
from ncmcm.cognitive_graphs.calculations import adj_matrix_ncmcm, fit_model
from ncmcm.cognitive_graphs.helpers import shift_pos_by, generate_equidistant_colors, map_names, make_integer_list
from ncmcm.cognitive_graphs.custom_models import CustomEnsembleModel
from ncmcm.statistical_testing.markov import markovian, stationarity


def behavioral_state_diagram(C,
                             B,
                             behaviors=None,
                             offset=2.5,
                             threshold=None,
                             adj_matrix=False,
                             interactive=None,
                             options=0,
                             weights_hist=False,
                             bins=15,
                             test=False,
                             **kwargs):
    """
    Creates a behavioral state diagram using the defined states (C and B) as a directed graph.
    Can also show some diagnostic/informative plots with the parameters "adj_matrix" or "weight_hist".
    The "interactive" parameter will create an HTML-plot using "pyvis".

    Parameters:

        C: np.ndarray, required
            Defines the cognitive states timeseries.

        B: np.ndarray, required
            Defines the behavior timeseries.

        behaviors: np.ndarray, optional
            Names for elements in B, indexed by their value (e.g. name of B=1 is at behaviors[1])

        threshold: float, optional
            A threshold which is used to display edges in the graph (smaller values are not plotted)

        offset: float, optional
            Distance between clusters

        bins: int, optional
            Amount of bins in histogram if "weights_hist"=True

        interactive: str, optional
            If the HTML-plot should be created, one defines the filename with path here

        options: int, str, optional
            This gives either an int from 0 to 2 for a predefined physics script or one can give a path to
            a JSON-file containing a physics script for a pyvis graph.
                0 will push nodes apart and pull them together by the edges.
                1 will remove all forces acting on the nodes so one can place them by hand.
                2 removes some of the strength of the forces in 0 to make it easier to place nodes.

        adj_matrix: bool, optional
            If the adjacency matrix should be plotted

        weights_hist: bool, optional
            If a histogram of transition weights should be plotted

    Returns:
        Boolean success indicator
    """

    if behaviors is not None:
        if type(B[0]) in (int, np.int32, np.int64):
            trans_B = behaviors
        else:
            B, _ = make_integer_list(B)
            trans_B = behaviors
    else:
        B, trans_B = make_integer_list(B)

    cognitive_states = np.unique(C)
    behaviors = np.unique(B)
    colordict = dict(zip(behaviors, generate_equidistant_colors(len(behaviors))))
    node_colors = list(colordict.values()) * len(cognitive_states)
    T, C_B_states = adj_matrix_ncmcm(C=C, B=B)

    # Create Matrix for drawing by removing diagonals and edges below threshold
    T_edges = T.copy()
    T_edges[np.diag_indices_from(T_edges)] = 0
    if threshold is None:
        threshold = np.max(T_edges) / 10
        print('Calculated threshold is: ', threshold)
    T_edges[T_edges < threshold] = 0

    # Plot transition distribution if wanted
    if weights_hist:
        tmp = T_edges.copy()
        tmp[tmp == 0] = np.nan
        plt.hist(tmp.reshape(-1, 1), bins=bins)
        plt.title(f'Distribution of edges after removing ones with weight below {np.round(threshold, 5)}')
        plt.ylabel('amount of edges')
        plt.xlabel('edge weights before scaling')
        plt.show(block=False)

    # Create the graph
    G_old = nx.DiGraph()
    G_old.add_nodes_from(C_B_states)
    T_edges = T_edges / (np.max(T_edges) / 10)
    nx.from_numpy_array(T_edges, create_using=G_old)
    edge_colors = [node_colors[u] for u, v in G_old.edges()]
    node_sizes = (np.diag(T) / np.max(np.diag(T)) * 250) * (np.sqrt(T.shape[0]) / offset)
    mapping = {node: map_names(trans_B, str(C_B_states[node])) for node in G_old.nodes()}
    G = nx.relabel_nodes(G_old, mapping)

    # Reposition Nodes according to subgroups
    cog_groups = []
    for c_num in range(len(cognitive_states)):
        cog_groups.append([n for n in np.unique(G.nodes) if n.split(':')[0] == 'C' + str(c_num + 1)])
    all_pos = []
    for c_node_group in cog_groups:
        all_pos.append(nx.circular_layout(G.subgraph(c_node_group)))
    adjusted_pos = {}
    degrees_list = np.linspace(0, 360, num=len(cognitive_states), endpoint=False)
    for idx, current_pos in enumerate(all_pos):
        adjusted_pos = shift_pos_by(current_pos, adjusted_pos, degrees_list[idx], offset)

    # Plot graphs
    if interactive is not None:

        if adj_matrix:
            fig, ax = plt.subplots(**kwargs)
            im = ax.imshow(T, cmap='Reds', interpolation='nearest', vmin=0, vmax=0.03)
            ax.set_title('Adjacency Matrix Heatmap')
            plt.colorbar(im, ax=ax)
            ax.set_yticks(np.arange(T.shape[0]), G.nodes)
            ax.set_xlabel('Nodes')
            ax.set_ylabel('Nodes')
            plt.show(block=False)

        net = Network(directed=True, filter_menu=True, select_menu=True, cdn_resources='remote')
        net.from_nx(G)
        for idx, node in enumerate(net.nodes):
            c, b = node['id'].split(':')
            node['cog_state'] = c
            node['behavior'] = b
            c_int = int(c[1:]) - 1
            b_int = np.where(np.asarray(trans_B) == b)[0][0]
            n_idx = (len(behaviors) * c_int + b_int)
            r, g, b = colordict[b_int]
            node['color'] = f'rgb({r * 255},{g * 255},{b * 255})'
            node['size'] = np.sqrt(node_sizes[n_idx])
            new = {name: int(T[n_idx, i] * (len(B) - 1)) for i, name in enumerate(G.nodes)}
            node['title'] = ''.join(f'{k}:{v}\n' for k, v in new.items() if v > 0)

        if type(options) is int:
            script_dir = os.path.dirname(os.path.abspath(__file__))

            if options not in [0, 1, 2]:
                print(ValueError(f"Option '{options}' not found in the options file."))

            with open(os.path.join(script_dir, "json_physics", "options.json"), 'r') as file:
                options_dict = json.load(file)
            physics = options_dict[str(options)]

        elif type(options) is str:
            with open(options, 'r') as file:
                physics = json.load(file)

        else:
            print('ERROR! No valid physics script selected.')
            return None
        physics = json.dumps(physics, indent=2)
        net.set_options(physics)

        net.show(f'{interactive}.html', notebook=False)
        print(f'Plot has been saved under: {interactive}.html')

    else:

        if adj_matrix:
            fig, ax = plt.subplots(1, 2, **kwargs)
            ax_a = ax[0]
            ax_g = ax[1]
            im_a = ax_a.imshow(T, cmap='Reds', interpolation='nearest', vmin=0, vmax=0.03)
            ax_a.set_title('Adjacency Matrix Heatmap')
            plt.colorbar(im_a, ax=ax_a)
            ax_a.set_yticks(np.arange(T.shape[0]), G.nodes)
            ax_a.set_xlabel('Nodes')
            ax_a.set_ylabel('Nodes')
        else:
            fig, ax_g = plt.subplots(**kwargs)

        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]

        nx.draw(G, adjusted_pos,
                with_labels=True,
                connectionstyle="arc3,rad=-0.2",
                node_color=node_colors,
                node_size=node_sizes,
                width=weights,
                arrows=True,
                arrowsize=10,
                edge_color=edge_colors,
                ax=ax_g)
        plt.title("Behavioral State Diagram")
        plt.show(block=False)

    if test:
        plt.close('all')
        return True
    else:
        plt.show()
        return True


def cluster_neural_activity(N,
                            B,
                            n_clusters,
                            nrep=10,
                            model=None,
                            ensemble=True,
                            sim_m=500,
                            sim_s=500,
                            chunks=None,
                            clustering='kmeans',
                            kmeans_init='auto',
                            stationary=False,
                            plot=True):
    """
       Clusters neuronal activity into cognitive clusters in probability space. The cluster sequences are tested for
       Markov properties and are returned in order of likelihood of originating from a 1st order Markov Process.

       Parameters:

           N: np.ndarray, required
                Neuronal activity timeseries (shape = (neurons, activity-timeseries))

           B: np.ndarray, required
                Behavioral timeseries data

           n_clusters: int, required
                Amount of clusters to be tested.

           nrep: int, optional
                Amount of sequences to be clustered and tested.

           model: model, optional
                A classification model with the ability to predict a probability.
                As a default a LogisticRegression is used.

           ensemble: bool, optional
                If an ensemble should be created for the classifier.

           sim_m: int, optional
                Amount of generated sequences in the markovian() method.

           sim_s: int, optional
                Amount of generated sequences in the stationary() method.

           chunks: int, optional
                Amount of chunks used in the stationary() method.

           clustering: str, optional
                Type of clustering used ('kmeans' or 'spectral')

           kmeans_init: str, optional
                Value for 'n_init' in KMeans (default: 'auto').

           stationary:bool, optional
                Amount of chunks used in the stationary() method.

       Returns:

           res: list
                A numpy array of cognitive state sequences (amount='n_rep') sorted by likelihood of stemming from a
                1st order Markov Process and the p-value of the markovian (and stationary) -method(s).
       """

    if type(B[0]) not in (int, np.int32, np.int64):
        B, trans_b = make_integer_list(B)
        print(f'Behaviors \'B\' were transformed into integers.\nThis is the translation: {trans_b}')

    if model is None:
        model = LogisticRegression()
    if ensemble:
        model = CustomEnsembleModel(model)

    yp_map, _ = fit_model(N,
                          B,
                          base_model=model)

    res = []
    p_vals = []

    for reps in range(nrep):
        print(f'Testing markovianity for {n_clusters} clusters - repetition {reps + 1}')
        _ = clustering_trajectories(yp_map, n_clusters, kmeans_init, clustering, chunks, sim_m, sim_s, stationary)
        res.append(_)
        if plot:
            p_vals.append(_[1:])

    if plot:
        p_vals = np.array(p_vals)

        # Check if the shape is (nrep, 2) or (nrep, 4)
        if p_vals.shape[1] == 2:
            data_to_plot = p_vals[:, 0]  # Only the second column
            fig, ax = plt.subplots()
            ax.boxplot(data_to_plot)
            #sns.boxplot(data=data_to_plot, ax=ax)
            ax.set_xticklabels(['Markov property'])

        else:
            # For (nrep, 4), ignore the first column and plot the rest
            data_to_plot = p_vals[:, :]  # Second, third, and fourth columns
            fig, ax = plt.subplots()
            ax.boxplot(data_to_plot)
            #sns.boxplot(data=data_to_plot, palette=["lightblue", "lightgreen", "lightgreen"], ax=ax)
            ax.set_xticklabels(['Markov property', 'Stationary property KS-test', 'Stationary property T-test'],
                               rotation=45)
        ax.axhline(0.05, linestyle='--', color='red')
        ax.fill_between(ax.get_xlim(), y1=0.05, y2=1, color='green', alpha=0.3)
        ax.fill_between(ax.get_xlim(), y1=0, y2=0.05, color='red', alpha=0.3)

        # Adding placeholders for title and axis labels
        ax.set_title(f'P-values for {nrep} clustered cognitive sequences ')
        ax.set_ylabel('P-value results')
        plt.show()

    if stationary:
        res = sorted(res, key=lambda x: x[3])
        res = sorted(res, key=lambda x: x[2])
    res_sorted = sorted(res, key=lambda x: x[1], reverse=True)

    return res_sorted


def clustering_trajectories(yp_map,
                            n_clusters,
                            kmeans_init='auto',
                            clustering='kmeans',
                            chunks=None,
                            sim_m=500,
                            sim_s=500,
                            stationary=False):
    """
    Clusters neuronal activity into cognitive clusters in probability space and tests them for 1st order
    Markov properties. Will return the sequence of cognitive clusters and the p-value(s) ("stationary"
    will indicate to test if the sequence comes from a stationary process).

    Parameters:

        yp_map: np.ndarray, required
            Behavioral probability timeseries

        n_clusters: int, required
            Amount of clusters to be tested.

        kmeans_init: str, optional
            Value for 'n_init' in KMeans (default: 'auto').

        clustering: str, optional
            Type of clustering used ('kmeans' or 'spectral')

        chunks: int, optional
            Specifies the amount of chunks if stationary property is tested.

        sim_m: int, optional
            Amount of generated sequences in the markovian() method.

        sim_s: int, optional
            Amount of generated sequences in the stationary() method.

        stationary: bool, optional
            Amount of chunks used in the stationary() method.

    Returns:

        xctmp: np.ndarray
            A numpy array of cognitive state sequences

        p_m: np.ndarray
            The p-value given by the "markovian" method

        p_ks: np.ndarray
              The p-value given by the "stationary" method for the ks-test

        p_tt: np.ndarray
              The p-value given by the "stationary" method for the t-test
    """
    # Clustering in probability space
    if clustering == 'kmeans':
        clusters = KMeans(n_clusters=n_clusters, n_init=kmeans_init).fit(yp_map)
        xctmp = clusters.labels_
    elif clustering == 'spectral':
        clusters = SpectralClustering(n_clusters=n_clusters).fit(yp_map)
        xctmp = clusters.row_labels_
    else:
        raise ValueError("Invalid value for 'clustering' parameter. "
                         "It should be either 'kmeans' or 'spectral'. ")

    # Statistical testing
    p_m, _ = markovian(xctmp, sim_memoryless=sim_m)
    if stationary:
        p_ks, _, p_t, _ = stationarity(xctmp, chunks=chunks, plot=False, sim_stationary=sim_s)
        return xctmp, p_m, p_ks, p_t

    return xctmp, p_m
