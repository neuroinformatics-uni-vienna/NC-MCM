import os
import matplotlib.pyplot as plt
import numpy as np
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
                             interactive=False,
                             weights_hist=False,
                             bins=15,
                             test=False):
    """
    Creates a behavioral state diagram using the defined states (C and B) as a directed graph.
    Can also show some diagnostic/informative plots with the parameters "adj_matrix" or "weight_hist".
    The "interactive" parameter will create an HTML-plot using "pyvis".

    Parameters:

        C (np.array, list): Defines the cognitive states timeseries.

        B (np.array, list): Defines the behavior timeseries.

        behaviors (np.array, list): Names for elements in B, indexed by their value (e.g. name of B=1 is at behaviors[1])

        threshold (float): A threshold which is used to display edges in the graph (smaller values are not plotted)

        offset (float): Distance between clusters
        bins (int): Amount of bins in histogram if "weights_hist"=True
        interactive (bool): If the HTML-plot should be created, otherwise the "matplotlib" plot is shown
        adj_matrix (bool): If the adjacency matrix should be plotted
        weights_hist (bool): If a histogram of transition weights should be plotted

    Returns:
        Boolean success indicator
    """

    if behaviors is not None:
        if type(B[0]) is int:
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
    if interactive:

        if adj_matrix:
            fig, ax = plt.subplots()
            im = ax.imshow(T, cmap='Reds', interpolation='nearest', vmin=0, vmax=0.03)
            ax.set_title('Adjacency Matrix Heatmap')
            plt.colorbar(im, ax=ax)
            ax.set_yticks(np.arange(T.shape[0]), G.nodes)
            ax.set_xlabel('Nodes')
            ax.set_ylabel('Nodes')
            plt.show(block=False)

        net = Network(directed=True)
        net.from_nx(G)
        for idx, node in enumerate(net.nodes):
            c, b = node['id'].split(':')
            c_int = int(c[1:]) - 1
            b_int = np.where(np.asarray(trans_B) == b)[0][0]
            n_idx = (len(behaviors) * c_int + b_int)
            r, g, b = colordict[b_int]
            node['color'] = f'rgb({r * 255},{g * 255},{b * 255})'
            node['size'] = np.sqrt(node_sizes[n_idx])
            new = {name: int(T[n_idx, i] * (len(B) - 1)) for i, name in enumerate(G.nodes)}
            node['title'] = ''.join(f'{k}:{v}\n' for k, v in new.items() if v > 0)

        net.show_buttons(['physics', 'nodes', 'edges'])
        name = str(input('File name for the html-plot? '))
        net.show(f'{name}.html', notebook=False)
        print(f'Plot has been saved under: {os.getcwd()}/{name}.html')

    else:

        if adj_matrix:
            fig, ax = plt.subplots(1, 2)
            ax_a = ax[0]
            ax_g = ax[1]
            im_a = ax_a.imshow(T, cmap='Reds', interpolation='nearest', vmin=0, vmax=0.03)
            ax_a.set_title('Adjacency Matrix Heatmap')
            plt.colorbar(im_a, ax=ax_a)
            ax_a.set_yticks(np.arange(T.shape[0]), G.nodes)
            ax_a.set_xlabel('Nodes')
            ax_a.set_ylabel('Nodes')
        else:
            fig, ax_g = plt.subplots()

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
                            stationary=False):
    """
       Clusters neuronal activity into cognitive clusters in probability space. The cluster sequences are tested for
       Markov properties and are returned in order of likelihood of originating from a 1st order Markov Process.

       Parameters:

           N (np.array, list): Neuronal activity timeseries (shape = (neurons, activity-timeseries))

           B (np.array, list): Behavioral timeseries data

           n_clusters (int): Amount of clusters to be tested.

           nrep (int): Amount of sequences to be clustered and tested.

           model: A classification model with the ability to predict a probability.

           ensemble (bool): If an ensemble should be created for the classifier.

           sim_m (int): Amount of generated sequences in the markovian() method.
           sim_s (int): Amount of generated sequences in the stationary() method.
           chunks (int): Amount of chunks used in the stationary() method.
           clustering: Type of clustering used ('kmeans' or 'spectral')
           kmeans_init: Value for 'n_init' in KMeans (default: 'auto').
           stationary (bool): Amount of chunks used in the stationary() method.

       Returns:

           A numpy array of cognitive state sequences (amount='n_rep') sorted by likelihood of stemming from a 1st order
           Markov Process and the p-value of the markovian() (and stationary()) -method(s).
       """

    if type(B[0]) is not int:
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

    for reps in range(nrep):
        print(f'Testing markovianity for {n_clusters} clusters - repetition {reps + 1}')
        _ = clustering_trajectories(yp_map, n_clusters, kmeans_init, clustering, chunks, sim_m, sim_s, stationary)
        res.append(_)

    if stationary:
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

      yp_map (np.array): Behavioral probability timeseires

      n_clusters (int): Amount of clusters to be tested.

      kmeans_init: Value for 'n_init' in KMeans (default: 'auto').

      clustering (str): Type of clustering used ('kmeans' or 'spectral')

      chunks (int): Specifies the amount of chunks if stationary property is tested.

      sim_m (int): Amount of generated sequences in the markovian() method.

      sim_s (int): Amount of generated sequences in the stationary() method.

      stationary (bool): Amount of chunks used in the stationary() method.

    Returns:
      A numpy array of cognitive state sequences and the p-value(s) of markovian() (and stationary()).
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
        _, p_adj_s = stationarity(xctmp, chunks=chunks, plot=False, sim_stationary=sim_s)
        return xctmp, p_m, p_adj_s

    return xctmp, p_m
