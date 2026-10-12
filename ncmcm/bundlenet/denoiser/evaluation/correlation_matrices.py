import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list


from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich import box

from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn,
)


class CorrelationMatrixAnalysis:
    """
    Compare the neuron-neuron correlation structure of original vs. denoised
    neuronal states.

    With discrete behavior (``behavior_type="discrete"``) every analysis is
    computed separately for each behavior. With continuous behavior
    (``behavior_type="continuous"``) samples cannot be grouped by behavior,
    so every analysis is computed once on the whole dataset; results keep
    the same ``{key: value}`` structure, with ``FULL_DATASET_KEY`` as the
    only key.
    """

    FULL_DATASET_KEY = "full dataset"

    def __init__(self, X_original, X_denoised, behaviors, behavior_type="discrete"):
        self.X_original = X_original
        self.X_denoised = X_denoised
        self.behaviors = behaviors
        self.behavior_type = behavior_type

        if behavior_type == "continuous":
            self.unique_behaviors = [self.FULL_DATASET_KEY]
        else:
            self.unique_behaviors = np.unique(behaviors)

        self.effective_ranks_original = {beh: [] for beh in self.unique_behaviors}
        self.effective_ranks_denoised = {beh: [] for beh in self.unique_behaviors}

    def _mask(self, beh):
        """
        Boolean mask of the samples belonging to ``beh``: every sample for
        continuous behavior (a single group, the whole dataset), otherwise
        the samples labeled ``beh``.
        """
        if self.behavior_type == "continuous":
            if beh != self.FULL_DATASET_KEY:
                raise ValueError(
                    "With behavior_type='continuous' the only group is the full dataset: "
                    f"pass behavior=None or behavior='{self.FULL_DATASET_KEY}', got {beh!r}."
                )
            return np.ones(len(self.X_original), dtype=bool)
        return self.behaviors == beh

    def _compute_correlation_matrix(self, X):
        """
        Compute the correlation matrix of the input data X.

        Parameters:
        - X: np.ndarray, input data (time x neurons).

        Returns:
        - corr_matrix: np.ndarray, correlation matrix (neurons x neurons).
        """
        return np.corrcoef(X.T)

    def _compute_effective_rank(self, X):
        corr_matrix = self._compute_correlation_matrix(X)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.clip(eigenvalues, a_min=1e-10, a_max=None)  # Avoid log(0)
        normalized_eigenvalues = eigenvalues / np.sum(eigenvalues)
        effective_rank = np.exp(-np.sum(normalized_eigenvalues * np.log(normalized_eigenvalues)))
        return effective_rank

    def effective_ranks(self, behavior=None):
        """
        Compute the effective rank of a matrix using the formula:
        effective_rank = exp(-sum(p_i * log(p_i))) where p_i are the normalized singular values.

        Computed per behavior (discrete behavior), or once on the whole
        dataset under ``FULL_DATASET_KEY`` (continuous behavior).
        """

        if behavior is None:
            for beh in self.unique_behaviors:
                mask = self._mask(beh)
                self.effective_ranks_original[beh] = self._compute_effective_rank(self.X_original[mask])
                self.effective_ranks_denoised[beh] = self._compute_effective_rank(self.X_denoised[mask])
        else:
            mask = self._mask(behavior)
            self.effective_ranks_original[behavior] = self._compute_effective_rank(self.X_original[mask])
            self.effective_ranks_denoised[behavior] = self._compute_effective_rank(self.X_denoised[mask])

        return { "original": self.effective_ranks_original, "denoised": self.effective_ranks_denoised }



    def explained_variance_ratios(self, n_components=10, behavior=None):
        """
        Compute the explained variance ratio of a matrix using PCA.
        """
        from sklearn.decomposition import PCA

        explained_variance_original = {
            beh: [] for beh in self.unique_behaviors
        }   

        explained_variance_denoised = {
            beh: [] for beh in self.unique_behaviors
        }

        if behavior is None:
            explained_variance_original = {}
            explained_variance_denoised = {}
            for beh in self.unique_behaviors:
                mask = self._mask(beh)

                corr_matrix_original = self._compute_correlation_matrix(self.X_original[mask])
                corr_matrix_denoised = self._compute_correlation_matrix(self.X_denoised[mask])

                svd_original = np.linalg.svd(corr_matrix_original, compute_uv=False)
                svd_denoised = np.linalg.svd(corr_matrix_denoised, compute_uv=False)

                explained_variance_original[beh] = svd_original[:n_components] / np.sum(svd_original)
                explained_variance_denoised[beh] = svd_denoised[:n_components] / np.sum(svd_denoised)

            return { "original": explained_variance_original, "denoised": explained_variance_denoised }
        else:
            mask = self._mask(behavior)
            svd_original = np.linalg.svd(self._compute_correlation_matrix(self.X_original[mask]), compute_uv=False)
            svd_denoised = np.linalg.svd(self._compute_correlation_matrix(self.X_denoised[mask]), compute_uv=False)

            explained_variance_original = svd_original[:n_components] / np.sum(svd_original)
            explained_variance_denoised = svd_denoised[:n_components] / np.sum(svd_denoised)

            return { "original": explained_variance_original, "denoised": explained_variance_denoised }


    def _hierarchical_clustering_full(self):
        """
        Perform hierarchical clustering on the correlation matrix of the full denoised dataset,
        then use the indexing to induce the clustering for the matrix for each behavior, for both 
        original and denoised data.
        """

        # Compute the correlation matrix for the full dataset
        corr_matrix_full = self._compute_correlation_matrix(self.X_denoised)

        # Perform hierarchical clustering
        Z = linkage(corr_matrix_full, method='ward')
        indexing = leaves_list(Z)

        # Reorder the correlation matrices for each behavior based on the full dataset clustering

        clustered_matrices_original = {
            beh: self._compute_correlation_matrix(self.X_original[self._mask(beh)])[np.ix_(indexing, indexing)]
            for beh in self.unique_behaviors
        }

        clustered_matrices_denoised = {
            beh: self._compute_correlation_matrix(self.X_denoised[self._mask(beh)])[np.ix_(indexing, indexing)]
            for beh in self.unique_behaviors
        }

        indexing_dict = {beh: indexing for beh in self.unique_behaviors}

        return { "original": clustered_matrices_original, "denoised": clustered_matrices_denoised, "indexing": indexing_dict }

    def _hierarchical_clustering_per_behavior(self):
        """
        Perform the clustering, for each behavior, on the denoised correlation matrix and use the indexing to
        order the original correlation matrix for that behavior. This is done for both original and denoised data.
        """

        clustered_matrices_original = {}
        clustered_matrices_denoised = {}
        indexing_dict = {}

        for beh in self.unique_behaviors:
            mask = self._mask(beh)

            # Compute the correlation matrices for the current behavior
            corr_matrix_original = self._compute_correlation_matrix(self.X_original[mask])
            corr_matrix_denoised = self._compute_correlation_matrix(self.X_denoised[mask])

            # Perform hierarchical clustering for the denoised data
            Z_denoised = linkage(corr_matrix_denoised, method='ward')
            indexing_denoised = leaves_list(Z_denoised)
            clustered_matrices_denoised[beh] = corr_matrix_denoised[np.ix_(indexing_denoised, indexing_denoised)]

            corr_matrix_original = corr_matrix_original[np.ix_(indexing_denoised, indexing_denoised)]
            clustered_matrices_original[beh] = corr_matrix_original
            indexing_dict[beh] = indexing_denoised

        return { "original": clustered_matrices_original, "denoised": clustered_matrices_denoised, "indexing": indexing_dict }

    def hierarchical_clustering(self, base='full'):
        """
        Perform hierarchical clustering on the correlation matrices.
        """

        if base == 'full':
            # Perform hierarchical clustering on the correlation matrix of the full denoised dataset,
            # then use the indexing to induce the clustering for the matrix for each behavior.
            return self._hierarchical_clustering_full()

        else:
            # Perform hierarchical clustering on the correlation matrix of each behavior separately.
            return self._hierarchical_clustering_per_behavior()

    def frobenius_norms(self, base='full'):
        """
        Compute the Frobenius norm of the difference between the original and denoised correlation matrices.
        """

        if base == 'full':
            clustered_matrices = self.hierarchical_clustering(base='full')
        else:
            clustered_matrices = self.hierarchical_clustering(base='per_behavior')

        frobenius_norms = {}
        for beh in self.unique_behaviors:
            frobenius_norms[beh] = np.linalg.norm(
                clustered_matrices['original'][beh] - clustered_matrices['denoised'][beh], ord='fro'
            )

        print(frobenius_norms)
        return frobenius_norms

    def silhouette_scores(self, base='full', max_clusters=8):
        """
        Compute the silhouette scores for the original and denoised correlation
        matrices by iterating over a range of cluster numbers to find the optimal
        clustering.

        Parameters:
        - base: str, 'full' or 'per_behavior' hierarchical clustering strategy
                to source the matrices.
        - max_clusters: int, the maximum number of clusters to test
                (loops from 2 to max_clusters).

        Returns:
        - dict: optimal silhouette scores and corresponding number of clusters
                ('best_k') for both 'original' and 'denoised' data per behavior.
        """
        # Correlation matrices pre-ordered by the selected hierarchical clustering
        clustered_matrices = self.hierarchical_clustering(
            base='full' if base == 'full' else 'per_behavior'
        )

        data_types = ("original", "denoised")

        silhouette_scores = {
            dt: {beh: {"best_score": -1.0, "best_k": None}
                for beh in self.unique_behaviors}
            for dt in data_types
        }

        # Silhouette requires at least 2 clusters
        cluster_range = range(2, max_clusters + 1)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )

        with progress:
            t_beh = progress.add_task(
                "Behaviors", total=len(self.unique_behaviors)
            )

            for beh in self.unique_behaviors:
                for dt in data_types:
                    corr_matrix = clustered_matrices[dt][beh]
                    affinity = (1 + corr_matrix) / 2
                    dist = np.clip(1 - corr_matrix, 0, 2)

                    t_k = progress.add_task(
                        f"  {beh} [dim]({dt})[/dim]", total=len(cluster_range)
                    )

                    best = silhouette_scores[dt][beh]
                    for k in cluster_range:
                        labels = SpectralClustering(
                            n_clusters=k, affinity='precomputed', random_state=42
                        ).fit_predict(affinity)
                        score = silhouette_score(dist, labels, metric='precomputed')

                        # Update if a better silhouette score is found
                        if score > best["best_score"]:
                            best["best_score"] = score
                            best["best_k"] = k

                        progress.update(
                            t_k,
                            advance=1,
                            description=(
                                f"  {beh} [dim]({dt})[/dim] "
                                f"[green]best {best['best_score']:.3f} "
                                f"(k={best['best_k']})[/green]"
                            ),
                        )

                    progress.remove_task(t_k)

                progress.advance(t_beh)

        return silhouette_scores

    def print_silhouette_summary(self, silhouette_scores: dict):
        """Print a summary table of the best silhouette score (and k) per behavior,
        for original vs. denoised data."""

        table = Table(
            title="Silhouette Scores Summary",
            title_style="bold cyan",
            box=box.ROUNDED,
            header_style="bold white",
            show_lines=False,
        )
        table.add_column("Behavior", style="bold", justify="left")
        table.add_column("Original score", justify="right")
        table.add_column("k", justify="center", style="dim")
        table.add_column("Denoised score", justify="right")
        table.add_column("k", justify="center", style="dim")
        table.add_column("Δ (den - orig)", justify="right")

        for beh in self.unique_behaviors:
            orig = silhouette_scores["original"][beh]
            den = silhouette_scores["denoised"][beh]

            delta = den["best_score"] - orig["best_score"]
            color = "green" if delta > 0 else "red" if delta < 0 else "white"

            table.add_row(
                str(beh),
                f"{orig['best_score']:.4f}",
                str(orig["best_k"]),
                f"{den['best_score']:.4f}",
                str(den["best_k"]),
                f"[{color}]{delta:+.4f}[/{color}]",
            )

        Console().print()
        Console().print(table)
        Console().print()

