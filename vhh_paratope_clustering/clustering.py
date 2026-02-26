"""Clustering of VHH structural paratopes.

This module provides methods for clustering VHH/nanobody paratopes
based on their 3D structural similarity. It uses pairwise RMSD
distance matrices (from the ``paratope`` module) and applies
hierarchical or other clustering algorithms.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering


def cluster_paratopes(distance_matrix, method="hierarchical",
                      threshold=2.0, n_clusters=None, linkage_method="average"):
    """Cluster paratopes based on a pairwise distance matrix.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Symmetric pairwise distance matrix of shape (N, N), e.g. from
        ``compute_paratope_distance_matrix``.
    method : str, optional
        Clustering method. One of 'hierarchical' (scipy linkage-based)
        or 'agglomerative' (scikit-learn). Default is 'hierarchical'.
    threshold : float, optional
        Distance threshold for flat cluster assignment when using
        ``method='hierarchical'``. Default is 2.0 Angstroms.
    n_clusters : int, optional
        Number of clusters for ``method='agglomerative'``. If None,
        ``distance_threshold`` is used instead.
    linkage_method : str, optional
        Linkage method for hierarchical clustering. One of 'single',
        'complete', 'average', 'weighted'. Default is 'average'.

    Returns
    -------
    numpy.ndarray
        Array of cluster labels of length N.
    """
    if method == "hierarchical":
        return _cluster_hierarchical(distance_matrix, threshold,
                                     linkage_method)
    elif method == "agglomerative":
        return _cluster_agglomerative(distance_matrix, n_clusters, threshold)
    else:
        raise ValueError(
            f"Unknown clustering method '{method}'. "
            "Use 'hierarchical' or 'agglomerative'."
        )


def _cluster_hierarchical(distance_matrix, threshold, linkage_method):
    """Hierarchical clustering using scipy."""
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.ones(n, dtype=int)
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=threshold, criterion="distance")
    return labels


def _cluster_agglomerative(distance_matrix, n_clusters, threshold):
    """Agglomerative clustering using scikit-learn."""
    kwargs = {"metric": "precomputed", "linkage": "average"}
    if n_clusters is not None:
        kwargs["n_clusters"] = n_clusters
    else:
        kwargs["n_clusters"] = None
        kwargs["distance_threshold"] = threshold
    model = AgglomerativeClustering(**kwargs)
    return model.fit_predict(distance_matrix)
