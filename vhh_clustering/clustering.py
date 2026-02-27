"""Dimensionality reduction and clustering of paratope feature vectors.

Supports UMAP, t-SNE, and PCA for projection, and HDBSCAN for clustering.
GPU acceleration via cuML is attempted first; falls back to CPU
implementations transparently.

Includes a helper to export clustering results to CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# GPU / CPU toggle helpers
# ---------------------------------------------------------------------------
_USE_GPU = False
try:
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from cuml.manifold import UMAP as cuUMAP, TSNE as cuTSNE

    _USE_GPU = True
except ImportError:
    pass

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import hdbscan
import umap


def gpu_available() -> bool:
    """Return ``True`` if RAPIDS cuML was successfully imported."""
    return _USE_GPU


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reduce_dimensions(
    matrix: np.ndarray,
    method: Literal["umap", "tsne", "pca"] = "umap",
    n_components: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Project *matrix* (n_samples × n_features) to *n_components* dims.

    The caller is responsible for standardising *matrix* beforehand.
    """
    if matrix.shape[0] < 2:
        return matrix[:, :n_components] if matrix.shape[1] >= n_components else matrix

    X = matrix

    if method == "umap":
        n_neighbors = min(15, max(2, X.shape[0] - 1))
        if _USE_GPU:
            reducer = cuUMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
        else:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
        return reducer.fit_transform(X)

    if method == "tsne":
        perplexity = min(30.0, max(1.0, (X.shape[0] - 1) / 3.0))
        if _USE_GPU:
            reducer = cuTSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
        else:
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
        return reducer.fit_transform(X)

    # PCA
    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    return pca.fit_transform(X)


def cluster(
    matrix: np.ndarray,
    min_cluster_size: int = 3,
) -> np.ndarray:
    """Cluster rows of *matrix* using HDBSCAN.

    Returns an integer label array (−1 = noise).

    HDBSCAN is chosen over KMeans because the number of structural
    paratope bins is not known a priori and HDBSCAN handles noise
    naturally—important when dealing with diverse predicted structures.

    The caller is responsible for standardising *matrix* beforehand.
    """
    if matrix.shape[0] < min_cluster_size:
        return np.zeros(matrix.shape[0], dtype=int)

    X = matrix

    if _USE_GPU:
        clusterer = cuHDBSCAN(min_cluster_size=min_cluster_size)
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
        )
    labels = clusterer.fit_predict(X)
    return np.asarray(labels)


def extract_tag(name: str) -> str:
    """Derive a default tag from a structure name.

    The tag is everything after the first underscore in *name*.  If no
    underscore is present the tag is an empty string.

    Examples
    --------
    >>> extract_tag("antibody_targetX")
    'targetX'
    >>> extract_tag("a_b_c")
    'b_c'
    >>> extract_tag("mystructure")
    ''
    """
    if "_" in name:
        return name.split("_", 1)[1]
    return ""


def build_result_dataframe(
    names: list[str],
    embedding: np.ndarray,
    labels: np.ndarray,
    hotspot_scores: list[float],
    cdr_sequences: list[dict[str, str]],
    tags: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble a tidy DataFrame ready for plotting / export.

    Parameters
    ----------
    tags:
        Optional per-structure tag strings.  When *None*, tags are
        derived automatically from *names* using :func:`extract_tag`.
    """
    if tags is None:
        tags = [extract_tag(n) for n in names]

    df = pd.DataFrame(
        {
            "structure": names,
            "tag": tags,
            "dim1": embedding[:, 0],
            "dim2": embedding[:, 1] if embedding.shape[1] > 1 else 0.0,
            "dim3": embedding[:, 2] if embedding.shape[1] > 2 else 0.0,
            "cluster": labels,
            "hotspot_score": hotspot_scores,
        }
    )
    # Expand CDR sequences
    for key in ("CDR-H1", "CDR-H2", "CDR-H3"):
        df[key] = [s.get(key, "") for s in cdr_sequences]
    return df


def export_csv(
    result_df: pd.DataFrame,
    path: Union[str, Path],
    *,
    include_index: bool = False,
) -> Path:
    """Write a clustering result DataFrame to a CSV file.

    Parameters
    ----------
    result_df:
        DataFrame returned by :func:`build_result_dataframe` (or any
        DataFrame containing clustering results).
    path:
        Destination file path.  Parent directories are created
        automatically if they do not exist.
    include_index:
        Whether to write the DataFrame index as a column. Defaults to
        ``False``.

    Returns
    -------
    Path
        The resolved path of the written CSV file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(dest, index=include_index)
    return dest.resolve()
