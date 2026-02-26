"""SPACE2-inspired structural clustering for VHH paratopes.

Implements the core algorithmic idea from SPACE2 (Spoendlin *et al.*,
*Frontiers in Molecular Biosciences*, 2023) adapted for single-domain
VHH antibodies:

1. Superimpose structures on framework Cα atoms (Kabsch alignment).
2. Compute pairwise CDR Cα RMSD across all structures.
3. Cluster using agglomerative clustering with a distance threshold.

This avoids a full SPACE2 dependency while preserving the structural
clustering logic, and works natively with the ``ParsedStructure`` /
``AnnotatedResidue`` types already used in this tool.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from vhh_clustering.cdr_annotation import AnnotatedResidue

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _extract_ca_coords(
    residues: Sequence[AnnotatedResidue],
    region_filter: str | None = None,
) -> np.ndarray:
    """Return an (N, 3) array of Cα coordinates.

    Parameters
    ----------
    residues:
        Annotated residues from a single structure.
    region_filter:
        If ``None``, return all Cα with coordinates.
        If ``"framework"``, return only framework Cα.
        If it starts with ``"CDR"``, return only CDR Cα.
    """
    coords = []
    for ar in residues:
        if ar.residue.ca_coord is None:
            continue
        if region_filter is None:
            coords.append(ar.residue.ca_coord)
        elif region_filter == "framework" and ar.region == "framework":
            coords.append(ar.residue.ca_coord)
        elif region_filter.startswith("CDR") and ar.region.startswith("CDR"):
            coords.append(ar.residue.ca_coord)
    if not coords:
        return np.empty((0, 3))
    return np.array(coords)


def _kabsch_transform(
    mobile: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Kabsch rotation and translation.

    Both arrays must have the same shape ``(N, 3)``.  Returns
    ``(R, centroid_mobile, centroid_target)`` so the transform can be
    applied to arbitrary point sets as::

        aligned = (pts - centroid_mobile) @ R.T + centroid_target
    """
    assert mobile.shape == target.shape and mobile.ndim == 2
    n = mobile.shape[0]
    if n == 0:
        return np.eye(3), np.zeros(3), np.zeros(3)

    centroid_m = mobile.mean(axis=0)
    centroid_t = target.mean(axis=0)
    m_centered = mobile - centroid_m
    t_centered = target - centroid_t

    H = m_centered.T @ t_centered
    U, _S, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    return R, centroid_m, centroid_t


def _kabsch_align(
    mobile: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Align *mobile* onto *target* using the Kabsch algorithm.

    Both arrays must have the same shape ``(N, 3)``.  Returns the
    rotated + translated *mobile* coordinates.
    """
    R, centroid_m, centroid_t = _kabsch_transform(mobile, target)
    return (mobile - centroid_m) @ R.T + centroid_t


def _ca_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Cα RMSD between two equal-length coordinate arrays."""
    if len(coords1) == 0 or len(coords2) == 0:
        return 0.0
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pairwise_cdr_rmsd(
    structures: list[list[AnnotatedResidue]],
) -> np.ndarray:
    """Compute pairwise CDR Cα RMSD after framework superposition.

    Parameters
    ----------
    structures:
        List of annotated residue lists, one per structure.

    Returns
    -------
    np.ndarray
        Symmetric (N, N) distance matrix of CDR Cα RMSDs (Å).
    """
    n = len(structures)
    dist_matrix = np.zeros((n, n))

    # Pre-extract coordinates
    fw_coords = [_extract_ca_coords(s, "framework") for s in structures]
    cdr_coords = [_extract_ca_coords(s, "CDR") for s in structures]

    for i in range(n):
        for j in range(i + 1, n):
            fw_i, fw_j = fw_coords[i], fw_coords[j]
            cdr_i, cdr_j = cdr_coords[i], cdr_coords[j]

            # Need matching lengths for alignment & RMSD
            min_fw = min(len(fw_i), len(fw_j))
            min_cdr = min(len(cdr_i), len(cdr_j))

            if min_fw < 3 or min_cdr < 1:
                dist_matrix[i, j] = dist_matrix[j, i] = 0.0
                continue

            # Truncate to common length (handles minor length differences)
            fw_i_t = fw_i[:min_fw]
            fw_j_t = fw_j[:min_fw]
            cdr_i_t = cdr_i[:min_cdr]
            cdr_j_t = cdr_j[:min_cdr]

            # Compute alignment transform from framework atoms
            R, centroid_m, centroid_t = _kabsch_transform(fw_j_t, fw_i_t)

            # Apply the same transform to CDR atoms
            aligned_cdr_j = (cdr_j_t - centroid_m) @ R.T + centroid_t

            d_val = _ca_rmsd(cdr_i_t, aligned_cdr_j)
            dist_matrix[i, j] = d_val
            dist_matrix[j, i] = d_val

    return dist_matrix


def structural_cluster(
    structures: list[list[AnnotatedResidue]],
    names: list[str],
    distance_threshold: float = 1.25,
) -> pd.DataFrame:
    """Cluster VHH structures by CDR Cα RMSD (SPACE2-style).

    Parameters
    ----------
    structures:
        List of annotated-residue lists, one per structure.
    names:
        Parallel list of structure identifiers.
    distance_threshold:
        Agglomerative clustering distance cutoff in Å.  The SPACE2
        default of 1.25 Å is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``structure``, ``structural_cluster``,
        and ``representative``.
    """
    n = len(structures)
    if n < 2:
        return pd.DataFrame(
            {
                "structure": names,
                "structural_cluster": list(range(n)),
                "representative": names,
            }
        )

    dist_matrix = pairwise_cdr_rmsd(structures)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        distance_threshold=distance_threshold,
        linkage="complete",
    )
    labels = clustering.fit_predict(dist_matrix)

    # Pick a representative per cluster (first member)
    representatives: dict[int, str] = {}
    for i, label in enumerate(labels):
        if label not in representatives:
            representatives[label] = names[i]

    return pd.DataFrame(
        {
            "structure": names,
            "structural_cluster": labels.tolist(),
            "representative": [representatives[lab] for lab in labels],
        }
    )
