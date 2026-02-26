"""Feature extraction for VHH paratope characterisation.

Computes a fixed-length feature vector per structure encompassing:
1. CDR composition & length
2. Surface accessibility proxy (relative SASA approximation)
3. Local geometry descriptors (Cα neighbourhood statistics)
4. Electrostatics proxy (charge composition)
5. Hotspot score (frequency-weighted CDR interface participation)

The resulting vector is suitable for dimensionality reduction and clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from Bio.Data.IUPACData import protein_letters
from scipy.spatial.distance import cdist, pdist

from vhh_clustering.cdr_annotation import IMGT_CDR_RANGES, AnnotatedResidue

# Amino-acid property look-ups
_CHARGED_POS = set("RKH")
_CHARGED_NEG = set("DE")
_HYDROPHOBIC = set("AILMFWVP")
_POLAR = set("STNQYC")
_AROMATIC = set("FWY")

# Historical CDR interface participation weights (literature-derived
# approximations; CDR-H3 contributes most to antigen binding).
CDR_INTERFACE_WEIGHTS: dict[str, float] = {
    "CDR-H1": 0.20,
    "CDR-H2": 0.25,
    "CDR-H3": 0.55,
}


def _aa_composition(seq: str) -> dict[str, float]:
    """Return fractional amino-acid composition."""
    n = max(len(seq), 1)
    return {aa: seq.count(aa) / n for aa in protein_letters.upper()}


def _charge_features(seq: str) -> list[float]:
    """Net charge proxy, positive fraction, negative fraction."""
    n = max(len(seq), 1)
    pos = sum(1 for c in seq if c in _CHARGED_POS) / n
    neg = sum(1 for c in seq if c in _CHARGED_NEG) / n
    return [pos - neg, pos, neg]


def _hydro_polar_aromatic(seq: str) -> list[float]:
    """Fraction hydrophobic, polar, aromatic."""
    n = max(len(seq), 1)
    return [
        sum(1 for c in seq if c in _HYDROPHOBIC) / n,
        sum(1 for c in seq if c in _POLAR) / n,
        sum(1 for c in seq if c in _AROMATIC) / n,
    ]


def _ca_geometry_stats(
    residues: Sequence[AnnotatedResidue],
) -> list[float]:
    """Local Cα geometry: mean & std of pairwise Cα distances within CDR residues.

    Also computes the mean Cα distance to the CDR centroid and the radius
    of gyration of CDR Cα atoms.  Returns 4 floats.
    """
    coords = []
    for ar in residues:
        if ar.region.startswith("CDR") and ar.residue.ca_coord is not None:
            coords.append(ar.residue.ca_coord)
    if len(coords) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    coords_arr = np.array(coords)
    centroid = coords_arr.mean(axis=0)
    dists_to_centroid = np.linalg.norm(coords_arr - centroid, axis=1)
    # Pairwise distances (upper triangle)
    pairwise_distances = pdist(coords_arr)
    return [
        float(np.mean(pairwise_distances)),
        float(np.std(pairwise_distances)),
        float(np.mean(dists_to_centroid)),
        float(np.std(dists_to_centroid)),  # radius-of-gyration proxy
    ]


def _sasa_proxy(residues: Sequence[AnnotatedResidue]) -> list[float]:
    """Approximate surface accessibility via Cα neighbour count.

    For each CDR residue, count how many other Cα atoms are within 8 Å
    (lower count → more exposed).  Returns [mean_exposure, std_exposure]
    for CDR residues, where exposure = 1 / (neighbour_count + 1).
    """
    all_ca = []
    for ar in residues:
        if ar.residue.ca_coord is not None:
            all_ca.append(ar.residue.ca_coord)
    if not all_ca:
        return [0.0, 0.0]
    all_ca_arr = np.array(all_ca)

    cdr_ca = []
    for ar in residues:
        if ar.region.startswith("CDR") and ar.residue.ca_coord is not None:
            cdr_ca.append(ar.residue.ca_coord)
    if not cdr_ca:
        return [0.0, 0.0]
    cdr_ca_arr = np.array(cdr_ca)

    # Vectorised pairwise distances: shape (n_cdr, n_all)
    dists = cdist(cdr_ca_arr, all_ca_arr)
    # Subtract 1 to exclude self (each CDR atom is also in all_ca)
    neighbour_counts = np.sum(dists < 8.0, axis=1) - 1
    exposures = 1.0 / (neighbour_counts + 1)
    return [float(np.mean(exposures)), float(np.std(exposures))]


def _hotspot_score(residues: Sequence[AnnotatedResidue]) -> float:
    """Weighted hotspot score combining CDR length and interface weights."""
    score = 0.0
    for cdr_name, weight in CDR_INTERFACE_WEIGHTS.items():
        cdr_len = sum(1 for ar in residues if ar.region == cdr_name)
        score += weight * cdr_len
    return score


@dataclass
class ParatopeFeatures:
    """Feature vector and metadata for one VHH structure."""

    name: str
    vector: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    cdr_sequences: dict[str, str] = field(default_factory=dict)
    hotspot_score: float = 0.0
    residue_details: list[dict] = field(default_factory=list)


def extract_features(
    name: str, annotated: list[AnnotatedResidue]
) -> ParatopeFeatures:
    """Build a fixed-length feature vector from annotated residues."""
    feature_names: list[str] = []
    parts: list[float] = []

    # Pre-compute CDR sequences once for reuse across feature groups
    cdr_seqs: dict[str, str] = {
        cdr_name: "".join(
            ar.residue.one_letter for ar in annotated if ar.region == cdr_name
        )
        for cdr_name in IMGT_CDR_RANGES
    }

    # --- 1. Per-CDR length ---
    for cdr_name in IMGT_CDR_RANGES:
        cdr_res = [ar for ar in annotated if ar.region == cdr_name]
        parts.append(float(len(cdr_res)))
        feature_names.append(f"{cdr_name}_length")

    # --- 2. Per-CDR amino-acid composition (20 features × 3 CDRs) ---
    aa_letters = sorted(protein_letters.upper())
    for cdr_name in IMGT_CDR_RANGES:
        seq = cdr_seqs[cdr_name]
        comp = _aa_composition(seq)
        for aa in aa_letters:
            parts.append(comp.get(aa, 0.0))
            feature_names.append(f"{cdr_name}_frac_{aa}")

    # --- 3. Charge / electrostatics proxy per CDR ---
    for cdr_name in IMGT_CDR_RANGES:
        seq = cdr_seqs[cdr_name]
        charge_feats = _charge_features(seq)
        parts.extend(charge_feats)
        feature_names.extend(
            [f"{cdr_name}_net_charge", f"{cdr_name}_pos_frac", f"{cdr_name}_neg_frac"]
        )

    # --- 4. Hydrophobic / polar / aromatic per CDR ---
    for cdr_name in IMGT_CDR_RANGES:
        seq = cdr_seqs[cdr_name]
        hpa = _hydro_polar_aromatic(seq)
        parts.extend(hpa)
        feature_names.extend(
            [f"{cdr_name}_hydro", f"{cdr_name}_polar", f"{cdr_name}_aromatic"]
        )

    # --- 5. Structural geometry (4 features) ---
    geom = _ca_geometry_stats(annotated)
    parts.extend(geom)
    feature_names.extend(
        ["cdr_pw_dist_mean", "cdr_pw_dist_std", "cdr_centroid_mean", "cdr_centroid_std"]
    )

    # --- 6. SASA proxy (2 features) ---
    sasa = _sasa_proxy(annotated)
    parts.extend(sasa)
    feature_names.extend(["sasa_exposure_mean", "sasa_exposure_std"])

    # --- 7. Hotspot score (1 feature) ---
    hs = _hotspot_score(annotated)
    parts.append(hs)
    feature_names.append("hotspot_score")

    # --- Per-CDR sequences already computed above ---

    # --- Residue-level details for output table ---
    residue_details = []
    for ar in annotated:
        residue_details.append(
            {
                "chain": ar.residue.chain_id,
                "res_seq": ar.residue.res_seq,
                "res_name": ar.residue.res_name,
                "one_letter": ar.residue.one_letter,
                "imgt_pos": ar.imgt_position,
                "region": ar.region,
                "b_factor": round(ar.residue.b_factor, 2),
                "is_paratope_candidate": ar.region.startswith("CDR"),
            }
        )

    return ParatopeFeatures(
        name=name,
        vector=np.array(parts, dtype=np.float64),
        feature_names=feature_names,
        cdr_sequences=cdr_seqs,
        hotspot_score=hs,
        residue_details=residue_details,
    )
