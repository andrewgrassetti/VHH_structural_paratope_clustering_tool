"""Unit tests for feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from vhh_clustering.cdr_annotation import AnnotatedResidue
from vhh_clustering.features import (
    ParatopeFeatures,
    _charge_features,
    _hydro_polar_aromatic,
    _hotspot_score,
    extract_features,
)
from vhh_clustering.parsing import Residue


def _make_annotated(
    one_letter: str, region: str, ca_coord=None, res_seq: int = 1
) -> AnnotatedResidue:
    """Helper to build a minimal AnnotatedResidue."""
    return AnnotatedResidue(
        residue=Residue(
            chain_id="A",
            res_seq=res_seq,
            res_name="ALA",
            one_letter=one_letter,
            ca_coord=np.array(ca_coord) if ca_coord else None,
        ),
        imgt_position=res_seq,
        region=region,
    )


class TestChargeFeatures:
    def test_neutral(self) -> None:
        result = _charge_features("AAAA")
        assert result[0] == pytest.approx(0.0)  # net charge

    def test_positive(self) -> None:
        result = _charge_features("RRRR")
        assert result[0] > 0  # net positive

    def test_negative(self) -> None:
        result = _charge_features("DDDD")
        assert result[0] < 0  # net negative


class TestHydrophobicPolarAromatic:
    def test_all_hydrophobic(self) -> None:
        result = _hydro_polar_aromatic("LLLL")
        assert result[0] == pytest.approx(1.0)

    def test_aromatic(self) -> None:
        result = _hydro_polar_aromatic("FFFF")
        assert result[2] == pytest.approx(1.0)


class TestHotspotScore:
    def test_empty(self) -> None:
        assert _hotspot_score([]) == 0.0

    def test_cdr_h3_weights_highest(self) -> None:
        residues_h3 = [_make_annotated("A", "CDR-H3", res_seq=110 + i) for i in range(5)]
        residues_h1 = [_make_annotated("A", "CDR-H1", res_seq=27 + i) for i in range(5)]
        score_h3 = _hotspot_score(residues_h3)
        score_h1 = _hotspot_score(residues_h1)
        assert score_h3 > score_h1


class TestExtractFeatures:
    def test_vector_length_consistent(self) -> None:
        residues = [
            _make_annotated("A", "framework", ca_coord=[0, 0, 0], res_seq=1),
            _make_annotated("R", "CDR-H1", ca_coord=[1, 1, 1], res_seq=30),
            _make_annotated("D", "CDR-H2", ca_coord=[5, 5, 5], res_seq=60),
            _make_annotated("W", "CDR-H3", ca_coord=[10, 10, 10], res_seq=110),
        ]
        feats = extract_features("test", residues)
        assert isinstance(feats, ParatopeFeatures)
        assert len(feats.vector) == len(feats.feature_names)
        assert feats.hotspot_score > 0

    def test_cdr_sequences_extracted(self) -> None:
        residues = [
            _make_annotated("G", "CDR-H1", res_seq=30),
            _make_annotated("S", "CDR-H1", res_seq=31),
        ]
        feats = extract_features("test", residues)
        assert feats.cdr_sequences["CDR-H1"] == "GS"
