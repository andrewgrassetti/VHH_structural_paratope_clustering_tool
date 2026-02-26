"""Unit tests for SPACE2-inspired structural clustering."""

from __future__ import annotations

import numpy as np
import pytest

from vhh_clustering.cdr_annotation import AnnotatedResidue
from vhh_clustering.parsing import Residue
from vhh_clustering.structural_clustering import (
    _ca_rmsd,
    _extract_ca_coords,
    _kabsch_align,
    pairwise_cdr_rmsd,
    structural_cluster,
)


def _make_ar(
    region: str, ca: list[float], res_seq: int = 1
) -> AnnotatedResidue:
    return AnnotatedResidue(
        residue=Residue(
            chain_id="A",
            res_seq=res_seq,
            res_name="ALA",
            one_letter="A",
            ca_coord=np.array(ca, dtype=float),
        ),
        imgt_position=res_seq,
        region=region,
    )


def _make_structure(
    fw_coords: list[list[float]], cdr_coords: list[list[float]]
) -> list[AnnotatedResidue]:
    """Build a structure with framework and CDR residues."""
    residues = []
    for i, c in enumerate(fw_coords):
        residues.append(_make_ar("framework", c, res_seq=i + 1))
    for i, c in enumerate(cdr_coords):
        residues.append(_make_ar("CDR-H1", c, res_seq=100 + i))
    return residues


class TestExtractCaCoords:
    def test_all_coords(self) -> None:
        s = [_make_ar("framework", [0, 0, 0]), _make_ar("CDR-H1", [1, 1, 1])]
        coords = _extract_ca_coords(s)
        assert coords.shape == (2, 3)

    def test_framework_only(self) -> None:
        s = [_make_ar("framework", [0, 0, 0]), _make_ar("CDR-H1", [1, 1, 1])]
        coords = _extract_ca_coords(s, "framework")
        assert coords.shape == (1, 3)

    def test_cdr_only(self) -> None:
        s = [_make_ar("framework", [0, 0, 0]), _make_ar("CDR-H1", [1, 1, 1])]
        coords = _extract_ca_coords(s, "CDR")
        assert coords.shape == (1, 3)


class TestKabschAlign:
    def test_identity(self) -> None:
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        aligned = _kabsch_align(pts, pts)
        np.testing.assert_allclose(aligned, pts, atol=1e-10)

    def test_translation(self) -> None:
        target = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        mobile = target + np.array([5, 5, 5])
        aligned = _kabsch_align(mobile, target)
        np.testing.assert_allclose(aligned, target, atol=1e-10)

    def test_rotation(self) -> None:
        target = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
        # 90° rotation around z-axis
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        mobile = (target @ R.T)
        aligned = _kabsch_align(mobile, target)
        np.testing.assert_allclose(aligned, target, atol=1e-10)


class TestCaRmsd:
    def test_identical_is_zero(self) -> None:
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        assert _ca_rmsd(pts, pts) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        a = np.array([[0, 0, 0]], dtype=float)
        b = np.array([[3, 4, 0]], dtype=float)
        assert _ca_rmsd(a, b) == pytest.approx(5.0)


class TestPairwiseCdrRmsd:
    def test_symmetric(self) -> None:
        fw = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        s1 = _make_structure(fw, [[2, 2, 0], [3, 2, 0]])
        s2 = _make_structure(fw, [[4, 4, 0], [5, 4, 0]])
        dist = pairwise_cdr_rmsd([s1, s2])
        np.testing.assert_allclose(dist, dist.T)

    def test_diagonal_zero(self) -> None:
        fw = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        s1 = _make_structure(fw, [[2, 2, 0]])
        dist = pairwise_cdr_rmsd([s1, s1])
        assert dist[0, 1] == pytest.approx(0.0, abs=1e-10)

    def test_identical_structures_zero(self) -> None:
        fw = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        cdr = [[2, 2, 0], [3, 2, 0]]
        s1 = _make_structure(fw, cdr)
        s2 = _make_structure(fw, cdr)
        dist = pairwise_cdr_rmsd([s1, s2])
        assert dist[0, 1] == pytest.approx(0.0, abs=1e-10)


class TestStructuralCluster:
    def test_identical_in_same_cluster(self) -> None:
        fw = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        cdr = [[2, 2, 0], [3, 2, 0]]
        s1 = _make_structure(fw, cdr)
        s2 = _make_structure(fw, cdr)
        df = structural_cluster([s1, s2], ["s1", "s2"])
        assert df.loc[0, "structural_cluster"] == df.loc[1, "structural_cluster"]

    def test_different_in_different_clusters(self) -> None:
        fw = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        s1 = _make_structure(fw, [[2, 2, 0], [3, 2, 0]])
        s2 = _make_structure(fw, [[20, 20, 0], [21, 20, 0]])
        df = structural_cluster([s1, s2], ["s1", "s2"], distance_threshold=1.0)
        assert df.loc[0, "structural_cluster"] != df.loc[1, "structural_cluster"]

    def test_single_structure(self) -> None:
        fw = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        s1 = _make_structure(fw, [[2, 2, 0]])
        df = structural_cluster([s1], ["s1"])
        assert len(df) == 1
        assert df.loc[0, "structural_cluster"] == 0

    def test_output_columns(self) -> None:
        fw = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        s1 = _make_structure(fw, [[2, 2, 0]])
        s2 = _make_structure(fw, [[3, 3, 0]])
        df = structural_cluster([s1, s2], ["s1", "s2"])
        assert "structure" in df.columns
        assert "structural_cluster" in df.columns
        assert "representative" in df.columns
