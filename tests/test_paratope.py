"""Tests for the paratope module."""

import numpy as np
import pytest

from vhh_paratope_clustering.paratope import (
    identify_paratope_residues,
    compute_paratope_distance_matrix,
    _compute_rmsd,
)


def _make_numbered():
    """Create mock numbered sequence with CDR positions."""
    return [
        ((1, " "), "Q"),
        ((27, " "), "G"),  # CDR1
        ((28, " "), "S"),
        ((38, " "), "F"),  # CDR1 end
        ((45, " "), "R"),  # framework
        ((56, " "), "T"),  # CDR2
        ((65, " "), "T"),  # CDR2 end
        ((80, " "), "A"),  # framework
        ((105, " "), "H"),  # CDR3
        ((117, " "), "Y"),  # CDR3 end
        ((118, " "), "W"),  # framework
    ]


class TestIdentifyParatopeResidues:
    """Tests for identify_paratope_residues function."""

    def test_returns_cdr_residues_only(self):
        """Test that only CDR residues are returned as paratope."""
        numbered = _make_numbered()
        paratope = identify_paratope_residues(numbered)

        paratope_positions = [pos[0] for pos, _ in paratope]
        # Should include CDR positions but not framework
        assert 27 in paratope_positions
        assert 56 in paratope_positions
        assert 105 in paratope_positions
        assert 1 not in paratope_positions
        assert 45 not in paratope_positions
        assert 118 not in paratope_positions

    def test_sorted_by_position(self):
        """Test that paratope residues are sorted by position."""
        numbered = _make_numbered()
        paratope = identify_paratope_residues(numbered)

        positions = [pos[0] for pos, _ in paratope]
        assert positions == sorted(positions)


class TestComputeRmsd:
    """Tests for the Kabsch RMSD calculation."""

    def test_identical_coordinates(self):
        """Test RMSD of identical coordinate sets is zero."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
        assert _compute_rmsd(coords, coords) == pytest.approx(0.0, abs=1e-10)

    def test_translated_coordinates(self):
        """Test RMSD after translation is zero (alignment removes it)."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
        coords2 = coords1 + np.array([10.0, 20.0, 30.0])
        assert _compute_rmsd(coords1, coords2) == pytest.approx(0.0, abs=1e-6)

    def test_rotated_coordinates(self):
        """Test RMSD after rotation is zero (Kabsch aligns them)."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # 90-degree rotation around z-axis
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        coords2 = (R @ coords1.T).T
        assert _compute_rmsd(coords1, coords2) == pytest.approx(0.0, abs=1e-6)

    def test_different_coordinates_nonzero_rmsd(self):
        """Test RMSD of different structures is non-zero."""
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                            [0.0, 1.0, 1.0]])
        rmsd = _compute_rmsd(coords1, coords2)
        assert rmsd > 0


class TestComputeParatopeDistanceMatrix:
    """Tests for compute_paratope_distance_matrix function."""

    def test_diagonal_is_zero(self):
        """Test that self-distances are zero."""
        coords = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ]
        dist = compute_paratope_distance_matrix(coords)
        assert dist[0, 0] == pytest.approx(0.0)
        assert dist[1, 1] == pytest.approx(0.0)

    def test_symmetry(self):
        """Test that distance matrix is symmetric."""
        coords = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ]
        dist = compute_paratope_distance_matrix(coords)
        assert dist.shape == (3, 3)
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_mismatched_shapes_raises(self):
        """Test that mismatched coordinate shapes raise ValueError."""
        coords = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
        ]
        with pytest.raises(ValueError, match="different shapes"):
            compute_paratope_distance_matrix(coords)
