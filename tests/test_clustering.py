"""Tests for the clustering module."""

import numpy as np
import pytest

from vhh_paratope_clustering.clustering import cluster_paratopes


class TestClusterParatopes:
    """Tests for cluster_paratopes function."""

    def _make_distance_matrix(self):
        """Create a test distance matrix with clear cluster structure."""
        # Two clusters: {0,1} close together, {2,3} close together
        dm = np.array([
            [0.0, 0.5, 5.0, 5.5],
            [0.5, 0.0, 5.5, 5.0],
            [5.0, 5.5, 0.0, 0.3],
            [5.5, 5.0, 0.3, 0.0],
        ])
        return dm

    def test_hierarchical_two_clusters(self):
        """Test hierarchical clustering finds two clusters."""
        dm = self._make_distance_matrix()
        labels = cluster_paratopes(dm, method="hierarchical", threshold=2.0)

        assert len(labels) == 4
        # Items 0 and 1 should be in the same cluster
        assert labels[0] == labels[1]
        # Items 2 and 3 should be in the same cluster
        assert labels[2] == labels[3]
        # The two groups should be in different clusters
        assert labels[0] != labels[2]

    def test_agglomerative_two_clusters(self):
        """Test agglomerative clustering with n_clusters."""
        dm = self._make_distance_matrix()
        labels = cluster_paratopes(dm, method="agglomerative", n_clusters=2)

        assert len(labels) == 4
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_agglomerative_with_threshold(self):
        """Test agglomerative clustering with distance threshold."""
        dm = self._make_distance_matrix()
        labels = cluster_paratopes(dm, method="agglomerative", threshold=2.0)

        assert len(labels) == 4
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]

    def test_single_element(self):
        """Test clustering with a single element."""
        dm = np.array([[0.0]])
        labels = cluster_paratopes(dm, method="hierarchical", threshold=2.0)
        assert len(labels) == 1

    def test_unknown_method_raises(self):
        """Test that an unknown method raises ValueError."""
        dm = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="Unknown clustering method"):
            cluster_paratopes(dm, method="invalid_method")
