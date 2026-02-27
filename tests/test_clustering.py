"""Unit tests for dimensionality reduction and clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from vhh_clustering.clustering import cluster, reduce_dimensions, build_result_dataframe, export_csv, extract_tag


class TestReduceDimensions:
    @pytest.fixture()
    def sample_matrix(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((10, 20))

    @pytest.mark.parametrize("method", ["umap", "tsne", "pca"])
    def test_output_shape(self, sample_matrix: np.ndarray, method: str) -> None:
        result = reduce_dimensions(sample_matrix, method=method, n_components=2)
        assert result.shape == (10, 2)

    @pytest.mark.parametrize("method", ["umap", "tsne", "pca"])
    def test_output_shape_3d(self, sample_matrix: np.ndarray, method: str) -> None:
        result = reduce_dimensions(sample_matrix, method=method, n_components=3)
        assert result.shape == (10, 3)

    def test_single_sample(self) -> None:
        X = np.ones((1, 5))
        result = reduce_dimensions(X, method="pca", n_components=2)
        assert result.shape[0] == 1

    def test_caller_standardises(self) -> None:
        """reduce_dimensions should accept pre-standardised input and not re-standardise."""
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((10, 20))
        X = StandardScaler().fit_transform(raw)
        result = reduce_dimensions(X, method="pca", n_components=2)
        assert result.shape == (10, 2)


class TestCluster:
    def test_cluster_labels_shape(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((15, 10))
        labels = cluster(X, min_cluster_size=3)
        assert labels.shape == (15,)

    def test_small_input(self) -> None:
        X = np.ones((2, 5))
        labels = cluster(X, min_cluster_size=3)
        assert len(labels) == 2


class TestBuildResultDataframe:
    def test_columns_2d(self) -> None:
        df = build_result_dataframe(
            names=["a", "b"],
            embedding=np.array([[1.0, 2.0], [3.0, 4.0]]),
            labels=np.array([0, 1]),
            hotspot_scores=[0.5, 0.8],
            cdr_sequences=[
                {"CDR-H1": "AA", "CDR-H2": "BB", "CDR-H3": "CC"},
                {"CDR-H1": "DD", "CDR-H2": "EE", "CDR-H3": "FF"},
            ],
        )
        assert "structure" in df.columns
        assert "cluster" in df.columns
        assert "dim3" in df.columns
        assert len(df) == 2

    def test_columns_3d(self) -> None:
        df = build_result_dataframe(
            names=["a", "b"],
            embedding=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            labels=np.array([0, 1]),
            hotspot_scores=[0.5, 0.8],
            cdr_sequences=[
                {"CDR-H1": "AA", "CDR-H2": "BB", "CDR-H3": "CC"},
                {"CDR-H1": "DD", "CDR-H2": "EE", "CDR-H3": "FF"},
            ],
        )
        assert "dim1" in df.columns
        assert "dim2" in df.columns
        assert "dim3" in df.columns
        assert df.loc[0, "dim3"] == pytest.approx(3.0)
        assert df.loc[1, "dim3"] == pytest.approx(6.0)

    def test_tag_column_auto(self) -> None:
        """Tags are auto-derived from names when not supplied."""
        df = build_result_dataframe(
            names=["antibody_targetX", "plain"],
            embedding=np.array([[1.0, 2.0], [3.0, 4.0]]),
            labels=np.array([0, 1]),
            hotspot_scores=[0.5, 0.8],
            cdr_sequences=[
                {"CDR-H1": "AA", "CDR-H2": "BB", "CDR-H3": "CC"},
                {"CDR-H1": "DD", "CDR-H2": "EE", "CDR-H3": "FF"},
            ],
        )
        assert "tag" in df.columns
        assert df.loc[0, "tag"] == "targetX"
        assert df.loc[1, "tag"] == ""

    def test_tag_column_explicit(self) -> None:
        """Explicit tags override auto-derived ones."""
        df = build_result_dataframe(
            names=["antibody_targetX", "plain"],
            embedding=np.array([[1.0, 2.0], [3.0, 4.0]]),
            labels=np.array([0, 1]),
            hotspot_scores=[0.5, 0.8],
            cdr_sequences=[
                {"CDR-H1": "AA", "CDR-H2": "BB", "CDR-H3": "CC"},
                {"CDR-H1": "DD", "CDR-H2": "EE", "CDR-H3": "FF"},
            ],
            tags=["custom1", "custom2"],
        )
        assert df.loc[0, "tag"] == "custom1"
        assert df.loc[1, "tag"] == "custom2"


class TestExtractTag:
    def test_single_underscore(self) -> None:
        assert extract_tag("antibody_targetX") == "targetX"

    def test_multiple_underscores(self) -> None:
        assert extract_tag("a_b_c") == "b_c"

    def test_no_underscore(self) -> None:
        assert extract_tag("mystructure") == ""

    def test_trailing_underscore(self) -> None:
        assert extract_tag("name_") == ""

    def test_leading_underscore(self) -> None:
        assert extract_tag("_tag") == "tag"


class TestExportCsv:
    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        return build_result_dataframe(
            names=["struct_a", "struct_b"],
            embedding=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            labels=np.array([0, 1]),
            hotspot_scores=[0.5, 0.8],
            cdr_sequences=[
                {"CDR-H1": "AA", "CDR-H2": "BB", "CDR-H3": "CC"},
                {"CDR-H1": "DD", "CDR-H2": "EE", "CDR-H3": "FF"},
            ],
        )

    def test_creates_csv_file(self, sample_df: pd.DataFrame, tmp_path) -> None:
        out = export_csv(sample_df, tmp_path / "out.csv")
        assert out.exists()
        assert out.suffix == ".csv"

    def test_csv_content_matches(self, sample_df: pd.DataFrame, tmp_path) -> None:
        export_csv(sample_df, tmp_path / "out.csv")
        loaded = pd.read_csv(tmp_path / "out.csv")
        assert list(loaded.columns) == list(sample_df.columns)
        assert len(loaded) == len(sample_df)
        assert loaded["structure"].tolist() == ["struct_a", "struct_b"]
        assert loaded["cluster"].tolist() == [0, 1]

    def test_no_index_by_default(self, sample_df: pd.DataFrame, tmp_path) -> None:
        export_csv(sample_df, tmp_path / "out.csv")
        loaded = pd.read_csv(tmp_path / "out.csv")
        assert "Unnamed: 0" not in loaded.columns

    def test_include_index(self, sample_df: pd.DataFrame, tmp_path) -> None:
        export_csv(sample_df, tmp_path / "out.csv", include_index=True)
        loaded = pd.read_csv(tmp_path / "out.csv")
        assert "Unnamed: 0" in loaded.columns or loaded.columns[0] != "structure"

    def test_creates_parent_directories(self, sample_df: pd.DataFrame, tmp_path) -> None:
        nested = tmp_path / "a" / "b" / "results.csv"
        out = export_csv(sample_df, nested)
        assert out.exists()

    def test_returns_resolved_path(self, sample_df: pd.DataFrame, tmp_path) -> None:
        out = export_csv(sample_df, tmp_path / "out.csv")
        assert out.is_absolute()

    def test_string_path_accepted(self, sample_df: pd.DataFrame, tmp_path) -> None:
        out = export_csv(sample_df, str(tmp_path / "out.csv"))
        assert out.exists()
