"""Tests for the structure module (ImmuneBuilder/NanoBodyBuilder2 integration)."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from vhh_paratope_clustering.structure import (
    predict_vhh_structure,
    predict_vhh_structures,
)

EXAMPLE_VHH_SEQUENCE = (
    "QVQLVESGGGLVQPGESLRLSCAASGSIFGIYAVHWFRMAPGKEREFTAGFGSHGSTNYA"
    "ASVKGRFTMSRDNAKNTTYLQMNSLKPADTAVYYCHALIKNELGFLDYWGPGTQVTVSS"
)


def _make_mock_pdb_content():
    """Create minimal valid PDB content for testing."""
    return (
        "ATOM      1  CA  ALA H   1       1.000   2.000   3.000  1.00  0.00\n"
        "ATOM      2  CA  GLY H   2       4.000   5.000   6.000  1.00  0.00\n"
        "END\n"
    )


class TestPredictVhhStructure:
    """Tests for predict_vhh_structure function."""

    @patch("vhh_paratope_clustering.structure._get_predictor")
    def test_predict_returns_structure(self, mock_get_predictor):
        """Test that prediction returns a BioPython Structure object."""
        mock_predictor = MagicMock()
        mock_nanobody = MagicMock()

        def mock_save(path):
            with open(path, "w") as f:
                f.write(_make_mock_pdb_content())

        mock_nanobody.save = mock_save
        mock_predictor.predict.return_value = mock_nanobody
        mock_get_predictor.return_value = mock_predictor

        structure = predict_vhh_structure(EXAMPLE_VHH_SEQUENCE)

        assert structure is not None
        mock_predictor.predict.assert_called_once_with(
            {"H": EXAMPLE_VHH_SEQUENCE}
        )

    @patch("vhh_paratope_clustering.structure._get_predictor")
    def test_predict_saves_to_file(self, mock_get_predictor):
        """Test that structure is saved when output_path is given."""
        mock_predictor = MagicMock()
        mock_nanobody = MagicMock()

        def mock_save(path):
            with open(path, "w") as f:
                f.write(_make_mock_pdb_content())

        mock_nanobody.save = mock_save
        mock_predictor.predict.return_value = mock_nanobody
        mock_get_predictor.return_value = mock_predictor

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            structure = predict_vhh_structure(
                EXAMPLE_VHH_SEQUENCE, output_path=tmp_path
            )
            assert os.path.exists(tmp_path)
            assert structure is not None
        finally:
            os.unlink(tmp_path)


class TestPredictVhhStructures:
    """Tests for predict_vhh_structures function."""

    @patch("vhh_paratope_clustering.structure._get_predictor")
    def test_predict_multiple(self, mock_get_predictor):
        """Test predicting multiple structures."""
        mock_predictor = MagicMock()
        mock_nanobody = MagicMock()

        def mock_save(path):
            with open(path, "w") as f:
                f.write(_make_mock_pdb_content())

        mock_nanobody.save = mock_save
        mock_predictor.predict.return_value = mock_nanobody
        mock_get_predictor.return_value = mock_predictor

        sequences = {
            "nb1": EXAMPLE_VHH_SEQUENCE,
            "nb2": EXAMPLE_VHH_SEQUENCE,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            structures = predict_vhh_structures(
                sequences, output_dir=tmpdir
            )
            assert len(structures) == 2
            assert "nb1" in structures
            assert "nb2" in structures
            assert os.path.exists(os.path.join(tmpdir, "nb1.pdb"))
            assert os.path.exists(os.path.join(tmpdir, "nb2.pdb"))
