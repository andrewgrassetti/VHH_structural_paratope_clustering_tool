"""Unit tests for the structure prediction module (ImmuneBuilder integration)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from vhh_clustering.structure_prediction import (
    immunebuilder_available,
    predict_structure,
    predict_structures,
)

# Minimal valid PDB content for mock predictions
_MOCK_PDB = """\
ATOM      1  N   ALA H   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA H   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  C   ALA H   1       3.000   4.000   5.000  1.00  0.00           C
ATOM      4  O   ALA H   1       4.000   5.000   6.000  1.00  0.00           O
ATOM      5  N   GLY H   2       5.000   6.000   7.000  1.00  0.00           N
ATOM      6  CA  GLY H   2       6.000   7.000   8.000  1.00  0.00           C
ATOM      7  C   GLY H   2       7.000   8.000   9.000  1.00  0.00           C
ATOM      8  O   GLY H   2       8.000   9.000  10.000  1.00  0.00           O
END
"""

EXAMPLE_VHH = (
    "QVQLVESGGGLVQPGESLRLSCAASGSIFGIYAVHWFRMAPGKEREFTAGFGSHGSTNYA"
    "ASVKGRFTMSRDNAKNTTYLQMNSLKPADTAVYYCHALIKNELGFLDYWGPGTQVTVSS"
)


class TestImmuneBuilderAvailable:
    def test_returns_bool(self) -> None:
        result = immunebuilder_available()
        assert isinstance(result, bool)


class TestPredictStructure:
    @patch("vhh_clustering.structure_prediction._get_predictor")
    def test_predict_returns_parsed_structure(self, mock_get_predictor) -> None:
        """Prediction should return a ParsedStructure."""
        mock_predictor = MagicMock()
        mock_nanobody = MagicMock()

        def mock_save(path):
            with open(path, "w") as f:
                f.write(_MOCK_PDB)

        mock_nanobody.save = mock_save
        mock_predictor.predict.return_value = mock_nanobody
        mock_get_predictor.return_value = mock_predictor

        result = predict_structure(EXAMPLE_VHH)
        assert result is not None
        assert len(result.residues) == 2
        mock_predictor.predict.assert_called_once_with({"H": EXAMPLE_VHH})

    @patch("vhh_clustering.structure_prediction._get_predictor")
    def test_predict_saves_to_file(self, mock_get_predictor) -> None:
        """When output_path is given, the PDB file should be saved."""
        mock_predictor = MagicMock()
        mock_nanobody = MagicMock()

        def mock_save(path):
            with open(path, "w") as f:
                f.write(_MOCK_PDB)

        mock_nanobody.save = mock_save
        mock_predictor.predict.return_value = mock_nanobody
        mock_get_predictor.return_value = mock_predictor

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = predict_structure(EXAMPLE_VHH, output_path=tmp_path)
            assert os.path.exists(tmp_path)
            assert result is not None
        finally:
            os.unlink(tmp_path)


class TestPredictStructures:
    @patch("vhh_clustering.structure_prediction._get_predictor")
    def test_predict_multiple(self, mock_get_predictor) -> None:
        """Should predict structures for multiple sequences."""
        mock_predictor = MagicMock()
        mock_nanobody = MagicMock()

        def mock_save(path):
            with open(path, "w") as f:
                f.write(_MOCK_PDB)

        mock_nanobody.save = mock_save
        mock_predictor.predict.return_value = mock_nanobody
        mock_get_predictor.return_value = mock_predictor

        sequences = {"nb1": EXAMPLE_VHH, "nb2": EXAMPLE_VHH}

        with tempfile.TemporaryDirectory() as tmpdir:
            results = predict_structures(sequences, output_dir=tmpdir)
            assert len(results) == 2
            assert "nb1" in results
            assert "nb2" in results
            assert os.path.exists(os.path.join(tmpdir, "nb1.pdb"))
            assert os.path.exists(os.path.join(tmpdir, "nb2.pdb"))
