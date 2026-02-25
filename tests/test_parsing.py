"""Unit tests for structure parsing and CDR annotation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vhh_clustering.parsing import Residue, ParsedStructure, parse_structure

# ---------------------------------------------------------------------------
# Helpers – minimal valid PDB content
# ---------------------------------------------------------------------------

MINIMAL_PDB = """\
HEADER    TEST VHH
ATOM      1  N   GLY A   1       1.000   2.000   3.000  1.00 30.00           N
ATOM      2  CA  GLY A   1       2.000   3.000   4.000  1.00 30.00           C
ATOM      3  C   GLY A   1       3.000   4.000   5.000  1.00 30.00           C
ATOM      4  O   GLY A   1       4.000   5.000   6.000  1.00 30.00           O
ATOM      5  N   ALA A   2       5.000   6.000   7.000  1.00 25.00           N
ATOM      6  CA  ALA A   2       6.000   7.000   8.000  1.00 25.00           C
ATOM      7  C   ALA A   2       7.000   8.000   9.000  1.00 25.00           C
ATOM      8  O   ALA A   2       8.000   9.000  10.000  1.00 25.00           O
ATOM      9  CB  ALA A   2       6.500   7.500   8.500  1.00 25.00           C
ATOM     10  N   ARG A   3       9.000  10.000  11.000  1.00 20.00           N
ATOM     11  CA  ARG A   3      10.000  11.000  12.000  1.00 20.00           C
ATOM     12  C   ARG A   3      11.000  12.000  13.000  1.00 20.00           C
END
"""


@pytest.fixture()
def pdb_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.pdb"
    p.write_text(MINIMAL_PDB)
    return p


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_pdb_residue_count(self, pdb_file: Path) -> None:
        result = parse_structure(pdb_file)
        assert len(result.residues) == 3

    def test_parse_pdb_sequence(self, pdb_file: Path) -> None:
        result = parse_structure(pdb_file)
        assert result.sequence == "GAR"

    def test_residue_ca_coord(self, pdb_file: Path) -> None:
        result = parse_structure(pdb_file)
        ca = result.residues[0].ca_coord
        assert ca is not None
        np.testing.assert_allclose(ca, [2.0, 3.0, 4.0])

    def test_residue_b_factor(self, pdb_file: Path) -> None:
        result = parse_structure(pdb_file)
        assert result.residues[0].b_factor == pytest.approx(30.0, abs=0.1)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.xyz"
        p.write_text("junk")
        with pytest.raises(ValueError, match="Unsupported"):
            parse_structure(p)


# ---------------------------------------------------------------------------
# CDR annotation tests
# ---------------------------------------------------------------------------

class TestCDRAnnotation:
    """Test CDR annotation with known VHH-like sequences."""

    def test_framework_classification(self, pdb_file: Path) -> None:
        """Residues at low sequence numbers should be framework."""
        from vhh_clustering.cdr_annotation import annotate_cdrs

        structure = parse_structure(pdb_file)
        annotated = annotate_cdrs(structure)
        # res_seq 1, 2, 3 are all < 27 → framework
        for ar in annotated:
            assert ar.region == "framework"

    def test_cdr_h1_classification(self) -> None:
        from vhh_clustering.cdr_annotation import _classify_imgt_position

        assert _classify_imgt_position(30) == "CDR-H1"
        assert _classify_imgt_position(27) == "CDR-H1"
        assert _classify_imgt_position(38) == "CDR-H1"

    def test_cdr_h2_classification(self) -> None:
        from vhh_clustering.cdr_annotation import _classify_imgt_position

        assert _classify_imgt_position(56) == "CDR-H2"
        assert _classify_imgt_position(60) == "CDR-H2"

    def test_cdr_h3_classification(self) -> None:
        from vhh_clustering.cdr_annotation import _classify_imgt_position

        assert _classify_imgt_position(110) == "CDR-H3"
        assert _classify_imgt_position(117) == "CDR-H3"

    def test_framework_position(self) -> None:
        from vhh_clustering.cdr_annotation import _classify_imgt_position

        assert _classify_imgt_position(1) == "framework"
        assert _classify_imgt_position(50) == "framework"
        assert _classify_imgt_position(130) == "framework"
