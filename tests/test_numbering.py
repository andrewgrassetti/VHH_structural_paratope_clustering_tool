"""Tests for the numbering module (ANARCI integration)."""

from unittest.mock import patch, MagicMock

import pytest

from vhh_paratope_clustering.numbering import (
    number_vhh_sequence,
    get_cdr_residues,
    get_cdr_sequences,
    IMGT_CDR_RANGES,
)

# Example VHH sequence (anti-GFP nanobody)
EXAMPLE_VHH_SEQUENCE = (
    "QVQLVESGGGLVQPGESLRLSCAASGSIFGIYAVHWFRMAPGKEREFTAGFGSHGSTNYA"
    "ASVKGRFTMSRDNAKNTTYLQMNSLKPADTAVYYCHALIKNELGFLDYWGPGTQVTVSS"
)


def _make_mock_numbered():
    """Create mock ANARCI numbering output for testing."""
    # Simulated IMGT numbering positions and amino acids
    numbered = [
        ((1, " "), "Q"),
        ((2, " "), "V"),
        ((3, " "), "Q"),
        ((4, " "), "L"),
        ((5, " "), "V"),
        # ... framework 1 ...
        ((27, " "), "G"),  # CDR1 start
        ((28, " "), "S"),
        ((29, " "), "I"),
        ((30, " "), "F"),
        ((31, " "), "G"),
        ((32, " "), "I"),
        ((33, " "), "Y"),
        ((34, " "), "A"),
        ((35, " "), "V"),
        ((36, " "), "H"),
        ((37, " "), "W"),
        ((38, " "), "F"),  # CDR1 end
        # ... framework 2 ...
        ((45, " "), "R"),
        ((56, " "), "T"),  # CDR2 start
        ((57, " "), "A"),
        ((58, " "), "G"),
        ((59, " "), "F"),
        ((60, " "), "G"),
        ((61, " "), "S"),
        ((62, " "), "H"),
        ((63, " "), "G"),
        ((64, " "), "S"),
        ((65, " "), "T"),  # CDR2 end
        # ... framework 3 ...
        ((80, " "), "A"),
        ((105, " "), "H"),  # CDR3 start
        ((106, " "), "A"),
        ((107, " "), "L"),
        ((108, " "), "I"),
        ((109, " "), "K"),
        ((110, " "), "N"),
        ((111, " "), "E"),
        ((112, " "), "L"),
        ((113, " "), "G"),
        ((114, " "), "F"),
        ((115, " "), "L"),
        ((116, " "), "D"),
        ((117, " "), "Y"),  # CDR3 end
        # ... framework 4 ...
        ((118, " "), "W"),
        ((119, " "), "G"),
        ((120, " "), "-"),
    ]
    return numbered


class TestNumberVhhSequence:
    """Tests for number_vhh_sequence function."""

    @patch("vhh_paratope_clustering.numbering._anarci_fn")
    def test_successful_numbering(self, mock_anarci):
        """Test that a valid VHH sequence is numbered correctly."""
        mock_numbered = _make_mock_numbered()
        # ANARCI returns (numbering_results, alignment_details)
        mock_anarci.return_value = (
            [[[mock_numbered, "H"]]],  # numbering_results
            [{"chain_type": "H", "species": "camelid"}],  # alignment_details
        )

        result = number_vhh_sequence(EXAMPLE_VHH_SEQUENCE, scheme="imgt")

        assert result == mock_numbered
        mock_anarci.assert_called_once()

    @patch("vhh_paratope_clustering.numbering._anarci_fn")
    def test_invalid_sequence_raises(self, mock_anarci):
        """Test that an invalid sequence raises ValueError."""
        mock_anarci.return_value = ([None], [None])

        with pytest.raises(ValueError, match="could not number"):
            number_vhh_sequence("INVALIDSEQUENCE")


class TestGetCdrResidues:
    """Tests for get_cdr_residues function."""

    def test_imgt_cdr_extraction(self):
        """Test extraction of CDR residues using IMGT definitions."""
        numbered = _make_mock_numbered()
        cdrs = get_cdr_residues(numbered)

        assert "CDR1" in cdrs
        assert "CDR2" in cdrs
        assert "CDR3" in cdrs

        # CDR1 positions 27-38
        cdr1_positions = [pos[0] for pos, _ in cdrs["CDR1"]]
        assert all(27 <= p <= 38 for p in cdr1_positions)
        assert len(cdrs["CDR1"]) == 12

        # CDR2 positions 56-65
        cdr2_positions = [pos[0] for pos, _ in cdrs["CDR2"]]
        assert all(56 <= p <= 65 for p in cdr2_positions)
        assert len(cdrs["CDR2"]) == 10

        # CDR3 positions 105-117
        cdr3_positions = [pos[0] for pos, _ in cdrs["CDR3"]]
        assert all(105 <= p <= 117 for p in cdr3_positions)
        assert len(cdrs["CDR3"]) == 13

    def test_gaps_are_skipped(self):
        """Test that gap characters are excluded from CDR residues."""
        numbered = _make_mock_numbered()
        cdrs = get_cdr_residues(numbered)

        for cdr_residues in cdrs.values():
            for _, aa in cdr_residues:
                assert aa != "-"

    def test_custom_cdr_definition(self):
        """Test using a custom CDR definition."""
        numbered = _make_mock_numbered()
        custom_def = {"my_region": (1, 5)}
        result = get_cdr_residues(numbered, cdr_definition=custom_def)

        assert "my_region" in result
        assert "CDR1" not in result
        positions = [pos[0] for pos, _ in result["my_region"]]
        assert all(1 <= p <= 5 for p in positions)


class TestGetCdrSequences:
    """Tests for get_cdr_sequences function."""

    def test_returns_strings(self):
        """Test that CDR sequences are returned as strings."""
        numbered = _make_mock_numbered()
        seqs = get_cdr_sequences(numbered)

        assert isinstance(seqs["CDR1"], str)
        assert isinstance(seqs["CDR2"], str)
        assert isinstance(seqs["CDR3"], str)
        assert seqs["CDR1"] == "GSIFGIYAVHWF"
        assert seqs["CDR2"] == "TAGFGSHGST"
        assert seqs["CDR3"] == "HALIKNELGFLDY"
