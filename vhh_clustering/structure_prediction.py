"""VHH structure prediction using ImmuneBuilder / NanoBodyBuilder2.

Integrates NanoBodyBuilder2_ from the OPIG ImmuneBuilder package to predict
3D structures of VHH (nanobody) sequences.  This enables a **sequence-first
workflow**: users can provide amino-acid sequences instead of pre-computed
PDB files, and the tool will predict structures on the fly.

NanoBodyBuilder2 achieves state-of-the-art accuracy for nanobody CDR loop
prediction (CDR-H3 RMSD 2.89 Å, 0.55 Å improvement over AlphaFold2).

.. _NanoBodyBuilder2: https://github.com/oxpig/ImmuneBuilder

References
----------
Abanades, B., Wong, W.K., Boyles, F., Georges, G., Bujotzek, A. and
Deane, C.M., 2023. ImmuneBuilder: Deep-Learning models for predicting the
structures of immune proteins. *Communications Biology*, 6(1), p.575.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from vhh_clustering.parsing import ParsedStructure, parse_structure

try:
    from ImmuneBuilder import NanoBodyBuilder2

    _HAS_IMMUNEBUILDER = True
except ImportError:
    NanoBodyBuilder2 = None  # type: ignore[assignment,misc]
    _HAS_IMMUNEBUILDER = False

_predictor = None


def immunebuilder_available() -> bool:
    """Return ``True`` if ImmuneBuilder is installed and importable."""
    return _HAS_IMMUNEBUILDER


def _get_predictor():
    """Return a cached NanoBodyBuilder2 predictor instance."""
    global _predictor
    if _predictor is None:
        if not _HAS_IMMUNEBUILDER:
            raise ImportError(
                "ImmuneBuilder is required for VHH structure prediction. "
                "Install it with:  pip install ImmuneBuilder\n"
                "See https://github.com/oxpig/ImmuneBuilder"
            )
        _predictor = NanoBodyBuilder2()
    return _predictor


def predict_structure(sequence: str, output_path: str | Path | None = None) -> ParsedStructure:
    """Predict the 3D structure of a VHH/nanobody from its amino-acid sequence.

    Uses NanoBodyBuilder2 from the OPIG ImmuneBuilder package.

    Parameters
    ----------
    sequence : str
        Amino acid sequence of the VHH/nanobody.
    output_path : str or Path, optional
        Path to save the predicted PDB file.  If *None*, a temporary file
        is used and deleted after parsing.

    Returns
    -------
    ParsedStructure
        Parsed structure ready for CDR annotation and feature extraction.
    """
    predictor = _get_predictor()
    nanobody = predictor.predict({"H": sequence})

    if output_path is not None:
        output_path = Path(output_path)
        nanobody.save(str(output_path))
        return parse_structure(output_path)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
    try:
        os.close(tmp_fd)
        nanobody.save(tmp_path)
        return parse_structure(tmp_path)
    finally:
        os.unlink(tmp_path)


def predict_structures(sequences: dict[str, str], output_dir: str | Path | None = None) -> dict[str, ParsedStructure]:
    """Predict structures for multiple VHH/nanobody sequences.

    Parameters
    ----------
    sequences : dict
        Mapping of identifier → amino acid sequence.
    output_dir : str or Path, optional
        Directory to save predicted PDB files (named ``{id}.pdb``).

    Returns
    -------
    dict
        Mapping of identifier → ``ParsedStructure``.
    """
    results: dict[str, ParsedStructure] = {}
    for name, seq in sequences.items():
        out_path = None
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{name}.pdb"
        results[name] = predict_structure(seq, output_path=out_path)
    return results
