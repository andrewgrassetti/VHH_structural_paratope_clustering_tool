"""VHH structure prediction using ImmuneBuilder/NanoBodyBuilder2.

This module wraps NanoBodyBuilder2 from the ImmuneBuilder package
(OPIG, http://opig.stats.ox.ac.uk/) to predict 3D structures of
VHH/nanobody sequences.

ImmuneBuilder reference:
    Abanades, B., Wong, W.K., Boyles, F., Georges, G., Bujotzek, A.
    and Deane, C.M., 2023. ImmuneBuilder: Deep-Learning models for
    predicting the structures of immune proteins. Communications
    Biology, 6(1), p.575.
"""

import os
import tempfile

from Bio.PDB import PDBParser

try:
    from ImmuneBuilder import NanoBodyBuilder2
except ImportError:
    NanoBodyBuilder2 = None


_predictor = None


def _get_predictor():
    """Return a cached NanoBodyBuilder2 predictor instance."""
    global _predictor
    if _predictor is None:
        if NanoBodyBuilder2 is None:
            raise ImportError(
                "ImmuneBuilder is required for VHH structure prediction. "
                "Install it with: pip install ImmuneBuilder "
                "(see https://github.com/oxpig/ImmuneBuilder)"
            )
        _predictor = NanoBodyBuilder2()
    return _predictor


def predict_vhh_structure(sequence, output_path=None):
    """Predict the 3D structure of a VHH/nanobody from its sequence.

    Uses NanoBodyBuilder2 from the OPIG ImmuneBuilder package to predict
    the structure. The model produces state-of-the-art accuracy for
    nanobody CDR loop prediction.

    Parameters
    ----------
    sequence : str
        Amino acid sequence of the VHH/nanobody.
    output_path : str, optional
        Path to save the predicted PDB file. If None, a temporary file
        is used and deleted after parsing.

    Returns
    -------
    Bio.PDB.Structure.Structure
        BioPython Structure object of the predicted nanobody.
    """
    predictor = _get_predictor()
    nanobody = predictor.predict({"H": sequence})

    if output_path is not None:
        nanobody.save(output_path)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("nanobody", output_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            nanobody.save(tmp_path)
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("nanobody", tmp_path)
        finally:
            os.unlink(tmp_path)

    return structure


def predict_vhh_structures(sequences, output_dir=None):
    """Predict structures for multiple VHH/nanobody sequences.

    Parameters
    ----------
    sequences : dict
        Dictionary mapping sequence identifiers to amino acid sequences.
    output_dir : str, optional
        Directory to save predicted PDB files. Files are named
        ``{identifier}.pdb``. If None, structures are not saved to disk.

    Returns
    -------
    dict
        Dictionary mapping identifiers to BioPython Structure objects.
    """
    structures = {}
    for name, seq in sequences.items():
        out_path = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{name}.pdb")
        structures[name] = predict_vhh_structure(seq, output_path=out_path)
    return structures
