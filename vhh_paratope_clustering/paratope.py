"""Paratope identification for VHH/nanobody structures.

This module identifies paratope residues (the antibody residues that
contact the antigen) from numbered VHH structures. It uses ANARCI
numbering (via the ``numbering`` module) to locate CDR and
framework residues that are part of the binding interface.

For VHH/nanobodies, the paratope typically consists of residues from
CDR1, CDR2, and CDR3, and may include select framework residues.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from Bio.PDB import Selection


from vhh_paratope_clustering.numbering import (
    number_vhh_sequence,
    get_cdr_residues,
    IMGT_CDR_RANGES,
)


def identify_paratope_residues(numbered_sequence, scheme="imgt",
                               cdr_definition=None):
    """Identify candidate paratope residues from a numbered VHH sequence.

    Returns CDR residues which form the primary paratope region in VHH
    antibodies. For more precise paratope identification, structural
    analysis with an antigen complex is required.

    Parameters
    ----------
    numbered_sequence : list of tuple
        Output from ``number_vhh_sequence``: list of
        (position_tuple, amino_acid) pairs.
    scheme : str, optional
        Numbering scheme used. Default is 'imgt'.
    cdr_definition : dict, optional
        Custom CDR definitions. If None, uses IMGT definitions.

    Returns
    -------
    list of tuple
        List of (position_tuple, amino_acid) for all paratope residues,
        ordered by position number.
    """
    cdrs = get_cdr_residues(numbered_sequence, scheme=scheme,
                            cdr_definition=cdr_definition)
    paratope = []
    for cdr_residues in cdrs.values():
        paratope.extend(cdr_residues)

    paratope.sort(key=lambda x: (x[0][0], x[0][1]))
    return paratope


def extract_paratope_coordinates(structure, paratope_residues,
                                 atom_name="CA"):
    """Extract 3D coordinates for paratope residues from a structure.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        BioPython Structure object (e.g. from ``predict_vhh_structure``).
    paratope_residues : list of tuple
        List of (position_tuple, amino_acid) for paratope residues,
        as returned by ``identify_paratope_residues``.
    atom_name : str, optional
        Atom to extract coordinates for. Default is 'CA' (C-alpha).

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 3) with coordinates for each paratope residue
        that has the specified atom in the structure.
    """
    model = structure[0]
    chain = list(model.get_chains())[0]

    paratope_positions = {pos[0] for pos, _ in paratope_residues}

    coords = []
    for residue in chain.get_residues():
        res_id = residue.get_id()
        res_num = res_id[1]
        if res_num in paratope_positions and atom_name in residue:
            coords.append(residue[atom_name].get_vector().get_array())

    return np.array(coords) if coords else np.empty((0, 3))


def compute_paratope_distance_matrix(coords_list):
    """Compute pairwise RMSD between sets of paratope coordinates.

    Parameters
    ----------
    coords_list : list of numpy.ndarray
        List of coordinate arrays, each of shape (N, 3). All arrays
        should represent equivalent paratope positions (same length).

    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix of shape (M, M) where M is the number
        of coordinate sets.

    Raises
    ------
    ValueError
        If coordinate arrays have different numbers of residues.
    """
    n = len(coords_list)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if coords_list[i].shape != coords_list[j].shape:
                raise ValueError(
                    f"Coordinate arrays {i} and {j} have different shapes: "
                    f"{coords_list[i].shape} vs {coords_list[j].shape}. "
                    "Ensure all paratope coordinate sets have the same "
                    "number of residues."
                )
            rmsd = _compute_rmsd(coords_list[i], coords_list[j])
            dist_matrix[i, j] = rmsd
            dist_matrix[j, i] = rmsd

    return dist_matrix


def _compute_rmsd(coords1, coords2):
    """Compute RMSD between two sets of coordinates after optimal alignment.

    Uses the Kabsch algorithm (via scipy) for optimal superposition.

    Parameters
    ----------
    coords1 : numpy.ndarray
        Coordinates of shape (N, 3).
    coords2 : numpy.ndarray
        Coordinates of shape (N, 3).

    Returns
    -------
    float
        RMSD after optimal superposition.
    """
    # Center the coordinates
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)

    # Find optimal rotation from c2 to c1
    rotation, rssd = Rotation.align_vectors(c1, c2)
    # rssd is root sum of squared distances; convert to RMSD
    n = len(c1)
    return rssd / np.sqrt(n)
