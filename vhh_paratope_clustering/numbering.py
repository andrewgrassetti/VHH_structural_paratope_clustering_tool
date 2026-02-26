"""VHH sequence numbering using ANARCI.

This module wraps ANARCI (Antibody Numbering and Antigen Receptor
ClassIfication) from OPIG (http://opig.stats.ox.ac.uk/) to number
VHH/nanobody sequences and identify CDR regions.

ANARCI reference:
    Dunbar, J. and Deane, C.M., 2016. ANARCI: antigen receptor numbering
    and receptor classification. Bioinformatics, 32(2), pp.298-300.
"""

try:
    from anarci import anarci as _anarci_fn, number
except ImportError:
    _anarci_fn = None
    number = None


# IMGT CDR definitions for heavy chain (used by VHH/nanobodies)
IMGT_CDR_RANGES = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}


def number_vhh_sequence(sequence, scheme="imgt"):
    """Number a VHH/nanobody sequence using ANARCI.

    Parameters
    ----------
    sequence : str
        Amino acid sequence of the VHH/nanobody.
    scheme : str, optional
        Numbering scheme to use. One of 'imgt', 'chothia', 'kabat',
        'martin', or 'aho'. Default is 'imgt'.

    Returns
    -------
    list of tuple
        List of ((chain_type, position_tuple), amino_acid) pairs from
        ANARCI numbering. position_tuple is (number, insertion_code).

    Raises
    ------
    ValueError
        If ANARCI fails to number the sequence (e.g. not a valid
        antibody/nanobody sequence).
    """
    if _anarci_fn is None:
        raise ImportError(
            "ANARCI is required for VHH numbering. Install it with: "
            "pip install ANARCI (see https://github.com/oxpig/ANARCI)"
        )
    sequences = [("VHH", sequence)]
    results = _anarci_fn(sequences, scheme=scheme, output=False)
    numbering_results, alignment_details = results

    if numbering_results[0] is None:
        raise ValueError(
            "ANARCI could not number the provided sequence. "
            "Ensure it is a valid VHH/nanobody sequence."
        )

    # numbering_results[0] is a list of domain results for the first sequence
    # Each domain result is a list of (position_tuple, amino_acid) pairs
    numbered = numbering_results[0][0][0]
    return numbered


def get_cdr_residues(numbered_sequence, scheme="imgt", cdr_definition=None):
    """Extract CDR residues from a numbered VHH sequence.

    Parameters
    ----------
    numbered_sequence : list of tuple
        Output from ``number_vhh_sequence``: list of
        (position_tuple, amino_acid) pairs.
    scheme : str, optional
        Numbering scheme used. Default is 'imgt'.
    cdr_definition : dict, optional
        Custom CDR definitions mapping CDR names to (start, end) position
        ranges (inclusive). If None, uses IMGT definitions.

    Returns
    -------
    dict
        Dictionary mapping CDR names ('CDR1', 'CDR2', 'CDR3') to lists of
        (position_tuple, amino_acid) pairs for residues in each CDR.
    """
    if cdr_definition is None:
        cdr_definition = IMGT_CDR_RANGES

    cdrs = {name: [] for name in cdr_definition}

    for position, amino_acid in numbered_sequence:
        if amino_acid == "-":
            continue
        pos_num = position[0]
        for cdr_name, (start, end) in cdr_definition.items():
            if start <= pos_num <= end:
                cdrs[cdr_name].append((position, amino_acid))
                break

    return cdrs


def get_cdr_sequences(numbered_sequence, scheme="imgt", cdr_definition=None):
    """Extract CDR amino acid sequences from a numbered VHH sequence.

    Parameters
    ----------
    numbered_sequence : list of tuple
        Output from ``number_vhh_sequence``.
    scheme : str, optional
        Numbering scheme used. Default is 'imgt'.
    cdr_definition : dict, optional
        Custom CDR definitions. If None, uses IMGT definitions.

    Returns
    -------
    dict
        Dictionary mapping CDR names to amino acid sequence strings.
    """
    cdrs = get_cdr_residues(numbered_sequence, scheme=scheme,
                            cdr_definition=cdr_definition)
    return {name: "".join(aa for _, aa in residues)
            for name, residues in cdrs.items()}
