"""CDR annotation for VHH (single-domain antibody) sequences.

Uses IMGT numbering via ANARCI_ from the Oxford Protein Informatics Group
(OPIG).  ANARCI (Antibody Numbering and Antigen Receptor ClassIfication) is
the canonical OPIG tool for antibody numbering and provides accurate IMGT,
Chothia, Kabat, Martin, and Aho numbering schemes.

VHH molecules lack a light chain, so only the heavy-chain CDRs (CDR-H1,
CDR-H2, CDR-H3) are annotated.  IMGT boundaries applied:
Uses IMGT numbering via the ``abnumber`` library.  VHH molecules lack a
light chain, so only the heavy-chain CDRs (CDR-H1, CDR-H2, CDR-H3) are
annotated.  IMGT boundaries applied:

* CDR-H1 : positions 27-38
* CDR-H2 : positions 56-65
* CDR-H3 : positions 105-117

Falls back to ``abnumber`` or a simple positional heuristic when ANARCI
is not installed.

.. _ANARCI: https://github.com/oxpig/ANARCI

References
----------
Dunbar, J. and Deane, C.M., 2016. ANARCI: antigen receptor numbering and
receptor classification. *Bioinformatics*, 32(2), pp.298-300.
Framework regions are everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from vhh_clustering.parsing import ParsedStructure, Residue

# IMGT CDR boundaries for VH (heavy chain)
IMGT_CDR_RANGES: dict[str, tuple[int, int]] = {
    "CDR-H1": (27, 38),
    "CDR-H2": (56, 65),
    "CDR-H3": (105, 117),
}

# ---------------------------------------------------------------------------
# Backend detection: prefer ANARCI (OPIG), fall back to abnumber, then to a
# simple positional heuristic.
# ---------------------------------------------------------------------------

_BACKEND = "positional"  # default fallback

try:
    # ANARCII v2 – the latest OPIG antibody numbering tool
    from anarcii import Anarcii as _Anarcii

    _BACKEND = "anarci"
except ImportError:
    _Anarcii = None  # type: ignore[assignment,misc]
    try:
        # Legacy ANARCI v1
        from anarci import anarci as _anarci_fn

        _BACKEND = "anarci"
    except ImportError:
        _anarci_fn = None  # type: ignore[assignment]
        try:
            from abnumber import Chain as AbChain

            _BACKEND = "abnumber"
        except ImportError:
            pass
# Try to use abnumber for accurate IMGT renumbering; fall back to a
# simple positional heuristic if it is not installed or numbering fails.
try:
    from abnumber import Chain as AbChain

    _HAS_ABNUMBER = True
except ImportError:
    _HAS_ABNUMBER = False


@dataclass
class AnnotatedResidue:
    """Residue augmented with CDR / framework annotation."""

    residue: Residue
    imgt_position: Optional[int] = None
    region: str = "framework"  # e.g. "CDR-H1", "CDR-H2", "CDR-H3", "framework"


def _classify_imgt_position(pos: int) -> str:
    """Return CDR label for an IMGT position, or 'framework'."""
    for name, (start, end) in IMGT_CDR_RANGES.items():
        if start <= pos <= end:
            return name
    return "framework"


def _number_with_anarci(sequence: str, scheme: str = "imgt") -> dict[int, int]:
    """Use ANARCI (OPIG) to obtain IMGT numbering for a VHH sequence.

    Supports both ANARCII v2 and legacy ANARCI v1.

    Returns a mapping from 0-based sequence index → IMGT position number.
    """
    if _Anarcii is not None:
        # ANARCII v2 API
        numberer = _Anarcii()
        results = numberer.number([sequence])
        if not results:
            return {}
        key = list(results.keys())[0]
        entry = results[key]
        if entry.get("error") is not None:
            return {}
        numbered = entry["numbering"]

        # Convert to scheme if needed
        if scheme != entry.get("scheme", "imgt"):
            converted = numberer.to_scheme(scheme)
            if converted:
                key = list(converted.keys())[0]
                entry = converted[key]
                numbered = entry["numbering"]
    else:
        # Legacy ANARCI v1 API
        results = _anarci_fn([("VHH", sequence)], scheme=scheme, output=False)
        numbering_results, _alignment_details = results
        if numbering_results[0] is None:
            return {}
        numbered = numbering_results[0][0][0]

    imgt_positions: dict[int, int] = {}
    idx = 0
    for (pos_num, _insertion), aa in numbered:
        if aa != "-":
            imgt_positions[idx] = pos_num
            idx += 1
    return imgt_positions


def _number_with_abnumber(sequence: str) -> dict[int, int]:
    """Use abnumber to obtain IMGT numbering for a VHH sequence.

    Returns a mapping from 0-based sequence index → IMGT position number.
    """
    from abnumber import Chain as AbChain

    imgt_positions: dict[int, int] = {}
    try:
        chain = AbChain(sequence, scheme="imgt", chain_type="H")
        idx = 0
        for pos, aa in chain:
            if aa != "-":
                imgt_positions[idx] = pos.number
                idx += 1
    except Exception:
        pass
    return imgt_positions


def annotate_cdrs(structure: ParsedStructure) -> list[AnnotatedResidue]:
    """Annotate residues with CDR regions using IMGT numbering.

    Numbering backends (in order of preference):

    1. **ANARCI** (OPIG) – the gold-standard OPIG antibody numbering tool.
    2. **abnumber** – lightweight Python wrapper (uses IMGT scheme).
    3. **Positional fallback** – uses the residue sequence number from the
       input file when no numbering library is available.
    """
    annotated: list[AnnotatedResidue] = []
    imgt_positions: dict[int, int] = {}

    if structure.sequence:
        if _BACKEND == "anarci":
            imgt_positions = _number_with_anarci(structure.sequence)
        elif _BACKEND == "abnumber":
            imgt_positions = _number_with_abnumber(structure.sequence)
        # else: positional fallback (imgt_positions stays empty)
def annotate_cdrs(structure: ParsedStructure) -> list[AnnotatedResidue]:
    """Annotate residues with CDR regions using IMGT numbering.

    Attempts ``abnumber`` renumbering first; falls back to a sequential
    positional approximation when renumbering fails (common with
    predicted structures that have non-standard sequences).
    """
    annotated: list[AnnotatedResidue] = []
    imgt_positions: dict[int, int] = {}  # original index -> IMGT position

    if _HAS_ABNUMBER and structure.sequence:
        try:
            chain = AbChain(
                structure.sequence, scheme="imgt", chain_type="H"
            )
            # Build mapping: 0-based index in sequence -> IMGT position int
            idx = 0
            for pos, aa in chain:
                if aa != "-":
                    imgt_positions[idx] = pos.number
                    idx += 1
        except Exception:
            # Renumbering failed; proceed with fallback
            imgt_positions = {}

    for i, res in enumerate(structure.residues):
        if i in imgt_positions:
            imgt_pos = imgt_positions[i]
        else:
            # Fallback: use the residue sequence number from the file
            imgt_pos = res.res_seq
        region = _classify_imgt_position(imgt_pos)
        annotated.append(
            AnnotatedResidue(residue=res, imgt_position=imgt_pos, region=region)
        )
    return annotated


def numbering_backend() -> str:
    """Return the name of the active numbering backend.

    Useful for diagnostics and UI display.
    """
    return _BACKEND
