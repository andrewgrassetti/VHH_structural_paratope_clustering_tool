"""CDR annotation for VHH (single-domain antibody) sequences.

Uses IMGT numbering via the ``abnumber`` library.  VHH molecules lack a
light chain, so only the heavy-chain CDRs (CDR-H1, CDR-H2, CDR-H3) are
annotated.  IMGT boundaries applied:

* CDR-H1 : positions 27-38
* CDR-H2 : positions 56-65
* CDR-H3 : positions 105-117

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
