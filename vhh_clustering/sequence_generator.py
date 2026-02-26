"""VHH CDR mutant sequence generator.

Generates a library of VHH mutant sequences based on the PDB 5U64 VHH
reference sequence.  Mutations are restricted to CDR-H1, CDR-H2, and
CDR-H3 loops and are *conservative* (biochemically similar substitutions)
to produce subtle variants suitable for structural clustering benchmarks.

The module can produce ~200 unique mutant sequences, each carrying 1–4
mutations spread across the three CDR loops.  Output is available as a
list of ``MutantSequence`` records or as a FASTA-formatted string.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# 5U64 VHH reference (anti-GFP enhancer nanobody, IMGT-numbered heavy chain)
# ---------------------------------------------------------------------------
REFERENCE_SEQUENCE: str = (
    "QVQLVESGGGLVQPGGSLRLSCAAS"   # Framework 1  (indices 0-24)
    "GFPVNRYS"                      # CDR-H1       (indices 25-32)
    "MRWYRQAPGKEREWVA"              # Framework 2  (indices 33-48)
    "G"                             # (index 49)
    "MSSAGDRS"                      # CDR-H2       (indices 50-57)
    "SYEDSVKGRFTISRDDARNTVYLQMNSLKPEDTAVYYC"  # Framework 3
    "NVNVGFEY"                      # CDR-H3       (indices 96-103)
    "WGQGTQVTVSS"                   # Framework 4  (indices 104-114)
)

# CDR boundaries as 0-based sequence indices (derived via IMGT/ANARCI).
CDR_INDICES: dict[str, list[int]] = {
    "CDR-H1": [25, 26, 27, 28, 29, 30, 31, 32],
    "CDR-H2": [50, 51, 52, 53, 54, 55, 56, 57],
    "CDR-H3": [96, 97, 98, 99, 100, 101, 102, 103],
}

# ---------------------------------------------------------------------------
# Conservative substitution table
# ---------------------------------------------------------------------------
# Each amino acid maps to a tuple of biochemically similar alternatives.
# Groups are derived from standard BLOSUM62-based similarity clusters:
#   Aliphatic : A V I L M
#   Aromatic  : F Y W
#   Positive  : R K H
#   Negative  : D E
#   Amide     : N Q
#   Hydroxy   : S T
#   Small     : G A S
#   Proline   : P (unique backbone – only mild substitutes)

CONSERVATIVE_SUBSTITUTIONS: dict[str, tuple[str, ...]] = {
    "A": ("V", "I", "L", "G", "S"),
    "V": ("A", "I", "L", "M"),
    "I": ("V", "L", "M", "A"),
    "L": ("I", "V", "M", "A"),
    "M": ("L", "I", "V"),
    "F": ("Y", "W"),
    "Y": ("F", "W"),
    "W": ("F", "Y"),
    "R": ("K", "H"),
    "K": ("R", "H"),
    "H": ("R", "K"),
    "D": ("E", "N"),
    "E": ("D", "Q"),
    "N": ("Q", "D", "S"),
    "Q": ("N", "E"),
    "S": ("T", "A", "N"),
    "T": ("S", "A"),
    "G": ("A", "S"),
    "P": ("A", "S"),
    "C": ("S", "A"),
}


@dataclass
class MutantSequence:
    """A mutant VHH sequence with metadata."""

    name: str
    sequence: str
    mutations: list[str] = field(default_factory=list)
    parent: str = "5U64_VHH"


def _pick_conservative_substitute(aa: str, rng: random.Random) -> str:
    """Return a conservative substitute for *aa*, or *aa* itself if none exist."""
    choices = CONSERVATIVE_SUBSTITUTIONS.get(aa, ())
    if not choices:
        return aa
    return rng.choice(choices)


def generate_mutants(
    *,
    n_mutants: int = 200,
    min_mutations: int = 1,
    max_mutations: int = 4,
    reference: str = REFERENCE_SEQUENCE,
    cdr_indices: dict[str, list[int]] | None = None,
    seed: int = 42,
) -> list[MutantSequence]:
    """Generate *n_mutants* unique conservative CDR mutants of *reference*.

    Parameters
    ----------
    n_mutants:
        Target number of unique mutant sequences to produce.
    min_mutations, max_mutations:
        Range of total mutations per variant (distributed across CDRs).
    reference:
        Parent VHH amino-acid sequence.
    cdr_indices:
        Mapping of CDR name → list of 0-based positions eligible for
        mutation.  Defaults to ``CDR_INDICES`` (5U64).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[MutantSequence]
        Up to *n_mutants* unique mutant records.
    """
    if cdr_indices is None:
        cdr_indices = CDR_INDICES

    rng = random.Random(seed)

    # Flatten all CDR positions into a single pool
    all_cdr_positions: list[tuple[str, int]] = []
    for cdr_name, indices in cdr_indices.items():
        for idx in indices:
            all_cdr_positions.append((cdr_name, idx))

    seen_sequences: set[str] = {reference}
    mutants: list[MutantSequence] = []

    # Allow generous retries to reach the target count
    max_attempts = n_mutants * 20
    attempts = 0

    while len(mutants) < n_mutants and attempts < max_attempts:
        attempts += 1
        n_muts = rng.randint(min_mutations, max_mutations)
        # Sample positions without replacement
        chosen = rng.sample(all_cdr_positions, min(n_muts, len(all_cdr_positions)))

        seq_list = list(reference)
        mutation_labels: list[str] = []

        for cdr_name, pos in chosen:
            original = reference[pos]
            substitute = _pick_conservative_substitute(original, rng)
            if substitute == original:
                continue
            seq_list[pos] = substitute
            mutation_labels.append(f"{original}{pos + 1}{substitute}({cdr_name})")

        if not mutation_labels:
            continue

        new_seq = "".join(seq_list)
        if new_seq in seen_sequences:
            continue

        seen_sequences.add(new_seq)
        mutants.append(
            MutantSequence(
                name=f"5U64_mut{len(mutants) + 1:03d}",
                sequence=new_seq,
                mutations=mutation_labels,
            )
        )

    return mutants


def to_fasta(
    mutants: Sequence[MutantSequence],
    include_reference: bool = True,
    line_width: int = 80,
) -> str:
    """Format mutant sequences as a FASTA string.

    Parameters
    ----------
    mutants:
        Iterable of ``MutantSequence`` records.
    include_reference:
        If *True*, the wild-type 5U64 reference is prepended.
    line_width:
        Maximum characters per sequence line.
    """
    lines: list[str] = []

    if include_reference:
        lines.append(">5U64_VHH_reference")
        for i in range(0, len(REFERENCE_SEQUENCE), line_width):
            lines.append(REFERENCE_SEQUENCE[i : i + line_width])

    for m in mutants:
        header = f">{m.name} mutations={';'.join(m.mutations)} parent={m.parent}"
        lines.append(header)
        for i in range(0, len(m.sequence), line_width):
            lines.append(m.sequence[i : i + line_width])

    return "\n".join(lines) + "\n"
