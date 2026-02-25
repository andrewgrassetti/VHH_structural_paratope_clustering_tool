"""Structure parsing for PDB and mmCIF files.

Extracts residue-level information from antibody fragment structures,
handling both experimental and predicted structures (AlphaFold, Boltz).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    from Bio.Data.IUPACData import protein_letters_3to1

    def three_to_one(three: str) -> str:
        key = three.strip().capitalize()
        if key in protein_letters_3to1:
            return protein_letters_3to1[key]
        raise KeyError(key)


@dataclass
class Residue:
    """A single residue extracted from a structure."""

    chain_id: str
    res_seq: int
    res_name: str  # three-letter code
    one_letter: str
    ca_coord: Optional[np.ndarray] = None  # Cα xyz
    all_coords: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    b_factor: float = 0.0  # mean B-factor / pLDDT for predicted structures


@dataclass
class ParsedStructure:
    """Container for a parsed VHH structure."""

    name: str
    residues: list[Residue]
    sequence: str = ""

    def __post_init__(self) -> None:
        self.sequence = "".join(r.one_letter for r in self.residues)


def _safe_one_letter(three: str) -> str:
    """Convert three-letter residue name to one-letter, returning 'X' on failure."""
    try:
        return three_to_one(three)
    except KeyError:
        return "X"


def _extract_residues(model) -> list[Residue]:
    """Pull residues from the first model of a BioPython structure."""
    residues: list[Residue] = []
    for chain in model:
        for res in chain.get_residues():
            het_flag = res.get_id()[0]
            if het_flag.strip() and het_flag != " ":
                continue  # skip HETATM / water
            three = res.get_resname().strip()
            one = _safe_one_letter(three)
            coords = np.array([a.get_vector().get_array() for a in res.get_atoms()])
            ca_coord = None
            if "CA" in res:
                ca_coord = np.array(res["CA"].get_vector().get_array())
            b_factors = [a.get_bfactor() for a in res.get_atoms()]
            mean_b = float(np.mean(b_factors)) if b_factors else 0.0
            residues.append(
                Residue(
                    chain_id=chain.id,
                    res_seq=res.get_id()[1],
                    res_name=three,
                    one_letter=one,
                    ca_coord=ca_coord,
                    all_coords=coords,
                    b_factor=mean_b,
                )
            )
    return residues


def parse_structure(filepath: str | Path) -> ParsedStructure:
    """Parse a PDB or mmCIF file and return a ``ParsedStructure``.

    Determines format from file extension (.pdb / .cif / .mmcif).
    """
    filepath = Path(filepath)
    name = filepath.stem
    ext = filepath.suffix.lower()

    if ext in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    elif ext == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    structure = parser.get_structure(name, str(filepath))
    model = structure[0]
    residues = _extract_residues(model)
    return ParsedStructure(name=name, residues=residues)


def parse_structure_from_bytes(data: bytes, filename: str) -> ParsedStructure:
    """Parse structure from in-memory bytes (for Streamlit uploads)."""
    ext = Path(filename).suffix.lower()
    # Write to a temporary path so BioPython can parse it
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        return parse_structure(tmp.name)
