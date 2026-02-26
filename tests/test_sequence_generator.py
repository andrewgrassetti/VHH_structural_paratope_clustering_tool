"""Unit tests for VHH CDR mutant sequence generator."""

from __future__ import annotations

import pytest

from vhh_clustering.sequence_generator import (
    CDR_INDICES,
    CONSERVATIVE_SUBSTITUTIONS,
    REFERENCE_SEQUENCE,
    MutantSequence,
    generate_mutants,
    to_fasta,
)


class TestReferenceSequence:
    def test_length(self) -> None:
        assert len(REFERENCE_SEQUENCE) == 115

    def test_cdr_indices_within_bounds(self) -> None:
        for cdr_name, indices in CDR_INDICES.items():
            for idx in indices:
                assert 0 <= idx < len(REFERENCE_SEQUENCE), (
                    f"{cdr_name} index {idx} out of bounds"
                )

    def test_cdr_h1_sequence(self) -> None:
        seq = "".join(REFERENCE_SEQUENCE[i] for i in CDR_INDICES["CDR-H1"])
        assert seq == "GFPVNRYS"

    def test_cdr_h2_sequence(self) -> None:
        seq = "".join(REFERENCE_SEQUENCE[i] for i in CDR_INDICES["CDR-H2"])
        assert seq == "MSSAGDRS"

    def test_cdr_h3_sequence(self) -> None:
        seq = "".join(REFERENCE_SEQUENCE[i] for i in CDR_INDICES["CDR-H3"])
        assert seq == "NVNVGFEY"


class TestConservativeSubstitutions:
    def test_all_standard_amino_acids_covered(self) -> None:
        standard = set("ACDEFGHIKLMNPQRSTVWY")
        assert standard == set(CONSERVATIVE_SUBSTITUTIONS.keys())

    def test_substitutes_differ_from_original(self) -> None:
        for aa, subs in CONSERVATIVE_SUBSTITUTIONS.items():
            for s in subs:
                assert s != aa, f"Substitute {s} equals original {aa}"


class TestGenerateMutants:
    def test_default_produces_200(self) -> None:
        mutants = generate_mutants()
        assert len(mutants) == 200

    def test_all_unique(self) -> None:
        mutants = generate_mutants()
        seqs = [m.sequence for m in mutants]
        assert len(set(seqs)) == len(seqs)

    def test_no_mutant_equals_reference(self) -> None:
        mutants = generate_mutants()
        for m in mutants:
            assert m.sequence != REFERENCE_SEQUENCE

    def test_mutations_only_in_cdrs(self) -> None:
        all_cdr_positions = set()
        for indices in CDR_INDICES.values():
            all_cdr_positions.update(indices)

        mutants = generate_mutants()
        for m in mutants:
            for i, (ref_aa, mut_aa) in enumerate(
                zip(REFERENCE_SEQUENCE, m.sequence)
            ):
                if ref_aa != mut_aa:
                    assert i in all_cdr_positions, (
                        f"Mutation at non-CDR position {i} in {m.name}"
                    )

    def test_sequence_length_preserved(self) -> None:
        mutants = generate_mutants()
        for m in mutants:
            assert len(m.sequence) == len(REFERENCE_SEQUENCE)

    def test_mutation_count_range(self) -> None:
        mutants = generate_mutants(min_mutations=1, max_mutations=4)
        for m in mutants:
            assert 1 <= len(m.mutations) <= 4

    def test_custom_count(self) -> None:
        mutants = generate_mutants(n_mutants=10)
        assert len(mutants) == 10

    def test_reproducible_with_seed(self) -> None:
        m1 = generate_mutants(n_mutants=10, seed=99)
        m2 = generate_mutants(n_mutants=10, seed=99)
        assert [m.sequence for m in m1] == [m.sequence for m in m2]

    def test_different_seeds_differ(self) -> None:
        m1 = generate_mutants(n_mutants=10, seed=1)
        m2 = generate_mutants(n_mutants=10, seed=2)
        seqs1 = {m.sequence for m in m1}
        seqs2 = {m.sequence for m in m2}
        assert seqs1 != seqs2

    def test_mutations_are_conservative(self) -> None:
        mutants = generate_mutants(n_mutants=50)
        all_cdr_positions = set()
        for indices in CDR_INDICES.values():
            all_cdr_positions.update(indices)

        for m in mutants:
            for i in all_cdr_positions:
                ref_aa = REFERENCE_SEQUENCE[i]
                mut_aa = m.sequence[i]
                if ref_aa != mut_aa:
                    allowed = CONSERVATIVE_SUBSTITUTIONS.get(ref_aa, ())
                    assert mut_aa in allowed, (
                        f"{m.name}: {ref_aa}->{mut_aa} at pos {i} "
                        f"not in conservative set {allowed}"
                    )


class TestToFasta:
    def test_includes_reference_by_default(self) -> None:
        mutants = generate_mutants(n_mutants=2)
        fasta = to_fasta(mutants)
        assert ">5U64_VHH_reference" in fasta
        assert REFERENCE_SEQUENCE in fasta.replace("\n", "")

    def test_excludes_reference(self) -> None:
        mutants = generate_mutants(n_mutants=2)
        fasta = to_fasta(mutants, include_reference=False)
        assert ">5U64_VHH_reference" not in fasta

    def test_all_mutants_present(self) -> None:
        mutants = generate_mutants(n_mutants=5)
        fasta = to_fasta(mutants)
        for m in mutants:
            assert f">{m.name}" in fasta

    def test_fasta_ends_with_newline(self) -> None:
        mutants = generate_mutants(n_mutants=2)
        fasta = to_fasta(mutants)
        assert fasta.endswith("\n")

    def test_mutation_annotations_in_header(self) -> None:
        mutants = generate_mutants(n_mutants=3)
        fasta = to_fasta(mutants)
        for m in mutants:
            assert "mutations=" in fasta
            assert "parent=5U64_VHH" in fasta
