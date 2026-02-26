# VHH Structural Paratope Clustering Tool

A Python tool for clustering VHH (nanobody) antibodies based on the structural
similarity of their paratopes (antigen-binding regions). This project leverages
tools from the [Oxford Protein Informatics Group (OPIG)](http://opig.stats.ox.ac.uk/)
for antibody numbering and structure prediction.

## OPIG Tool Integration

This project builds on two key OPIG software packages:

- **[ANARCI](https://github.com/oxpig/ANARCI)** — Antibody Numbering and Antigen
  Receptor ClassIfication. Used to number VHH sequences (IMGT, Chothia, Kabat, etc.)
  and identify CDR (complementarity-determining region) boundaries that define the
  paratope.

  > Dunbar, J. and Deane, C.M., 2016. ANARCI: antigen receptor numbering and
  > receptor classification. *Bioinformatics*, 32(2), pp.298-300.

- **[ImmuneBuilder / NanoBodyBuilder2](https://github.com/oxpig/ImmuneBuilder)** —
  Deep-learning models for predicting the 3D structures of immune proteins. The
  `NanoBodyBuilder2` model predicts nanobody structures with state-of-the-art
  accuracy for CDR loop conformations.

  > Abanades, B., Wong, W.K., Boyles, F., Georges, G., Bujotzek, A. and Deane,
  > C.M., 2023. ImmuneBuilder: Deep-Learning models for predicting the structures
  > of immune proteins. *Communications Biology*, 6(1), p.575.

## Installation

### Requirements

- Python ≥ 3.9
- [ANARCI](https://github.com/oxpig/ANARCI) (requires HMMER)
- [ImmuneBuilder](https://github.com/oxpig/ImmuneBuilder) (requires PyTorch and OpenMM)

### Install

```bash
# Install ANARCI dependencies (via conda)
conda install -c bioconda hmmer=3.3.2 -y

# Install the package and its dependencies
pip install -r requirements.txt
```

## Usage

### 1. Number a VHH sequence (ANARCI)

```python
from vhh_paratope_clustering import number_vhh_sequence, get_cdr_residues

sequence = "QVQLVESGGGLVQPGESLRLSCAASGSIFGIYAVHWFRMAPGKEREFTAGFGSHGSTN..."
numbered = number_vhh_sequence(sequence, scheme="imgt")
cdrs = get_cdr_residues(numbered)

for cdr_name, residues in cdrs.items():
    print(f"{cdr_name}: {''.join(aa for _, aa in residues)}")
```

### 2. Predict a VHH structure (NanoBodyBuilder2)

```python
from vhh_paratope_clustering import predict_vhh_structure

structure = predict_vhh_structure(sequence, output_path="nanobody.pdb")
```

### 3. Identify paratope residues and extract coordinates

```python
from vhh_paratope_clustering import identify_paratope_residues, extract_paratope_coordinates

paratope = identify_paratope_residues(numbered)
coords = extract_paratope_coordinates(structure, paratope)
```

### 4. Cluster paratopes by structural similarity

```python
from vhh_paratope_clustering import cluster_paratopes
from vhh_paratope_clustering.paratope import compute_paratope_distance_matrix

# coords_list: list of coordinate arrays from multiple nanobodies
distance_matrix = compute_paratope_distance_matrix(coords_list)
labels = cluster_paratopes(distance_matrix, method="hierarchical", threshold=2.0)
```

## Project Structure

```
vhh_paratope_clustering/
├── __init__.py       # Package exports
├── numbering.py      # ANARCI-based VHH sequence numbering and CDR identification
├── structure.py      # NanoBodyBuilder2-based VHH structure prediction
├── paratope.py       # Paratope residue identification and RMSD computation
└── clustering.py     # Hierarchical and agglomerative clustering of paratopes
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```