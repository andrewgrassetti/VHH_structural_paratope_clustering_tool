# VHH Structural Paratope Clustering Tool

A tool that accepts VHH (nanobody) antibody fragment structures, identifies paratope residues, computes feature vectors, and clusters them using dimensionality reduction.

## OPIG Tool Integration

This project leverages tools from the [Oxford Protein Informatics Group (OPIG)](http://opig.stats.ox.ac.uk/):

### [ANARCI](https://github.com/oxpig/ANARCI) — Antibody Numbering

ANARCI (Antibody Numbering and Antigen Receptor ClassIfication) is used as the
**primary CDR annotation backend** for IMGT numbering of VHH sequences.  It
supports IMGT, Chothia, Kabat, Martin, and Aho numbering schemes.

> Dunbar, J. and Deane, C.M., 2016. ANARCI: antigen receptor numbering and
> receptor classification. *Bioinformatics*, 32(2), pp.298-300.

### [ImmuneBuilder / NanoBodyBuilder2](https://github.com/oxpig/ImmuneBuilder) — Structure Prediction

NanoBodyBuilder2 enables a **sequence-first workflow**: users can provide VHH
amino-acid sequences and the tool will predict 3D structures on the fly.
NanoBodyBuilder2 achieves state-of-the-art accuracy for nanobody CDR loop
prediction.

> Abanades, B., Wong, W.K., Boyles, F., Georges, G., Bujotzek, A. and Deane,
> C.M., 2023. ImmuneBuilder: Deep-Learning models for predicting the structures
> of immune proteins. *Communications Biology*, 6(1), p.575.

## Architecture

```
┌──────────────────┐
│  Sequence Input   │──── NanoBodyBuilder2 (OPIG) ────┐
│  (optional)       │                                  │
└──────────────────┘                                   ▼
┌──────────────┐     ┌────────────────┐     ┌───────────────────┐
│  Structure   │────▶│  CDR Annotation│────▶│ Feature Extraction│
│  Parsing     │     │  ANARCI (OPIG) │     │ (composition,     │
│  (PDB/mmCIF) │     │  / abnumber    │     │  geometry, SASA,  │
│              │     │                │     │  electrostatics)  │
└──────────────┘     └────────────────┘     └────────┬──────────┘
                                                     │
                                                     ▼
                                            ┌────────────────────┐
                                            │ Dim. Reduction     │
                                            │ (UMAP/t-SNE/PCA)  │
                                            │ + HDBSCAN Cluster  │
                                            └────────┬───────────┘
                                                     │
                                                     ▼
                                            ┌────────────────────┐
                                            │ Streamlit UI       │
                                            │ (interactive plot, │
                                            │  tables, download) │
                                            └────────────────────┘
```

### Modules

| Module | File | Purpose |
|--------|------|---------|
| Parsing | `vhh_clustering/parsing.py` | Read PDB/mmCIF → `ParsedStructure` (residues, coords, B-factors) |
| CDR Annotation | `vhh_clustering/cdr_annotation.py` | IMGT numbering via **ANARCI** (OPIG); fallback to `abnumber` |
| Structure Prediction | `vhh_clustering/structure_prediction.py` | Sequence → structure via **NanoBodyBuilder2** (OPIG ImmuneBuilder) |
| Feature Extraction | `vhh_clustering/features.py` | Fixed-length vector: CDR composition, SASA proxy, geometry, charge, hotspot score |
| Clustering | `vhh_clustering/clustering.py` | UMAP/t-SNE/PCA projection + HDBSCAN clustering; GPU fallback |
| Streamlit UI | `app.py` | Upload structures or paste sequences, process, visualise, and export results |

### Feature vector components

- **CDR composition & length** – per-CDR residue counts and 20-dimensional amino acid frequency
- **Electrostatics proxy** – net charge, positive/negative fraction per CDR
- **Hydrophobicity / polarity / aromaticity** – fractional content per CDR
- **Local geometry** – Cα pairwise distance statistics across CDR residues (mean, std, centroid distance, radius-of-gyration proxy)
- **SASA proxy** – Cα neighbour-count exposure estimate for CDR residues
- **Hotspot score** – weighted sum of CDR lengths (CDR-H3 weighted highest, reflecting its dominant role in antigen binding)

### Clustering rationale

HDBSCAN is used because:
- The number of paratope bins is unknown a priori
- Noise handling is important for diverse predicted structures
- It does not require specifying *k* like KMeans

## How to Run Locally

### Prerequisites

- Python ≥ 3.10
- pip

### Setup

```bash
# Clone
git clone https://github.com/andrewgrassetti/VHH_structural_paratope_clustering_tool.git
cd VHH_structural_paratope_clustering_tool

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install core dependencies
pip install -r requirements.txt
```

### Install OPIG tools (recommended)

```bash
# ANARCI – antibody numbering (requires HMMER)
conda install -c bioconda hmmer=3.3.2 -y  # or install HMMER manually
pip install ANARCI

# ImmuneBuilder – NanoBodyBuilder2 for structure prediction (requires PyTorch)
pip install ImmuneBuilder
```

Both OPIG tools are optional: the application gracefully falls back to
`abnumber` for numbering and disables the sequence input tab when
ImmuneBuilder is not available.

### Run the Streamlit app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`),
upload VHH structure files (`.pdb` / `.cif`) and/or paste VHH sequences,
and explore the results.

Sample structures are provided in `sample_data/` for quick testing.

### Run tests

```bash
python -m pytest tests/ -v
```

## GPU Acceleration

The tool automatically detects RAPIDS cuML. If installed, UMAP, t-SNE, and HDBSCAN will use GPU acceleration:

```bash
# Install cuML (requires NVIDIA GPU + CUDA)
pip install cuml-cu12  # adjust for your CUDA version
```

CPU fallback is always available—no GPU is required.

## Numbering Scheme

The tool uses **IMGT numbering** for CDR boundary detection:
- CDR-H1: positions 27–38
- CDR-H2: positions 56–65
- CDR-H3: positions 105–117

The numbering backend is selected automatically:

1. **ANARCI** (preferred) – OPIG's canonical antibody numbering tool
2. **abnumber** – lightweight fallback
3. **Positional** – uses raw residue sequence numbers when neither library is available

## Assumptions & Extension Points

### Assumptions (MVP)
- Input structures are single-chain VHH (heavy chain only)
- First model in each file is used
- Predicted structures (AlphaFold, Boltz, NanoBodyBuilder2) are provided as PDB/mmCIF files or generated from sequence
- B-factor column stores pLDDT for predicted structures

### Future extensions
- **Better paratope prediction**: integrate ML-based methods (e.g., Parapred, ScanNet)
- **Full electrostatics**: APBS or PDB2PQR-based Poisson-Boltzmann calculation
- **Hotspot scoring**: leverage known antibody–antigen complex databases for position-specific binding frequency
- **Structure alignment**: TM-align or US-align for structural motif similarity
- **Ensemble methods**: combine multiple feature representations
- **GPU embedding**: PyTorch geometric or ESM-based residue embeddings
- **Multi-chain support**: handle Fab and full IgG structures
- **Batch API**: REST endpoint for programmatic access