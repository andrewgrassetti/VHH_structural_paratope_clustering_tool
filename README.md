# VHH Structural Paratope Clustering Tool

A tool that accepts VHH (nanobody) antibody fragment structures, identifies paratope residues, computes feature vectors, and clusters them using dimensionality reduction.

## Architecture

```
┌──────────────┐     ┌────────────────┐     ┌───────────────────┐
│  Structure   │────▶│  CDR Annotation│────▶│ Feature Extraction│
│  Parsing     │     │  (IMGT)        │     │ (composition,     │
│  (PDB/mmCIF) │     │                │     │  geometry, SASA,  │
│              │     │                │     │  electrostatics)  │
└──────────────┘     └────────────────┘     └────────┬──────────┘
                                                     │
                                      ┌──────────────┴──────────────┐
                                      ▼                             ▼
                             ┌────────────────────┐  ┌──────────────────────┐
                             │ Feature-based      │  │ Structural (SPACE2)  │
                             │ UMAP/t-SNE/PCA     │  │ FW Cα alignment +   │
                             │ + HDBSCAN          │  │ CDR Cα RMSD +       │
                             │                    │  │ Agglomerative       │
                             └────────┬───────────┘  └──────────┬──────────┘
                                      │                         │
                                      └────────────┬────────────┘
                                                   ▼
                                          ┌────────────────────┐
                                          │ Streamlit UI       │
                                          │ (interactive plot, │
                                          │  tables, download) │
                                          └────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  Sequence Generator  →  5U64 VHH CDR mutant library  →  FASTA download   │
└────────────────────────────────────────────────────────────────────────────┘
```

### Modules

| Module | File | Purpose |
|--------|------|---------|
| Parsing | `vhh_clustering/parsing.py` | Read PDB/mmCIF → `ParsedStructure` (residues, coords, B-factors) |
| CDR Annotation | `vhh_clustering/cdr_annotation.py` | IMGT numbering via `abnumber`; CDR-H1/H2/H3 classification |
| Feature Extraction | `vhh_clustering/features.py` | Fixed-length vector: CDR composition, SASA proxy, geometry, charge, hotspot score |
| Clustering | `vhh_clustering/clustering.py` | UMAP/t-SNE/PCA projection + HDBSCAN clustering; GPU fallback |
| Structural Clustering | `vhh_clustering/structural_clustering.py` | SPACE2-inspired CDR Cα RMSD clustering with Kabsch framework alignment |
| Sequence Generator | `vhh_clustering/sequence_generator.py` | Generate conservative CDR mutant libraries from the 5U64 VHH reference |
| Streamlit UI | `app.py` | Upload, process, visualise, generate sequences, and export results |

### Feature vector components

- **CDR composition & length** – per-CDR residue counts and 20-dimensional amino acid frequency
- **Electrostatics proxy** – net charge, positive/negative fraction per CDR
- **Hydrophobicity / polarity / aromaticity** – fractional content per CDR
- **Local geometry** – Cα pairwise distance statistics across CDR residues (mean, std, centroid distance, radius-of-gyration proxy)
- **SASA proxy** – Cα neighbour-count exposure estimate for CDR residues
- **Hotspot score** – weighted sum of CDR lengths (CDR-H3 weighted highest, reflecting its dominant role in antigen binding)

### Clustering methods

#### Feature-based (HDBSCAN)

HDBSCAN is used because:
- The number of paratope bins is unknown a priori
- Noise handling is important for diverse predicted structures
- It does not require specifying *k* like KMeans

#### Structural (CDR Cα RMSD) – SPACE2-inspired

Inspired by the [SPACE2 algorithm](https://github.com/oxpig/SPACE2) (Spoendlin *et al.*, *Frontiers in Molecular Biosciences*, 2023), adapted for VHH single-domain antibodies:

1. **Framework alignment** – structures are superimposed on framework Cα atoms using the Kabsch algorithm
2. **CDR Cα RMSD** – pairwise root-mean-square deviation of CDR loop Cα atoms after alignment
3. **Agglomerative clustering** – complete-linkage clustering with a configurable distance threshold (default 1.25 Å)

This approach directly compares 3D CDR loop geometry, complementing the feature-based method which captures sequence composition and statistical geometry descriptors.

### Sequence Generator

The **🧪 Sequence Generator** tab produces a mutant library based on the PDB 5U64 VHH reference sequence (115 aa):

- **Conservative mutations only** – substitutions are biochemically similar (e.g., V→I, D→E, F→Y)
- **CDR-restricted** – mutations occur exclusively in CDR-H1, CDR-H2, and CDR-H3 loops
- **1–4 mutations per variant** (configurable)
- **~200 unique sequences** by default, suitable as test cases for structure prediction tools
- **FASTA export** – download for use with AlphaFold, ImmuneBuilder, Boltz, or other tools

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

# Install dependencies
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`), upload VHH structure files (`.pdb` / `.cif`), and explore the results.

The app has two tabs:
- **📊 Structure Clustering** – upload structures, choose a clustering method, visualise and download results
- **🧪 Sequence Generator** – generate a 5U64 VHH CDR mutant library and download as FASTA

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

The MVP uses **IMGT numbering** via the `abnumber` library for CDR boundary detection:
- CDR-H1: positions 27–38
- CDR-H2: positions 56–65
- CDR-H3: positions 105–117

When `abnumber` renumbering fails (common with some predicted structures), the tool falls back to using residue sequence numbers from the input file.

## Assumptions & Extension Points

### Assumptions (MVP)
- Input structures are single-chain VHH (heavy chain only)
- First model in each file is used
- Predicted structures (AlphaFold, Boltz) are provided as PDB/mmCIF files
- B-factor column stores pLDDT for predicted structures

### Future extensions
- **In-app structure prediction**: integrate ImmuneBuilder or ESMFold for on-the-fly modelling of generated mutants
- **Better paratope prediction**: integrate ML-based methods (e.g., Parapred, ScanNet)
- **Full electrostatics**: APBS or PDB2PQR-based Poisson-Boltzmann calculation
- **Hotspot scoring**: leverage known antibody–antigen complex databases for position-specific binding frequency
- **Structure alignment**: TM-align or US-align for structural motif similarity
- **Ensemble methods**: combine multiple feature representations
- **GPU embedding**: PyTorch geometric or ESM-based residue embeddings
- **Multi-chain support**: handle Fab and full IgG structures
- **Batch API**: REST endpoint for programmatic access