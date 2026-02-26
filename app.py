"""Streamlit application – VHH Structural Paratope Clustering Tool.

Run with:  ``streamlit run app.py``
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from vhh_clustering.cdr_annotation import annotate_cdrs
from vhh_clustering.clustering import (
    build_result_dataframe,
    cluster,
    gpu_available,
    reduce_dimensions,
)
from vhh_clustering.features import extract_features
from vhh_clustering.parsing import parse_structure_from_bytes
from vhh_clustering.sequence_generator import (
    REFERENCE_SEQUENCE,
    generate_mutants,
    to_fasta,
)
from vhh_clustering.structural_clustering import structural_cluster

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VHH Paratope Clustering",
    layout="wide",
)

st.title("🧬 VHH Structural Paratope Clustering Tool")
st.markdown(
    "Upload VHH antibody fragment structures (PDB / mmCIF) to identify "
    "paratope residues, compute feature vectors, and cluster them."
)

if gpu_available():
    st.info("🚀 GPU acceleration via RAPIDS cuML is available.")
else:
    st.info("Running on **CPU**. Install RAPIDS cuML for GPU acceleration.")

# ---------------------------------------------------------------------------
# Sidebar – parameters
# ---------------------------------------------------------------------------
st.sidebar.header("Parameters")
dim_method = st.sidebar.selectbox(
    "Dimensionality reduction", ["umap", "tsne", "pca"], index=0
)
min_cluster_size = st.sidebar.slider("HDBSCAN min cluster size", 2, 20, 3)

clustering_mode = st.sidebar.selectbox(
    "Clustering method",
    ["Feature-based (HDBSCAN)", "Structural (CDR Cα RMSD)"],
    index=0,
    help=(
        "**Feature-based**: clusters on a multi-dimensional paratope "
        "feature vector using HDBSCAN.  "
        "**Structural**: SPACE2-inspired approach – superimposes "
        "structures on framework Cα atoms and clusters by pairwise "
        "CDR Cα RMSD using agglomerative clustering."
    ),
)

if clustering_mode == "Structural (CDR Cα RMSD)":
    rmsd_threshold = st.sidebar.slider(
        "RMSD distance threshold (Å)", 0.5, 5.0, 1.25, 0.25,
        help="Agglomerative clustering distance cutoff in Å (SPACE2 default: 1.25).",
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_cluster, tab_seqgen = st.tabs(
    ["📊 Structure Clustering", "🧪 Sequence Generator"]
)

# ===========================================================================
# TAB 1 – Structure Clustering
# ===========================================================================
with tab_cluster:
    uploaded_files = st.file_uploader(
        "Upload VHH structure files",
        type=["pdb", "cif", "mmcif"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        all_features = []
        all_annotated: list[list] = []
        all_details: dict[str, list[dict]] = {}

        progress = st.progress(0, text="Processing structures…")
        for idx, f in enumerate(uploaded_files):
            progress.progress(
                (idx + 1) / len(uploaded_files),
                text=f"Processing {f.name}…",
            )
            try:
                structure = parse_structure_from_bytes(f.read(), f.name)
                annotated = annotate_cdrs(structure)
                feats = extract_features(structure.name, annotated)
                all_features.append(feats)
                all_annotated.append(annotated)
                all_details[structure.name] = feats.residue_details
            except Exception as exc:
                st.error(f"Failed to process **{f.name}**: {exc}")

        if len(all_features) < 2:
            st.warning(
                "Upload at least **two** structures to enable clustering and "
                "dimensionality reduction."
            )
            if all_features:
                feats = all_features[0]
                st.subheader(f"Residue details – {feats.name}")
                st.dataframe(
                    pd.DataFrame(feats.residue_details), use_container_width=True
                )
            st.stop()

        # ----- Build feature matrix -----
        names = [f.name for f in all_features]
        matrix = np.vstack([f.vector for f in all_features])
        hotspot_scores = [f.hotspot_score for f in all_features]
        cdr_seqs = [f.cdr_sequences for f in all_features]

        # ----- Dim reduction -----
        embedding = reduce_dimensions(matrix, method=dim_method)

        # ----- Clustering (mode-dependent) -----
        if clustering_mode == "Structural (CDR Cα RMSD)":
            struct_df = structural_cluster(
                all_annotated, names, distance_threshold=rmsd_threshold
            )
            labels = np.array(struct_df["structural_cluster"].tolist())
        else:
            labels = cluster(matrix, min_cluster_size=min_cluster_size)

        result_df = build_result_dataframe(
            names, embedding, labels, hotspot_scores, cdr_seqs
        )

        # Merge structural cluster info when available
        if clustering_mode == "Structural (CDR Cα RMSD)":
            result_df["representative"] = struct_df["representative"].tolist()

        # ----- Interactive scatter plot -----
        st.subheader("Cluster Plot")
        fig = px.scatter(
            result_df,
            x="dim1",
            y="dim2",
            color=result_df["cluster"].astype(str),
            hover_data=[
                "structure", "hotspot_score", "CDR-H1", "CDR-H2", "CDR-H3",
            ],
            title=f"Paratope Embedding ({dim_method.upper()})",
            labels={
                "dim1": "Dimension 1",
                "dim2": "Dimension 2",
                "color": "Cluster",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        # ----- Results table -----
        st.subheader("Summary Table")
        st.dataframe(result_df, use_container_width=True)

        # ----- Per-structure residue details -----
        st.subheader("Per-Structure Residue Details")
        selected = st.selectbox("Select structure", names)
        if selected and selected in all_details:
            st.dataframe(
                pd.DataFrame(all_details[selected]),
                use_container_width=True,
            )

        # ----- Downloads -----
        st.subheader("Download Results")
        col1, col2 = st.columns(2)
        with col1:
            csv_buf = result_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_buf,
                "paratope_clusters.csv",
                "text/csv",
            )
        with col2:
            json_buf = result_df.to_json(orient="records", indent=2)
            st.download_button(
                "📥 Download JSON",
                json_buf,
                "paratope_clusters.json",
                "application/json",
            )
    else:
        st.info("👈 Upload VHH structure files to get started.")

# ===========================================================================
# TAB 2 – Sequence Generator
# ===========================================================================
with tab_seqgen:
    st.subheader("5U64 VHH CDR Mutant Library Generator")
    st.markdown(
        "Generate a library of VHH mutant sequences based on the **PDB 5U64** "
        "reference VHH.  Mutations are **conservative** (biochemically similar "
        "substitutions) and restricted to **CDR-H1, CDR-H2, and CDR-H3** loops.  "
        "Download the resulting FASTA file for external structure prediction "
        "(e.g. AlphaFold, ImmuneBuilder, Boltz)."
    )

    st.code(REFERENCE_SEQUENCE, language=None)
    st.caption(
        "Reference: 5U64 VHH (115 aa) — "
        "CDR-H1: GFPVNRYS · CDR-H2: MSSAGDRS · CDR-H3: NVNVGFEY"
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        n_variants = st.number_input(
            "Number of variants", min_value=1, max_value=1000, value=200
        )
    with col_b:
        min_muts = st.number_input(
            "Min mutations per variant", min_value=1, max_value=8, value=1
        )
    with col_c:
        max_muts = st.number_input(
            "Max mutations per variant", min_value=1, max_value=8, value=4
        )

    seed = st.number_input("Random seed", min_value=0, value=42)

    if st.button("🧬 Generate mutant library"):
        with st.spinner("Generating mutants…"):
            mutants = generate_mutants(
                n_mutants=int(n_variants),
                min_mutations=int(min_muts),
                max_mutations=int(max_muts),
                seed=int(seed),
            )

        st.success(f"Generated **{len(mutants)}** unique mutant sequences.")

        # Summary table
        summary_data = []
        for m in mutants:
            summary_data.append(
                {
                    "Name": m.name,
                    "# Mutations": len(m.mutations),
                    "Mutations": "; ".join(m.mutations),
                    "Sequence": m.sequence,
                }
            )
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # Mutation count distribution chart
        from collections import Counter

        mut_counts = Counter(len(m.mutations) for m in mutants)
        dist_df = pd.DataFrame(
            sorted(mut_counts.items()), columns=["Mutations", "Count"]
        )
        fig = px.bar(
            dist_df,
            x="Mutations",
            y="Count",
            title="Mutation count distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        # FASTA download
        fasta_str = to_fasta(mutants, include_reference=True)
        st.download_button(
            "📥 Download FASTA",
            fasta_str,
            "5U64_VHH_mutant_library.fasta",
            "text/plain",
        )
