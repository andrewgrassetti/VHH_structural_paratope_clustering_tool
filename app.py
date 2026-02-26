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

from vhh_clustering.cdr_annotation import annotate_cdrs, numbering_backend
from vhh_clustering.clustering import (
    build_result_dataframe,
    cluster,
    gpu_available,
    reduce_dimensions,
)
from vhh_clustering.features import extract_features
from vhh_clustering.parsing import parse_structure_from_bytes
from vhh_clustering.structure_prediction import immunebuilder_available, predict_structure

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VHH Paratope Clustering",
    layout="wide",
)

st.title("🧬 VHH Structural Paratope Clustering Tool")
st.markdown(
    "Upload VHH antibody fragment structures (PDB / mmCIF) or paste "
    "sequences to predict structures, identify paratope residues, "
    "compute feature vectors, and cluster them."
)

# ---------------------------------------------------------------------------
# Status indicators
# ---------------------------------------------------------------------------
_backend = numbering_backend()
_backend_labels = {"anarci": "ANARCI (OPIG)", "abnumber": "abnumber", "positional": "positional fallback"}
st.info(
    f"🔬 CDR numbering: **{_backend_labels.get(_backend, _backend)}** · "
    f"{'🚀 GPU' if gpu_available() else '💻 CPU'} clustering · "
    f"{'✅' if immunebuilder_available() else '❌'} NanoBodyBuilder2 (OPIG)"
)

# ---------------------------------------------------------------------------
# Sidebar – parameters
# ---------------------------------------------------------------------------
st.sidebar.header("Parameters")
dim_method = st.sidebar.selectbox(
    "Dimensionality reduction", ["umap", "tsne", "pca"], index=0
)
min_cluster_size = st.sidebar.slider("HDBSCAN min cluster size", 2, 20, 3)

# ---------------------------------------------------------------------------
# Sequence input (NanoBodyBuilder2 from OPIG ImmuneBuilder)
# ---------------------------------------------------------------------------
st.sidebar.header("Predict Structures from Sequence")
if immunebuilder_available():
    st.sidebar.markdown(
        "Paste VHH sequences below (one per line, optionally prefixed "
        "with a name: ``name:SEQUENCE``)."
    )
    seq_input = st.sidebar.text_area(
        "VHH sequences",
        height=120,
        placeholder="nb1:EVQLVESGGGLVQ...\nnb2:QVQLQESGGG...",
    )
else:
    seq_input = ""
    st.sidebar.markdown(
        "Install [ImmuneBuilder](https://github.com/oxpig/ImmuneBuilder) "
        "to enable structure prediction from sequences via **NanoBodyBuilder2** (OPIG)."
    )

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload VHH structure files",
    type=["pdb", "cif", "mmcif"],
    accept_multiple_files=True,
)

# ---------------------------------------------------------------------------
# Process inputs
# ---------------------------------------------------------------------------
all_features = []
all_details: dict[str, list[dict]] = {}

# --- Predict structures from sequences (NanoBodyBuilder2) ---
if seq_input and seq_input.strip() and immunebuilder_available():
    seq_lines = [l.strip() for l in seq_input.strip().splitlines() if l.strip()]
    seq_progress = st.progress(0, text="Predicting structures from sequences…")
    for idx, line in enumerate(seq_lines):
        seq_progress.progress(
            (idx + 1) / len(seq_lines),
            text=f"Predicting structure {idx + 1}/{len(seq_lines)}…",
        )
        if ":" in line:
            name, sequence = line.split(":", 1)
            name = name.strip()
            sequence = sequence.strip()
        else:
            name = f"seq_{idx + 1}"
            sequence = line.strip()
        try:
            structure = predict_structure(sequence)
            structure.name = name
            annotated = annotate_cdrs(structure)
            feats = extract_features(name, annotated)
            all_features.append(feats)
            all_details[name] = feats.residue_details
        except Exception as exc:
            st.error(f"Failed to predict **{name}**: {exc}")

# --- Process uploaded PDB/mmCIF files ---
if uploaded_files:
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
            all_details[structure.name] = feats.residue_details
        except Exception as exc:
            st.error(f"Failed to process **{f.name}**: {exc}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if all_features and len(all_features) < 2:
    st.warning(
        "Upload at least **two** structures (or combine sequences and files) "
        "to enable clustering and dimensionality reduction."
    )
    feats = all_features[0]
    st.subheader(f"Residue details – {feats.name}")
    st.dataframe(pd.DataFrame(feats.residue_details), use_container_width=True)
    st.stop()

if len(all_features) >= 2:
    # ----- Build feature matrix -----
    names = [f.name for f in all_features]
    matrix = np.vstack([f.vector for f in all_features])
    hotspot_scores = [f.hotspot_score for f in all_features]
    cdr_seqs = [f.cdr_sequences for f in all_features]

    # ----- Dim reduction + clustering -----
    embedding = reduce_dimensions(matrix, method=dim_method)
    labels = cluster(matrix, min_cluster_size=min_cluster_size)

    result_df = build_result_dataframe(
        names, embedding, labels, hotspot_scores, cdr_seqs
    )

    # ----- Interactive scatter plot -----
    st.subheader("Cluster Plot")
    fig = px.scatter(
        result_df,
        x="dim1",
        y="dim2",
        color=result_df["cluster"].astype(str),
        hover_data=["structure", "hotspot_score", "CDR-H1", "CDR-H2", "CDR-H3"],
        title=f"Paratope Embedding ({dim_method.upper()})",
        labels={"dim1": "Dimension 1", "dim2": "Dimension 2", "color": "Cluster"},
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
            "📥 Download CSV", csv_buf, "paratope_clusters.csv", "text/csv"
        )
    with col2:
        json_buf = result_df.to_json(orient="records", indent=2)
        st.download_button(
            "📥 Download JSON",
            json_buf,
            "paratope_clusters.json",
            "application/json",
        )
elif not uploaded_files and not (seq_input and seq_input.strip()):
    st.info("👈 Upload VHH structure files or paste sequences to get started.")
