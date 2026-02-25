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

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload VHH structure files",
    type=["pdb", "cif", "mmcif"],
    accept_multiple_files=True,
)

if uploaded_files:
    all_features = []
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
            st.dataframe(pd.DataFrame(feats.residue_details), use_container_width=True)
        st.stop()

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
else:
    st.info("👈 Upload VHH structure files to get started.")
