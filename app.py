"""Streamlit application – VHH Structural Paratope Clustering Tool.

Run with:  ``streamlit run app.py``
"""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler

from vhh_clustering.cdr_annotation import annotate_cdrs
from vhh_clustering.clustering import (
    build_result_dataframe,
    cluster,
    gpu_available,
    reduce_dimensions,
)
from vhh_clustering.features import extract_features
from vhh_clustering.parsing import parse_structure_from_bytes
from vhh_clustering.structural_clustering import pairwise_cdr_rmsd, structural_cluster

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

with st.sidebar.expander("Advanced Parameters"):
    dim_method = st.selectbox(
        "Dimensionality reduction", ["umap", "tsne", "pca"], index=0
    )
    min_cluster_size = st.slider("HDBSCAN min cluster size", 2, 20, 3)
    plot_dimensions = st.radio("Plot dimensions", ["2D", "3D"], index=0)

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
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
            with st.expander(f"⚠️ Failed to process **{f.name}**: {exc}"):
                st.code(traceback.format_exc())

    n_processed = len(all_features)
    st.metric("Structures successfully processed", n_processed)

    if n_processed < 2:
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

    st.success("✅ Processing complete")

    # ----- Build feature matrix -----
    names = [f.name for f in all_features]
    matrix = np.vstack([f.vector for f in all_features])
    hotspot_scores = [f.hotspot_score for f in all_features]
    cdr_seqs = [f.cdr_sequences for f in all_features]

    # ----- Standardise once for both dim reduction and clustering -----
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    # ----- Dim reduction (2D or 3D based on user selection) -----
    n_components = 3 if plot_dimensions == "3D" else 2
    embedding = reduce_dimensions(matrix_scaled, method=dim_method, n_components=n_components)

    # ----- Clustering (mode-dependent) -----
    if clustering_mode == "Structural (CDR Cα RMSD)":
        struct_df = structural_cluster(
            all_annotated, names, distance_threshold=rmsd_threshold
        )
        labels = np.array(struct_df["structural_cluster"].tolist())
    else:
        labels = cluster(matrix_scaled, min_cluster_size=min_cluster_size)

    result_df = build_result_dataframe(
        names, embedding, labels, hotspot_scores, cdr_seqs
    )

    # Merge structural cluster info when available
    if clustering_mode == "Structural (CDR Cα RMSD)":
        result_df["representative"] = struct_df["representative"].tolist()

    # ----- Interactive scatter plot (2D or 3D) -----
    st.subheader("Cluster Plot")
    hover_cols = ["structure", "cluster", "hotspot_score", "CDR-H1", "CDR-H2", "CDR-H3"]
    if plot_dimensions == "3D":
        fig = px.scatter_3d(
            result_df,
            x="dim1",
            y="dim2",
            z="dim3",
            color=result_df["cluster"].astype(str),
            hover_data=hover_cols,
            title=f"Paratope Embedding ({dim_method.upper()})",
            labels={
                "dim1": "Dimension 1",
                "dim2": "Dimension 2",
                "dim3": "Dimension 3",
                "color": "Cluster",
            },
            height=700,
        )
        fig.update_layout(scene_camera={"eye": {"x": 1.5, "y": 1.5, "z": 1.0}})
    else:
        fig = px.scatter(
            result_df,
            x="dim1",
            y="dim2",
            color=result_df["cluster"].astype(str),
            hover_data=hover_cols,
            title=f"Paratope Embedding ({dim_method.upper()})",
            labels={
                "dim1": "Dimension 1",
                "dim2": "Dimension 2",
                "color": "Cluster",
            },
            height=700,
        )
    st.plotly_chart(fig, use_container_width=True)

    # ----- RMSD heatmap (structural mode only) -----
    if clustering_mode == "Structural (CDR Cα RMSD)":
        st.subheader("CDR Cα RMSD Heatmap")
        dist_matrix = pairwise_cdr_rmsd(all_annotated)
        # Replace inf with a large display value
        display_matrix = np.where(np.isinf(dist_matrix), np.nan, dist_matrix)
        heatmap_df = pd.DataFrame(display_matrix, index=names, columns=names)
        fig_hm = px.imshow(
            heatmap_df,
            labels={"color": "RMSD (Å)"},
            title="Pairwise CDR Cα RMSD (Å)",
            color_continuous_scale="RdBu_r",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # ----- Cluster summary statistics -----
    st.subheader("Cluster Summary")
    summary_rows = []
    for cluster_id in sorted(result_df["cluster"].unique()):
        cluster_members = result_df[result_df["cluster"] == cluster_id]
        valid_scores = cluster_members["hotspot_score"].dropna()
        if valid_scores.empty:
            representative = cluster_members.iloc[0]["structure"]
        else:
            representative = cluster_members.loc[valid_scores.idxmax(), "structure"]
        summary_rows.append(
            {
                "cluster": cluster_id,
                "members": len(cluster_members),
                "representative": representative,
                "mean_hotspot_score": round(cluster_members["hotspot_score"].mean(), 3),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

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
