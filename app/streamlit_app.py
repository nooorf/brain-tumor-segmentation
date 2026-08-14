# Brain Tumor Segmentation - Streamlit Web Application
#
# Model: 3D U-Net (MONAI), hosted privately on Hugging Face Hub
# Model weights (best_model.pth) are excluded from git due to file size.

import os
import tempfile

import numpy as np
import streamlit as st
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from skimage import measure
from scipy import ndimage

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import NormalizeIntensity

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

load_dotenv()

HF_REPO_ID = "foroughm423/brain-tumor-segmentation-3d-unet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96)
MODALITIES = ["t1", "t1ce", "t2", "flair"]
CLASS_LABELS = {1: "NCR/NET (necrotic core)", 2: "ED (edema)", 3: "ET (enhancing tumor)"}
CLASS_COLORS = {1: "royalblue", 2: "gold", 3: "crimson"}
FIGURES_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

MODEL_PATH = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="best_model.pth",
    token=os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN"),
)

# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path):
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# ── Helper functions ──────────────────────────────────────────────────────────

def load_nifti_from_upload(uploaded_file):
    """Save an uploaded NIfTI file to a temp path and load it with nibabel."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    volume = nib.load(tmp_path).get_fdata()
    os.unlink(tmp_path)
    return volume


def run_inference(model, volumes_dict):
    """
    volumes_dict: {"t1": array, "t1ce": array, "t2": array, "flair": array}
    Returns: (pred_classes as numpy array, dict of normalized modality arrays)
    """
    stacked = np.stack([volumes_dict[m] for m in MODALITIES], axis=0)  # (4, H, W, D)
    tensor = torch.from_numpy(stacked).float()

    normalize = NormalizeIntensity(nonzero=True, channel_wise=True)
    tensor = normalize(tensor)

    input_tensor = tensor.unsqueeze(0).to(DEVICE)  # (1, 4, H, W, D)

    with torch.no_grad():
        output = sliding_window_inference(input_tensor, PATCH_SIZE, sw_batch_size=1, predictor=model)
        pred_classes = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    normalized_volumes = {mod: tensor[i].numpy() for i, mod in enumerate(MODALITIES)}
    return pred_classes, normalized_volumes


def render_2d_slice(display_volume, pred, slice_idx, modality_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(display_volume[:, :, slice_idx].T, cmap="gray", origin="lower")
    axes[0].set_title(f"{modality_name} — slice {slice_idx}")
    axes[0].axis("off")

    axes[1].imshow(display_volume[:, :, slice_idx].T, cmap="gray", origin="lower")
    seg_slice = pred[:, :, slice_idx].T
    masked = np.ma.masked_where(seg_slice == 0, seg_slice)
    axes[1].imshow(masked, cmap="jet", alpha=0.6, origin="lower", vmin=0, vmax=3)
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")

    plt.tight_layout()
    return fig


def render_3d(pred, flair):
    labeled_array, num_features = ndimage.label(pred > 0)
    if num_features > 0:
        sizes = ndimage.sum(pred > 0, labeled_array, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        cleaned_mask = (labeled_array == largest_label)
    else:
        cleaned_mask = pred > 0

    fig = go.Figure()

    brain_mask = (flair > np.percentile(flair[flair > 0], 40)).astype(float) if (flair > 0).any() else None
    if brain_mask is not None:
        try:
            brain_verts, brain_faces, _, _ = measure.marching_cubes(brain_mask, level=0.5, step_size=2)
            fig.add_trace(go.Mesh3d(
                x=brain_verts[:, 0], y=brain_verts[:, 1], z=brain_verts[:, 2],
                i=brain_faces[:, 0], j=brain_faces[:, 1], k=brain_faces[:, 2],
                color="lightgray", opacity=0.10, name="Brain", showlegend=True,
            ))
        except Exception:
            pass

    for class_id, color in CLASS_COLORS.items():
        class_mask = (pred == class_id) & cleaned_mask
        if class_mask.sum() < 30:
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(class_mask.astype(float), level=0.5)
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, opacity=0.9, name=CLASS_LABELS[class_id], showlegend=True,
            ))
        except Exception:
            pass

    axis_style = dict(backgroundcolor="rgb(15,15,15)", gridcolor="rgb(60,60,60)", showbackground=True, zeroline=False)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (mm)", **axis_style),
            yaxis=dict(title="Y (mm)", **axis_style),
            zaxis=dict(title="Z (mm)", **axis_style),
            bgcolor="rgb(15,15,15)",
        ),
        paper_bgcolor="rgb(15,15,15)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.6)", bordercolor="gray", borderwidth=1),
        height=650,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("About")
    st.markdown("**Model:** 3D U-Net (MONAI)")
    st.markdown("**Dataset:** BraTS2020")

    st.markdown("---")
    st.markdown("**Key Results (test set)**")
    st.metric("WT Dice", "0.864")
    st.metric("TC Dice", "0.750")
    st.metric("ET Dice", "0.695")

    st.markdown("---")
    st.markdown(
        "⚠️ **Medical Disclaimer:** This model is for research and "
        "educational purposes only. It is not a substitute for "
        "professional medical diagnosis."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_main, tab_results = st.tabs(["Segmentation Demo", "Model Results"])

# ── Tab 1: Segmentation Demo ───────────────────────────────────────────────────

with tab_main:
    st.title("🧠 Brain Tumor Segmentation")
    st.markdown(
        "Upload the 4 MRI modalities (T1, T1ce, T2, FLAIR) for a single patient "
        "in NIfTI format (`.nii` or `.nii.gz`) to generate a 3D tumor segmentation."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        t1_file = st.file_uploader("T1", type=["nii", "gz"], key="t1")
    with col2:
        t1ce_file = st.file_uploader("T1ce", type=["nii", "gz"], key="t1ce")
    with col3:
        t2_file = st.file_uploader("T2", type=["nii", "gz"], key="t2")
    with col4:
        flair_file = st.file_uploader("FLAIR", type=["nii", "gz"], key="flair")

    all_uploaded = all([t1_file, t1ce_file, t2_file, flair_file])

    if all_uploaded:
        with st.spinner("Loading model..."):
            model = load_model(MODEL_PATH)

        with st.spinner("Running segmentation... this may take a minute."):
            volumes = {
                "t1": load_nifti_from_upload(t1_file),
                "t1ce": load_nifti_from_upload(t1ce_file),
                "t2": load_nifti_from_upload(t2_file),
                "flair": load_nifti_from_upload(flair_file),
            }
            pred_classes, normalized_volumes = run_inference(model, volumes)

        st.success("Segmentation complete.")

        tumor_per_slice = (pred_classes > 0).sum(axis=(0, 1))
        default_slice = int(np.argmax(tumor_per_slice)) if tumor_per_slice.max() > 0 else pred_classes.shape[2] // 2

        subtab1, subtab2 = st.tabs(["2D Slice View", "3D Rendering"])

        with subtab1:
            st.markdown(
                "The slider below moves through axial slices of the brain, "
                "from bottom to top. It starts at the slice with the most "
                "predicted tumor volume, so the segmentation is visible right away."
            )

            modality_display = st.radio(
                "Background modality",
                options=["FLAIR", "T1", "T1ce", "T2"],
                index=0,
                horizontal=True,
            )
            modality_key = {"FLAIR": "flair", "T1": "t1", "T1ce": "t1ce", "T2": "t2"}[modality_display]

            slice_idx = st.slider("Slice", 0, pred_classes.shape[2] - 1, default_slice)
            fig2d = render_2d_slice(normalized_volumes[modality_key], pred_classes, slice_idx, modality_display)
            st.pyplot(fig2d)

        with subtab2:
            with st.spinner("Building 3D rendering..."):
                fig3d = render_3d(pred_classes, normalized_volumes["flair"])
            st.plotly_chart(fig3d, use_container_width=True)

    else:
        st.info("Upload all 4 MRI modalities to run segmentation.")

# ── Tab 2: Model Results ──────────────────────────────────────────────────────

with tab_results:
    st.header("Model Performance Summary")

    st.markdown(
        "This project trained a 3D U-Net from scratch on the BraTS2020 dataset "
        "(369 patients). Results are reported using the standard BraTS evaluation "
        "regions — Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) — "
        "on a held-out test set never used during training or model selection."
    )

    st.subheader("Standard BraTS Metrics (Test Set, n=56)")
    data = {
        "Region": ["WT (Whole Tumor)", "TC (Tumor Core)", "ET (Enhancing)"],
        "This Model": ["0.864", "0.750", "0.695"],
        "Reference*": ["0.865", "0.779", "0.641"],
    }
    st.table(data)
    st.caption(
        "*Reference: plain 3D U-Net baseline, BraTS2020, 50 epochs, RTX 3090 24GB "
        "(Inc0mple et al.). This model matches WT, exceeds ET, and is close on TC — "
        "using a consumer GPU (RTX 4060, 8.6GB) with roughly 1/3 the VRAM."
    )

    st.markdown("---")
    st.subheader("Visual Comparison")

    fig_comparison = os.path.join(FIGURES_PATH, "baseline_vs_improved_comparison.png")
    fig_distribution = os.path.join(FIGURES_PATH, "test_set_dice_distribution.png")

    if os.path.exists(fig_comparison) and os.path.exists(fig_distribution):
        col1, col2 = st.columns(2)
        col1.image(fig_comparison, caption="Baseline vs Improved Training", use_container_width=True)
        col2.image(fig_distribution, caption="Test Set Dice Distribution", use_container_width=True)

    st.markdown("---")
    st.markdown("### Engineering Journey")
    st.markdown(
        "1. **Baseline (15 epochs):** Dice 0.657, still rising at the final epoch.\n"
        "2. **Error analysis:** NCR/NET identified as the weakest class; tumor size "
        "not correlated with performance (r=0.177) — ruling out a simple explanation.\n"
        "3. **Extended training (45 epochs)** with a checkpoint/resume system "
        "(survived a real power/network outage): Dice improved to 0.686, "
        "plateauing around epoch 35-45.\n\n"
        "Final test set Dice: 0.672 (raw per-class average), with only a 1.4 "
        "point gap from validation — indicating good generalization."
    )

    st.markdown("---")
    st.markdown(
        "**Dataset:** BraTS2020 (369 patients)  \n"
        "**Model:** 3D U-Net (MONAI, 4.8M parameters)  \n"
        "**Loss:** Dice + Focal  \n"
        "**Framework:** PyTorch  "
    )