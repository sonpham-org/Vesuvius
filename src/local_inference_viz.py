"""
Vesuvius: Local GPU Inference + Full Visualization Pipeline

Designed for NVIDIA RTX 4090 (or any CUDA GPU).
Runs inference on a training sample and generates all 4 visualization types.

Setup (on your 4090 machine):
    pip install jax[cuda12] keras-nightly==3.12.0.dev2025100703 tifffile \
                scipy scikit-image matplotlib pandas tensorflow-cpu

    # Get the exact medicai wheel from Kaggle:
    pip install /path/to/medicai-0.0.3-py3-none-any.whl

Usage:
    KERAS_BACKEND=jax python src/local_inference_viz.py [--sample-idx N]

The script:
1. Loads TransUNet+SEResNeXt50 model
2. Runs 8-fold TTA inference on a training volume
3. Applies post-processing (0.549 recipe)
4. Generates 4 visualization types comparing to ground truth
5. Saves all outputs to output/visualizations/local_inference/
"""

import os
import argparse
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import zoom
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects
import tifffile
import time

import keras
from keras import ops
from medicai.transforms import Compose, NormalizeIntensity
from medicai.models import TransUNet
from medicai.utils.inference import SlidingWindowInference


# === Post-Processing ===

def build_anisotropic_struct(z_radius, xy_radius):
    z, r = z_radius, xy_radius
    if z == 0 and r == 0: return None
    if z == 0 and r > 0:
        size = 2*r+1; struct = np.zeros((1,size,size), dtype=bool); cy=cx=r
        for dy in range(-r,r+1):
            for dx in range(-r,r+1):
                if dy*dy+dx*dx<=r*r: struct[0,cy+dy,cx+dx]=True
        return struct
    if z > 0 and r == 0:
        struct = np.zeros((2*z+1,1,1), dtype=bool); struct[:,0,0]=True; return struct
    depth=2*z+1; size=2*r+1; struct=np.zeros((depth,size,size), dtype=bool); cz=z; cy=cx=r
    for dz in range(-z,z+1):
        for dy in range(-r,r+1):
            for dx in range(-r,r+1):
                if dy*dy+dx*dx<=r*r: struct[cz+dz,cy+dy,cx+dx]=True
    return struct


def topo_postprocess(fg_prob, T_low=0.15, T_high=0.50, z_radius=3, xy_radius=1, dust_min_size=150):
    strong = fg_prob >= T_high
    weak = fg_prob >= T_low
    if not strong.any(): return np.zeros_like(fg_prob, dtype=np.uint8)
    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)
    if not mask.any(): return np.zeros_like(fg_prob, dtype=np.uint8)
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)
    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)
    return mask.astype(np.uint8)


# === 3D Rendering ===

def render_surface(mask, ax, color, alpha, downsample=0.3):
    vol = zoom(mask.astype(np.float32), downsample, order=1)
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, step_size=1, allow_degenerate=False)
    except (ValueError, RuntimeError):
        return 0
    verts = verts / downsample
    mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidths=0.1)
    mesh.set_facecolor(color)
    mesh.set_edgecolor((*plt.cm.colors.to_rgb(color), 0.05))
    ax.add_collection3d(mesh)
    return len(faces)


# === Visualizations ===

def viz_prob_heatmaps(fg_prob, gt_label, sid, out_root, n_slices=8):
    D = fg_prob.shape[0]
    indices = np.linspace(0, D - 1, n_slices, dtype=int)
    fig, axes = plt.subplots(2, n_slices, figsize=(3*n_slices, 6))
    for i, z in enumerate(indices):
        im = axes[0, i].imshow(fg_prob[z], cmap="inferno", vmin=0, vmax=1)
        axes[0, i].set_title(f"z={z}", fontsize=9); axes[0, i].axis("off")
        axes[1, i].imshow(fg_prob[z], cmap="inferno", vmin=0, vmax=1)
        gt_s = (gt_label[z]==1).astype(float)
        if gt_s.any(): axes[1, i].contour(gt_s, levels=[0.5], colors="lime", linewidths=0.8)
        axes[1, i].set_title(f"z={z} + GT", fontsize=9); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("P(surface)", fontsize=10)
    axes[1, 0].set_ylabel("P + GT contour", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="P(surface)")
    fig.suptitle(f"Probability Heatmaps — {sid}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_root, "prob_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def viz_before_after_pp(raw_pred, pp_pred, gt_label, sid, out_root, n_slices=8):
    D = raw_pred.shape[0]
    indices = np.linspace(0, D - 1, n_slices, dtype=int)
    cmap3 = ListedColormap(["black", "white", "gray"])
    fig, axes = plt.subplots(3, n_slices, figsize=(3*n_slices, 9))
    for i, z in enumerate(indices):
        axes[0, i].imshow(raw_pred[z], cmap=cmap3, vmin=0, vmax=2)
        axes[0, i].set_title(f"z={z}", fontsize=9); axes[0, i].axis("off")
        axes[1, i].imshow(pp_pred[z], cmap="gray", vmin=0, vmax=1); axes[1, i].axis("off")
        axes[2, i].imshow(gt_label[z], cmap=cmap3, vmin=0, vmax=2); axes[2, i].axis("off")
    axes[0, 0].set_ylabel("Raw argmax", fontsize=11)
    axes[1, 0].set_ylabel("Post-processed", fontsize=11)
    axes[2, 0].set_ylabel("Ground truth", fontsize=11)
    fig.suptitle(f"Before/After PP vs GT — {sid}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_root, "pp_compare.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def viz_diff_maps(raw_pred, pp_pred, gt_label, sid, out_root, n_slices=8):
    D = raw_pred.shape[0]
    indices = np.linspace(0, D - 1, n_slices, dtype=int)
    raw_bin = (raw_pred == 1).astype(np.uint8)
    pp_bin = pp_pred.astype(np.uint8)
    gt_bin = (gt_label == 1).astype(np.uint8)
    added = (pp_bin == 1) & (raw_bin == 0)
    removed = (pp_bin == 0) & (raw_bin == 1)

    fig, axes = plt.subplots(2, n_slices, figsize=(3*n_slices, 6))
    for i, z in enumerate(indices):
        rgb = np.zeros((*raw_bin.shape[1:], 3), dtype=np.float32)
        rgb[(pp_bin[z]==1) & (raw_bin[z]==1)] = [1,1,1]
        rgb[added[z] & (gt_bin[z]==1)] = [0,1,0]
        rgb[added[z] & (gt_bin[z]==0)] = [1,0,0]
        rgb[removed[z] & (gt_bin[z]==0)] = [0.3,0.5,1]
        rgb[removed[z] & (gt_bin[z]==1)] = [1,0.6,0]
        axes[0, i].imshow(rgb); axes[0, i].set_title(f"z={z}", fontsize=9); axes[0, i].axis("off")
        axes[1, i].imshow(gt_bin[z], cmap="gray", vmin=0, vmax=1); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("PP diff", fontsize=11)
    axes[1, 0].set_ylabel("GT", fontsize=11)
    legend_elements = [
        Patch(facecolor="white", edgecolor="gray", label="Unchanged FG"),
        Patch(facecolor="green", label="Added (correct)"),
        Patch(facecolor="red", label="Added (wrong)"),
        Patch(facecolor="cornflowerblue", label="Removed (correct)"),
        Patch(facecolor="orange", label="Removed (wrong)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle(f"PP Diff Map — {sid}", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    path = os.path.join(out_root, "diff_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

    n_added = added.sum(); n_removed = removed.sum()
    n_added_ok = (added & (gt_bin==1)).sum(); n_removed_ok = (removed & (gt_bin==0)).sum()
    print(f"  Saved: {path}")
    print(f"  PP added {n_added:,} ({n_added_ok:,} correct, {n_added-n_added_ok:,} wrong)")
    print(f"  PP removed {n_removed:,} ({n_removed_ok:,} correct, {n_removed-n_removed_ok:,} wrong)")


def viz_3d_surfaces(pp_pred, gt_label, sid, out_root):
    pp_bin = pp_pred.astype(np.uint8)
    gt_bin = (gt_label == 1).astype(np.uint8)
    shape = pp_pred.shape

    # Side by side
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    n1 = render_surface(pp_bin, ax1, "royalblue", 0.5)
    ax1.set_xlim(0, shape[0]); ax1.set_ylim(0, shape[1]); ax1.set_zlim(0, shape[2])
    ax1.set_title(f"Prediction ({n1:,} tri)", fontsize=12); ax1.view_init(elev=25, azim=45)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    n2 = render_surface(gt_bin, ax2, "forestgreen", 0.5)
    ax2.set_xlim(0, shape[0]); ax2.set_ylim(0, shape[1]); ax2.set_zlim(0, shape[2])
    ax2.set_title(f"Ground Truth ({n2:,} tri)", fontsize=12); ax2.view_init(elev=25, azim=45)
    fig.suptitle(f"3D Surface — {sid}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_root, "3d_side_by_side.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")

    # Multi-angle overlay
    fig, axes_arr = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={"projection": "3d"})
    for i, az in enumerate([0, 90, 180, 270]):
        ax = axes_arr[i]
        render_surface(gt_bin, ax, "forestgreen", 0.3)
        render_surface(pp_bin, ax, "royalblue", 0.3)
        ax.set_xlim(0, shape[0]); ax.set_ylim(0, shape[1]); ax.set_zlim(0, shape[2])
        ax.view_init(elev=20, azim=az); ax.set_title(f"azim={az}", fontsize=10)
    fig.suptitle(f"Overlay: Pred (blue) vs GT (green) — {sid}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_root, "3d_overlay_angles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# === Main ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=0, help="Training sample index")
    parser.add_argument("--weights", default="weights/transunet.seresnext50.160px.comboloss.weights.h5")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="output/visualizations/local_inference")
    parser.add_argument("--n-slices", type=int, default=8)
    args = parser.parse_args()

    OUT_ROOT = args.out_dir
    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f"Keras {keras.__version__} backend: {keras.config.backend()}")
    try:
        import jax; print(f"JAX {jax.__version__} devices: {jax.devices()}")
    except ImportError:
        pass

    # --- Load model ---
    print("\nLoading model...")
    model = TransUNet(
        input_shape=(160, 160, 160, 1),
        encoder_name='seresnext50',
        classifier_activation=None,
        num_classes=3,
    )
    model.load_weights(args.weights)
    print(f"Model: {model.count_params()/1e6:.1f}M params")

    swi = SlidingWindowInference(
        model, num_classes=3, roi_size=(160,160,160),
        sw_batch_size=1, mode='gaussian', overlap=0.1,
    )

    # --- Load training sample ---
    img_dir = os.path.join(args.data_dir, "train_images")
    lbl_dir = os.path.join(args.data_dir, "train_labels")
    files = sorted(os.listdir(img_dir))
    fname = files[args.sample_idx]
    sid = fname.replace(".tif", "")
    print(f"\nSample: {sid} (index {args.sample_idx})")

    volume_raw = tifffile.imread(os.path.join(img_dir, fname))
    gt_label = tifffile.imread(os.path.join(lbl_dir, fname))
    print(f"Volume: {volume_raw.shape} | Label classes: {np.unique(gt_label)}")

    # Preprocess
    volume = volume_raw.astype(np.float32)[None, ..., None]
    data = {"image": volume}
    pipeline = Compose([NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=False)])
    volume = pipeline(data)["image"]

    # --- Inference with TTA ---
    print("\nRunning TTA inference (8 views)...")
    t0 = time.time()
    logits = []
    logits.append(swi(volume))
    for axis in [1, 2, 3]:
        img_f = np.flip(volume, axis=axis)
        p = swi(img_f)
        p = np.flip(p, axis=axis)
        logits.append(p)
    for k in [1, 2, 3]:
        img_r = np.rot90(volume, k=k, axes=(2, 3))
        p = swi(img_r)
        p = np.rot90(p, k=-k, axes=(2, 3))
        logits.append(p)

    mean_logits = np.mean(logits, axis=0)
    mean_prob = ops.softmax(mean_logits, axis=-1)
    raw_pred = np.array(mean_prob).argmax(-1).astype(np.uint8).squeeze()
    fg_prob = np.array(mean_prob[0, ..., 1]).astype(np.float32)
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s")
    print(f"fg_prob range: [{fg_prob.min():.3f}, {fg_prob.max():.3f}]")

    # --- Post-processing ---
    pp_pred = topo_postprocess(fg_prob)
    print(f"PP pred fg voxels: {pp_pred.sum():,} | GT class-1: {(gt_label==1).sum():,}")

    # --- Save .npy intermediates ---
    npy_dir = os.path.join(OUT_ROOT, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    np.save(os.path.join(npy_dir, "fg_prob.npy"), fg_prob)
    np.save(os.path.join(npy_dir, "raw_pred.npy"), raw_pred)
    np.save(os.path.join(npy_dir, "pp_pred.npy"), pp_pred)
    np.save(os.path.join(npy_dir, "gt_label.npy"), gt_label)
    print(f"Saved .npy files to {npy_dir}")

    # --- Visualizations ---
    print("\n=== 1/4 Probability Heatmaps ===")
    viz_prob_heatmaps(fg_prob, gt_label, sid, OUT_ROOT, args.n_slices)

    print("\n=== 2/4 Before/After PP ===")
    viz_before_after_pp(raw_pred, pp_pred, gt_label, sid, OUT_ROOT, args.n_slices)

    print("\n=== 3/4 Diff Maps ===")
    viz_diff_maps(raw_pred, pp_pred, gt_label, sid, OUT_ROOT, args.n_slices)

    print("\n=== 4/4 3D Surfaces ===")
    viz_3d_surfaces(pp_pred, gt_label, sid, OUT_ROOT)

    print(f"\nAll outputs saved to {OUT_ROOT}/")


if __name__ == "__main__":
    main()
