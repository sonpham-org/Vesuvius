"""
Vesuvius Surface Detection — Visualization Suite

Generates 4 visualization types comparing model predictions to ground truth:
1. Probability heatmaps (fg_prob per slice)
2. Before/after post-processing comparison with GT
3. Diff maps showing what PP changed
4. 3D surface renders (prediction vs GT)

Usage on Kaggle:
    Run as a cell in a notebook after inference, or call functions directly.

Usage locally (with pre-saved .npy files):
    python src/visualize.py --pred pred.npy --gt gt.npy --prob prob.npy --out output/visualizations/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import zoom
import scipy.ndimage as ndi
import os


# ---------------------------------------------------------------------------
# 1. Probability Heatmaps
# ---------------------------------------------------------------------------

def plot_prob_heatmaps(
    fg_prob,
    gt_label=None,
    n_slices=8,
    out_dir="output/visualizations/prob_heatmaps",
):
    """
    Plot foreground probability heatmaps at evenly spaced Z-slices.

    Parameters
    ----------
    fg_prob : np.ndarray, shape (D, H, W), float32 in [0, 1]
    gt_label : np.ndarray, shape (D, H, W), optional, int labels {0, 1, 2}
    n_slices : int, number of slices to show
    out_dir : str, output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    D = fg_prob.shape[0]
    indices = np.linspace(0, D - 1, n_slices, dtype=int)

    rows = 2 if gt_label is not None else 1
    fig, axes = plt.subplots(rows, n_slices, figsize=(3 * n_slices, 3 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, z in enumerate(indices):
        im = axes[0, i].imshow(fg_prob[z], cmap="inferno", vmin=0, vmax=1)
        axes[0, i].set_title(f"z={z}", fontsize=9)
        axes[0, i].axis("off")

        if gt_label is not None:
            # Show GT class 1 boundary as green contour over prob
            axes[1, i].imshow(fg_prob[z], cmap="inferno", vmin=0, vmax=1)
            gt_slice = (gt_label[z] == 1).astype(float)
            if gt_slice.any():
                axes[1, i].contour(gt_slice, levels=[0.5], colors="lime", linewidths=0.8)
            axes[1, i].set_title(f"z={z} + GT", fontsize=9)
            axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Prob(class 1)", fontsize=10)
    if gt_label is not None:
        axes[1, 0].set_ylabel("Prob + GT contour", fontsize=10)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="P(surface)")
    fig.suptitle("Foreground Probability Heatmaps", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "prob_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 2. Before / After Post-Processing + GT
# ---------------------------------------------------------------------------

def plot_before_after_pp(
    raw_pred,
    pp_pred,
    gt_label,
    n_slices=8,
    out_dir="output/visualizations/pp_compare",
):
    """
    Side-by-side comparison: raw argmax | post-processed | ground truth.

    Parameters
    ----------
    raw_pred : np.ndarray (D,H,W) — argmax class map (0,1,2) or binary (0,1)
    pp_pred  : np.ndarray (D,H,W) — post-processed binary mask (0,1)
    gt_label : np.ndarray (D,H,W) — ground truth labels {0,1,2}
    """
    os.makedirs(out_dir, exist_ok=True)
    D = raw_pred.shape[0]
    indices = np.linspace(0, D - 1, n_slices, dtype=int)

    fig, axes = plt.subplots(3, n_slices, figsize=(3 * n_slices, 9))

    cmap3 = ListedColormap(["black", "white", "gray"])

    for i, z in enumerate(indices):
        # Raw prediction
        axes[0, i].imshow(raw_pred[z], cmap=cmap3, vmin=0, vmax=2)
        axes[0, i].set_title(f"z={z}", fontsize=9)
        axes[0, i].axis("off")

        # Post-processed
        axes[1, i].imshow(pp_pred[z], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")

        # Ground truth (class 1 = white, class 2 = gray)
        axes[2, i].imshow(gt_label[z], cmap=cmap3, vmin=0, vmax=2)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Raw argmax", fontsize=11)
    axes[1, 0].set_ylabel("Post-processed", fontsize=11)
    axes[2, 0].set_ylabel("Ground truth", fontsize=11)
    fig.suptitle("Before/After Post-Processing vs Ground Truth", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "pp_compare.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 3. Diff Maps — What did post-processing change?
# ---------------------------------------------------------------------------

def plot_diff_maps(
    raw_pred,
    pp_pred,
    gt_label,
    n_slices=8,
    out_dir="output/visualizations/diff_maps",
):
    """
    Show voxels added/removed by post-processing, color-coded:
      green  = correctly added (matches GT)
      red    = incorrectly added (not in GT)
      blue   = correctly removed (was wrong)
      orange = incorrectly removed (was right)
      white  = unchanged foreground

    Parameters
    ----------
    raw_pred : np.ndarray (D,H,W) — binary (class 1) from argmax before PP
    pp_pred  : np.ndarray (D,H,W) — binary mask after PP
    gt_label : np.ndarray (D,H,W) — ground truth (class 1 = surface)
    """
    os.makedirs(out_dir, exist_ok=True)
    D = raw_pred.shape[0]
    indices = np.linspace(0, D - 1, n_slices, dtype=int)

    raw_bin = (raw_pred == 1).astype(np.uint8)
    pp_bin = (pp_pred == 1).astype(np.uint8)
    gt_bin = (gt_label == 1).astype(np.uint8)

    added = (pp_bin == 1) & (raw_bin == 0)
    removed = (pp_bin == 0) & (raw_bin == 1)

    fig, axes = plt.subplots(2, n_slices, figsize=(3 * n_slices, 6))

    for i, z in enumerate(indices):
        # Top row: color-coded diff
        rgb = np.zeros((*raw_bin.shape[1:], 3), dtype=np.float32)

        # Unchanged foreground = white
        unchanged_fg = (pp_bin[z] == 1) & (raw_bin[z] == 1)
        rgb[unchanged_fg] = [1, 1, 1]

        # Added by PP
        added_correct = added[z] & (gt_bin[z] == 1)
        added_wrong = added[z] & (gt_bin[z] == 0)
        rgb[added_correct] = [0, 1, 0]    # green
        rgb[added_wrong] = [1, 0, 0]      # red

        # Removed by PP
        removed_correct = removed[z] & (gt_bin[z] == 0)
        removed_wrong = removed[z] & (gt_bin[z] == 1)
        rgb[removed_correct] = [0.3, 0.5, 1]  # blue
        rgb[removed_wrong] = [1, 0.6, 0]      # orange

        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"z={z}", fontsize=9)
        axes[0, i].axis("off")

        # Bottom row: GT for reference
        axes[1, i].imshow(gt_bin[z], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("PP diff", fontsize=11)
    axes[1, 0].set_ylabel("GT (class 1)", fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="white", edgecolor="gray", label="Unchanged FG"),
        Patch(facecolor="green", label="Added (correct)"),
        Patch(facecolor="red", label="Added (wrong)"),
        Patch(facecolor="cornflowerblue", label="Removed (correct)"),
        Patch(facecolor="orange", label="Removed (wrong)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9)

    fig.suptitle("Post-Processing Diff Map", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    path = os.path.join(out_dir, "diff_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Print stats
    n_added = added.sum()
    n_removed = removed.sum()
    n_added_correct = (added & (gt_bin == 1)).sum()
    n_removed_correct = (removed & (gt_bin == 0)).sum()
    print(f"  PP added {n_added:,} voxels ({n_added_correct:,} correct, {n_added - n_added_correct:,} wrong)")
    print(f"  PP removed {n_removed:,} voxels ({n_removed_correct:,} correct, {n_removed - n_removed_correct:,} wrong)")


# ---------------------------------------------------------------------------
# 4. 3D Surface Renders
# ---------------------------------------------------------------------------

def _render_surface(mask, ax, color, alpha, downsample=0.5):
    """Render a single 3D binary mask as a surface on the given axis."""
    vol = zoom(mask.astype(np.float32), downsample, order=1)
    try:
        verts, faces, _, _ = measure.marching_cubes(
            vol, level=0.5, step_size=1, allow_degenerate=False
        )
    except (ValueError, RuntimeError):
        return 0
    verts = verts / downsample  # scale back to original coords

    mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidths=0.1)
    mesh.set_facecolor(color)
    mesh.set_edgecolor((*plt.cm.colors.to_rgb(color), 0.05))
    ax.add_collection3d(mesh)
    return len(faces)


def plot_3d_surfaces(
    pp_pred,
    gt_label,
    downsample=0.4,
    elev=25,
    azim=45,
    out_dir="output/visualizations/3d_surfaces",
):
    """
    3D surface renders: side-by-side and overlay.

    Parameters
    ----------
    pp_pred  : np.ndarray (D,H,W) binary
    gt_label : np.ndarray (D,H,W) with labels {0,1,2}
    """
    os.makedirs(out_dir, exist_ok=True)
    gt_bin = (gt_label == 1).astype(np.uint8)
    pred_bin = (pp_pred == 1).astype(np.uint8)
    shape = pred_bin.shape

    # --- Side-by-side ---
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    n1 = _render_surface(pred_bin, ax1, "royalblue", 0.5, downsample)
    ax1.set_xlim(0, shape[0]); ax1.set_ylim(0, shape[1]); ax1.set_zlim(0, shape[2])
    ax1.set_title(f"Prediction ({n1:,} tri)", fontsize=12)
    ax1.view_init(elev=elev, azim=azim)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    n2 = _render_surface(gt_bin, ax2, "forestgreen", 0.5, downsample)
    ax2.set_xlim(0, shape[0]); ax2.set_ylim(0, shape[1]); ax2.set_zlim(0, shape[2])
    ax2.set_title(f"Ground Truth ({n2:,} tri)", fontsize=12)
    ax2.view_init(elev=elev, azim=azim)

    fig.suptitle("3D Surface Comparison", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "3d_side_by_side.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Overlay ---
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    _render_surface(gt_bin, ax, "forestgreen", 0.3, downsample)
    _render_surface(pred_bin, ax, "royalblue", 0.3, downsample)
    ax.set_xlim(0, shape[0]); ax.set_ylim(0, shape[1]); ax.set_zlim(0, shape[2])
    ax.set_title("Overlay: Prediction (blue) vs GT (green)", fontsize=12)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    path = os.path.join(out_dir, "3d_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Multiple angles ---
    fig, axes_arr = plt.subplots(1, 4, figsize=(20, 5),
                                  subplot_kw={"projection": "3d"})
    for i, az in enumerate([0, 90, 180, 270]):
        ax = axes_arr[i]
        _render_surface(gt_bin, ax, "forestgreen", 0.3, downsample)
        _render_surface(pred_bin, ax, "royalblue", 0.3, downsample)
        ax.set_xlim(0, shape[0]); ax.set_ylim(0, shape[1]); ax.set_zlim(0, shape[2])
        ax.view_init(elev=20, azim=az)
        ax.set_title(f"azim={az}", fontsize=10)
    fig.suptitle("Overlay from 4 Angles", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "3d_multi_angle.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# All-in-one runner
# ---------------------------------------------------------------------------

def run_all_visualizations(
    fg_prob,
    raw_pred,
    pp_pred,
    gt_label,
    out_root="output/visualizations",
    n_slices=8,
):
    """
    Run all 4 visualization types and save to out_root.

    Parameters
    ----------
    fg_prob  : (D,H,W) float32 — class-1 softmax probability
    raw_pred : (D,H,W) uint8   — argmax before PP (0,1,2)
    pp_pred  : (D,H,W) uint8   — binary mask after PP (0,1)
    gt_label : (D,H,W) uint8   — ground truth labels (0,1,2)
    """
    print(f"Volume shape: {fg_prob.shape}")
    print(f"fg_prob range: [{fg_prob.min():.3f}, {fg_prob.max():.3f}]")
    print(f"raw_pred classes: {np.unique(raw_pred)}")
    print(f"pp_pred classes: {np.unique(pp_pred)}")
    print(f"gt_label classes: {np.unique(gt_label)}")
    print()

    print("=== 1/4 Probability Heatmaps ===")
    plot_prob_heatmaps(fg_prob, gt_label, n_slices,
                       os.path.join(out_root, "prob_heatmaps"))

    print("\n=== 2/4 Before/After PP ===")
    plot_before_after_pp(raw_pred, pp_pred, gt_label, n_slices,
                         os.path.join(out_root, "pp_compare"))

    print("\n=== 3/4 Diff Maps ===")
    plot_diff_maps(raw_pred, pp_pred, gt_label, n_slices,
                   os.path.join(out_root, "diff_maps"))

    print("\n=== 4/4 3D Surfaces ===")
    plot_3d_surfaces(pp_pred, gt_label, out_dir=os.path.join(out_root, "3d_surfaces"))

    print(f"\nAll visualizations saved to {out_root}/")


# ---------------------------------------------------------------------------
# CLI: load from .npy files
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vesuvius visualization suite")
    parser.add_argument("--prob", required=True, help="Path to fg_prob .npy file")
    parser.add_argument("--raw", required=True, help="Path to raw_pred .npy file")
    parser.add_argument("--pred", required=True, help="Path to pp_pred .npy file")
    parser.add_argument("--gt", required=True, help="Path to gt_label .npy file")
    parser.add_argument("--out", default="output/visualizations", help="Output directory")
    parser.add_argument("--slices", type=int, default=8, help="Number of slices to show")
    args = parser.parse_args()

    fg_prob = np.load(args.prob)
    raw_pred = np.load(args.raw)
    pp_pred = np.load(args.pred)
    gt_label = np.load(args.gt)

    run_all_visualizations(fg_prob, raw_pred, pp_pred, gt_label, args.out, args.slices)
