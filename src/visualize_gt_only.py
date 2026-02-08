"""
Visualize ground truth labels locally (no model needed).
Useful for verifying visualization code and understanding the data.

Usage:
    python src/visualize_gt_only.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import zoom
import tifffile
import os
import pandas as pd


def render_surface(mask, ax, color, alpha, downsample=0.4):
    vol = zoom(mask.astype(np.float32), downsample, order=1)
    try:
        verts, faces, _, _ = measure.marching_cubes(
            vol, level=0.5, step_size=1, allow_degenerate=False
        )
    except (ValueError, RuntimeError):
        return 0
    verts = verts / downsample
    mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidths=0.1)
    mesh.set_facecolor(color)
    mesh.set_edgecolor((*plt.cm.colors.to_rgb(color), 0.05))
    ax.add_collection3d(mesh)
    return len(faces)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    out_root = os.path.join(os.path.dirname(__file__), "..", "output", "visualizations", "gt_exploration")
    os.makedirs(out_root, exist_ok=True)

    train_img_dir = os.path.join(data_dir, "train_images")
    train_lbl_dir = os.path.join(data_dir, "train_labels")

    # Pick first available sample
    files = sorted(os.listdir(train_lbl_dir))[:3]
    print(f"Visualizing {len(files)} samples: {files}")

    for fname in files:
        sid = fname.replace(".tif", "")
        print(f"\n--- Sample {sid} ---")

        img = tifffile.imread(os.path.join(train_img_dir, fname))
        lbl = tifffile.imread(os.path.join(train_lbl_dir, fname))
        print(f"  Image: {img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
        print(f"  Label: {lbl.shape}, classes={np.unique(lbl)}")
        print(f"  Class 0: {(lbl==0).sum():,} | Class 1: {(lbl==1).sum():,} | Class 2: {(lbl==2).sum():,}")

        D = lbl.shape[0]
        n_slices = 8
        indices = np.linspace(0, D - 1, n_slices, dtype=int)
        cmap3 = ListedColormap(["black", "white", "gray"])

        # --- Slice view: image + label side by side ---
        fig, axes = plt.subplots(2, n_slices, figsize=(3 * n_slices, 6))
        for i, z in enumerate(indices):
            axes[0, i].imshow(img[z], cmap="gray")
            axes[0, i].set_title(f"z={z}", fontsize=9)
            axes[0, i].axis("off")

            axes[1, i].imshow(lbl[z], cmap=cmap3, vmin=0, vmax=2)
            axes[1, i].axis("off")

        axes[0, 0].set_ylabel("CT scan", fontsize=11)
        axes[1, 0].set_ylabel("Label", fontsize=11)
        fig.suptitle(f"Sample {sid} — {img.shape}", fontsize=14)
        plt.tight_layout()
        path = os.path.join(out_root, f"{sid}_slices.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # --- Class distribution per slice ---
        fig, ax = plt.subplots(figsize=(10, 4))
        c1_per_slice = [(lbl[z] == 1).sum() for z in range(D)]
        c2_per_slice = [(lbl[z] == 2).sum() for z in range(D)]
        ax.fill_between(range(D), c1_per_slice, alpha=0.7, label="Class 1 (surface)")
        ax.fill_between(range(D), c2_per_slice, alpha=0.5, label="Class 2 (masked)")
        ax.set_xlabel("Z slice")
        ax.set_ylabel("Voxel count")
        ax.set_title(f"Class distribution per slice — {sid}")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(out_root, f"{sid}_class_dist.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # --- 3D surface of GT label ---
        gt_bin = (lbl == 1).astype(np.uint8)
        if gt_bin.sum() > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            n_tri = render_surface(gt_bin, ax, "forestgreen", 0.5, downsample=0.3)
            shape = gt_bin.shape
            ax.set_xlim(0, shape[0]); ax.set_ylim(0, shape[1]); ax.set_zlim(0, shape[2])
            ax.set_title(f"GT Surface — {sid} ({n_tri:,} triangles)", fontsize=12)
            ax.view_init(elev=25, azim=45)
            plt.tight_layout()
            path = os.path.join(out_root, f"{sid}_3d_gt.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")
        else:
            print(f"  No class-1 voxels, skipping 3D render")

    print(f"\nAll outputs in: {out_root}")


if __name__ == "__main__":
    main()
