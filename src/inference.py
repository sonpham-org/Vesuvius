"""
Inference pipeline for Vesuvius Challenge Surface Detection.

Implements:
- Sliding window inference
- Test-time augmentation (TTA)
- Topology-aware post-processing
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
import pandas as pd
from tqdm import tqdm
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects


class SlidingWindowInference:
    """
    Sliding window inference for 3D volumes.

    Supports Gaussian weighting for smooth patch boundaries.
    """

    def __init__(
        self,
        model: nn.Module,
        roi_size: Tuple[int, int, int] = (160, 160, 160),
        overlap: float = 0.5,
        mode: str = 'gaussian',
        device: str = 'cuda',
    ):
        self.model = model
        self.roi_size = roi_size
        self.overlap = overlap
        self.mode = mode
        self.device = device

        # Create Gaussian weights for blending
        if mode == 'gaussian':
            self.weights = self._create_gaussian_weights()
        else:
            self.weights = torch.ones(roi_size)

    def _create_gaussian_weights(self) -> torch.Tensor:
        """Create 3D Gaussian weights for patch blending."""
        d, h, w = self.roi_size
        sigma = min(d, h, w) / 4

        # Create 1D Gaussians
        z = torch.arange(d).float() - d / 2
        y = torch.arange(h).float() - h / 2
        x = torch.arange(w).float() - w / 2

        gz = torch.exp(-z**2 / (2 * sigma**2))
        gy = torch.exp(-y**2 / (2 * sigma**2))
        gx = torch.exp(-x**2 / (2 * sigma**2))

        # Create 3D Gaussian
        weights = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]

        return weights

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """
        Run sliding window inference.

        Args:
            volume: Input volume (1, D, H, W, 1) or (D, H, W)

        Returns:
            output: Predicted logits (D, H, W, C)
        """
        # Handle input shape
        if volume.ndim == 5:
            volume = volume[0, ..., 0]  # Remove batch and channel dims
        elif volume.ndim == 4:
            volume = volume[0]

        d, h, w = volume.shape
        pd, ph, pw = self.roi_size

        # Calculate step sizes
        step_d = int(pd * (1 - self.overlap))
        step_h = int(ph * (1 - self.overlap))
        step_w = int(pw * (1 - self.overlap))

        # Initialize output and weight accumulator
        # We'll determine num_classes from first prediction
        output = None
        weight_sum = np.zeros((d, h, w), dtype=np.float32)

        # Generate patch positions
        positions = []
        for z in range(0, max(1, d - pd + 1), step_d):
            for y in range(0, max(1, h - ph + 1), step_h):
                for x in range(0, max(1, w - pw + 1), step_w):
                    positions.append((z, y, x))

        # Add positions near edges
        if d > pd:
            positions.append((d - pd, 0, 0))
        if h > ph:
            positions.append((0, h - ph, 0))
        if w > pw:
            positions.append((0, 0, w - pw))
        if d > pd and h > ph and w > pw:
            positions.append((d - pd, h - ph, w - pw))

        # Remove duplicates
        positions = list(set(positions))

        # Process each patch
        self.model.eval()
        weights_np = self.weights.numpy()

        with torch.no_grad():
            for z, y, x in tqdm(positions, desc="Sliding window", leave=False):
                # Extract patch
                patch = volume[z:z+pd, y:y+ph, x:x+pw]

                # Pad if necessary
                if patch.shape != self.roi_size:
                    pad_d = pd - patch.shape[0]
                    pad_h = ph - patch.shape[1]
                    pad_w = pw - patch.shape[2]
                    patch = np.pad(
                        patch,
                        ((0, pad_d), (0, pad_h), (0, pad_w)),
                        mode='constant'
                    )

                # Prepare for model
                patch_tensor = torch.from_numpy(patch).float()
                patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
                patch_tensor = patch_tensor.to(self.device)

                # Predict
                pred = self.model(patch_tensor)
                pred = pred.cpu().numpy()[0]  # (C, D, H, W)

                # Initialize output array
                if output is None:
                    num_classes = pred.shape[0]
                    output = np.zeros((num_classes, d, h, w), dtype=np.float32)

                # Get valid region sizes
                valid_d = min(pd, d - z)
                valid_h = min(ph, h - y)
                valid_w = min(pw, w - x)

                # Accumulate weighted predictions
                for c in range(num_classes):
                    output[c, z:z+valid_d, y:y+valid_h, x:x+valid_w] += \
                        pred[c, :valid_d, :valid_h, :valid_w] * \
                        weights_np[:valid_d, :valid_h, :valid_w]

                weight_sum[z:z+valid_d, y:y+valid_h, x:x+valid_w] += \
                    weights_np[:valid_d, :valid_h, :valid_w]

        # Normalize by weights
        weight_sum = np.maximum(weight_sum, 1e-8)
        for c in range(output.shape[0]):
            output[c] /= weight_sum

        # Transpose to (D, H, W, C)
        output = np.transpose(output, (1, 2, 3, 0))

        return output


class TestTimeAugmentation:
    """Test-time augmentation with geometric transforms."""

    @staticmethod
    def augment_and_predict(
        volume: np.ndarray,
        inference_fn,
        use_flips: bool = True,
        use_rotations: bool = True,
    ) -> np.ndarray:
        """
        Apply TTA and average predictions.

        Args:
            volume: Input volume (1, D, H, W, 1)
            inference_fn: Function that takes volume and returns logits
            use_flips: Whether to use flip augmentations
            use_rotations: Whether to use rotation augmentations

        Returns:
            Averaged logits (D, H, W, C)
        """
        logits_list = []

        # Original
        logits_list.append(inference_fn(volume))

        if use_flips:
            # Flips along each spatial axis
            for axis in [1, 2, 3]:  # D, H, W in (B, D, H, W, C) format
                vol_flipped = np.flip(volume, axis=axis).copy()
                logits = inference_fn(vol_flipped)
                logits = np.flip(logits, axis=axis - 1).copy()  # Adjust axis for output
                logits_list.append(logits)

        if use_rotations:
            # 90-degree rotations in H-W plane
            for k in [1, 2, 3]:
                vol_rotated = np.rot90(volume, k=k, axes=(2, 3)).copy()
                logits = inference_fn(vol_rotated)
                logits = np.rot90(logits, k=-k, axes=(1, 2)).copy()
                logits_list.append(logits)

        # Average logits
        mean_logits = np.mean(logits_list, axis=0)

        return mean_logits


class TopologyPostProcessor:
    """Topology-aware post-processing pipeline."""

    def __init__(
        self,
        t_low: float = 0.30,
        t_high: float = 0.80,
        z_radius: int = 3,
        xy_radius: int = 2,
        dust_min_size: int = 100,
    ):
        self.t_low = t_low
        self.t_high = t_high
        self.z_radius = z_radius
        self.xy_radius = xy_radius
        self.dust_min_size = dust_min_size

    def _build_anisotropic_struct(self) -> Optional[np.ndarray]:
        """Build anisotropic structuring element."""
        z, r = self.z_radius, self.xy_radius

        if z == 0 and r == 0:
            return None

        if z == 0 and r > 0:
            size = 2 * r + 1
            struct = np.zeros((1, size, size), dtype=bool)
            cy, cx = r, r
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy * dy + dx * dx <= r * r:
                        struct[0, cy + dy, cx + dx] = True
            return struct

        if z > 0 and r == 0:
            struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
            struct[:, 0, 0] = True
            return struct

        depth = 2 * z + 1
        size = 2 * r + 1
        struct = np.zeros((depth, size, size), dtype=bool)
        cz, cy, cx = z, r, r
        for dz in range(-z, z + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy * dy + dx * dx <= r * r:
                        struct[cz + dz, cy + dy, cx + dx] = True
        return struct

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to probability/class map.

        Args:
            probs: Either probability map (D, H, W) or class indices (D, H, W)

        Returns:
            Binary mask (D, H, W) as uint8
        """
        # Step 1: Hysteresis thresholding
        strong = probs >= self.t_high
        weak = probs >= self.t_low

        if not strong.any():
            return np.zeros_like(probs, dtype=np.uint8)

        # 26-connectivity propagation
        struct_hyst = ndi.generate_binary_structure(3, 3)
        mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

        if not mask.any():
            return np.zeros_like(probs, dtype=np.uint8)

        # Step 2: Anisotropic morphological closing
        struct_close = self._build_anisotropic_struct()
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

        # Step 3: Dust removal
        if self.dust_min_size > 0:
            mask = remove_small_objects(mask.astype(bool), min_size=self.dust_min_size)

        return mask.astype(np.uint8)


def run_inference(
    model: nn.Module,
    data_dir: str,
    test_csv: str,
    output_path: str,
    roi_size: Tuple[int, int, int] = (160, 160, 160),
    overlap: float = 0.5,
    use_tta: bool = True,
    post_process: bool = True,
    device: str = 'cuda',
    **pp_kwargs
) -> str:
    """
    Run full inference pipeline and create submission.

    Args:
        model: Trained model
        data_dir: Path to data directory
        test_csv: Path to test.csv
        output_path: Path for output submission.zip
        roi_size: Patch size for sliding window
        overlap: Overlap fraction for sliding window
        use_tta: Whether to use test-time augmentation
        post_process: Whether to apply post-processing
        device: Device to run on
        **pp_kwargs: Post-processing parameters

    Returns:
        Path to submission.zip
    """
    model = model.to(device)
    model.eval()

    # Initialize components
    slider = SlidingWindowInference(
        model=model,
        roi_size=roi_size,
        overlap=overlap,
        device=device,
    )

    if post_process:
        post_processor = TopologyPostProcessor(**pp_kwargs)

    # Load test data info
    test_df = pd.read_csv(test_csv)
    test_dir = Path(data_dir) / 'test_images'

    # Create temporary output directory
    output_dir = Path(output_path).parent / 'submission_masks'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {len(test_df)} test volumes...")

    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            image_id = row['id']
            tif_path = test_dir / f'{image_id}.tif'

            # Load volume
            volume = tifffile.imread(str(tif_path)).astype(np.float32)
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
            volume = volume[None, ..., None]  # (1, D, H, W, 1)

            # Run inference with optional TTA
            if use_tta:
                logits = TestTimeAugmentation.augment_and_predict(
                    volume,
                    slider,
                    use_flips=True,
                    use_rotations=True,
                )
            else:
                logits = slider(volume)

            # Get class predictions
            class_map = logits.argmax(axis=-1).astype(np.uint8)

            # Apply post-processing
            if post_process:
                # For argmax output, we use the class map directly
                # Post-processor treats values >= t_high as foreground
                mask = post_processor(class_map.astype(np.float32))
            else:
                mask = (class_map == 1).astype(np.uint8)

            # Save to zip
            out_path = output_dir / f'{image_id}.tif'
            tifffile.imwrite(str(out_path), mask)
            zf.write(str(out_path), arcname=f'{image_id}.tif')
            out_path.unlink()  # Remove temp file

    # Cleanup
    output_dir.rmdir()

    print(f"Submission saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test post-processor
    pp = TopologyPostProcessor()
    test_probs = np.random.rand(64, 64, 64)
    mask = pp(test_probs)
    print(f"Post-processor test: {test_probs.shape} -> {mask.shape}, unique: {np.unique(mask)}")
