"""
Dataset classes for Vesuvius Challenge Surface Detection.

Handles loading 3D TIFF volumes and their corresponding masks.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


class VesuviusDataset(Dataset):
    """
    Dataset for Vesuvius 3D surface detection.

    Loads 3D TIFF volumes and corresponding label masks.
    Supports random 3D patch extraction for training.
    """

    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        patch_size: Tuple[int, int, int] = (160, 160, 160),
        is_train: bool = True,
        transform: Optional[Callable] = None,
        num_patches_per_volume: int = 4,
        ids: Optional[List[int]] = None,
    ):
        """
        Args:
            data_dir: Path to data directory containing train_images/ and train_labels/
            csv_path: Path to train.csv or test.csv
            patch_size: (D, H, W) size of patches to extract
            is_train: If True, extracts random patches. If False, returns full volume.
            transform: Optional augmentation transform
            num_patches_per_volume: Number of patches to sample per volume during training
            ids: Optional list of specific IDs to include
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.is_train = is_train
        self.transform = transform
        self.num_patches_per_volume = num_patches_per_volume

        # Load CSV
        df = pd.read_csv(csv_path)
        if ids is not None:
            df = df[df['id'].isin(ids)]
        self.ids = df['id'].tolist()

        # Determine paths
        if is_train:
            self.image_dir = self.data_dir / 'train_images'
            self.label_dir = self.data_dir / 'train_labels'
        else:
            self.image_dir = self.data_dir / 'test_images'
            self.label_dir = None

        # Cache loaded volumes for faster training
        self._cache = {}
        self._use_cache = False

    def enable_cache(self):
        """Enable volume caching (uses more RAM but faster)."""
        self._use_cache = True

    def disable_cache(self):
        """Disable volume caching and clear cache."""
        self._use_cache = False
        self._cache = {}

    def __len__(self) -> int:
        if self.is_train:
            return len(self.ids) * self.num_patches_per_volume
        return len(self.ids)

    def _load_volume(self, volume_id: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load image and optionally label volumes."""
        if self._use_cache and volume_id in self._cache:
            return self._cache[volume_id]

        # Load image
        image_path = self.image_dir / f'{volume_id}.tif'
        image = tifffile.imread(str(image_path)).astype(np.float32)

        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Load label if available
        label = None
        if self.label_dir is not None:
            label_path = self.label_dir / f'{volume_id}.tif'
            if label_path.exists():
                label = tifffile.imread(str(label_path)).astype(np.int64)

        result = (image, label)

        if self._use_cache:
            self._cache[volume_id] = result

        return result

    def _extract_random_patch(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract a random patch from the volume."""
        d, h, w = image.shape
        pd, ph, pw = self.patch_size

        # Random start positions
        d_start = np.random.randint(0, max(1, d - pd + 1))
        h_start = np.random.randint(0, max(1, h - ph + 1))
        w_start = np.random.randint(0, max(1, w - pw + 1))

        # Extract patches
        image_patch = image[
            d_start:d_start + pd,
            h_start:h_start + ph,
            w_start:w_start + pw
        ]

        # Pad if necessary
        if image_patch.shape != self.patch_size:
            image_patch = self._pad_volume(image_patch, self.patch_size)

        label_patch = None
        if label is not None:
            label_patch = label[
                d_start:d_start + pd,
                h_start:h_start + ph,
                w_start:w_start + pw
            ]
            if label_patch.shape != self.patch_size:
                label_patch = self._pad_volume(label_patch, self.patch_size, pad_value=2)

        return image_patch, label_patch

    def _pad_volume(
        self,
        volume: np.ndarray,
        target_size: Tuple[int, int, int],
        pad_value: float = 0
    ) -> np.ndarray:
        """Pad volume to target size."""
        d, h, w = volume.shape
        td, th, tw = target_size

        pad_d = max(0, td - d)
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volume = np.pad(
                volume,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=pad_value
            )

        return volume[:td, :th, :tw]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Determine which volume to use
        if self.is_train:
            volume_idx = idx // self.num_patches_per_volume
        else:
            volume_idx = idx

        volume_id = self.ids[volume_idx]

        # Load volume
        image, label = self._load_volume(volume_id)

        # Extract patch if training
        if self.is_train and label is not None:
            image, label = self._extract_random_patch(image, label)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']

        # Convert to tensors
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        image = torch.from_numpy(image).unsqueeze(0).float()

        result = {'image': image, 'id': volume_id}

        if label is not None:
            label = torch.from_numpy(label).long()
            result['label'] = label

        return result


class Augmentation3D:
    """3D augmentation transforms for training."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        intensity_prob: float = 0.3,
    ):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.intensity_prob = intensity_prob

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Apply 3D augmentations."""

        # Random flips along each axis
        for axis in [0, 1, 2]:
            if np.random.random() < self.flip_prob:
                image = np.flip(image, axis=axis).copy()
                if mask is not None:
                    mask = np.flip(mask, axis=axis).copy()

        # Random 90-degree rotations in H-W plane
        if np.random.random() < self.rotate_prob:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(1, 2)).copy()

        # Random intensity augmentations
        if np.random.random() < self.intensity_prob:
            # Random brightness
            image = image + np.random.uniform(-0.1, 0.1)
            image = np.clip(image, 0, 1)

        if np.random.random() < self.intensity_prob:
            # Random contrast
            factor = np.random.uniform(0.8, 1.2)
            mean = image.mean()
            image = (image - mean) * factor + mean
            image = np.clip(image, 0, 1)

        if np.random.random() < self.intensity_prob:
            # Random Gaussian noise
            noise = np.random.normal(0, 0.02, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)

        return {'image': image, 'mask': mask}


def get_dataloaders(
    data_dir: str,
    train_csv: str,
    batch_size: int = 2,
    patch_size: Tuple[int, int, int] = (160, 160, 160),
    num_patches_per_volume: int = 4,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""

    # Read all IDs
    df = pd.read_csv(train_csv)
    all_ids = df['id'].tolist()

    # Split into train/val
    np.random.seed(seed)
    np.random.shuffle(all_ids)
    split_idx = int(len(all_ids) * (1 - val_split))
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]

    print(f"Train samples: {len(train_ids)}, Val samples: {len(val_ids)}")

    # Create datasets
    train_transform = Augmentation3D()

    train_dataset = VesuviusDataset(
        data_dir=data_dir,
        csv_path=train_csv,
        patch_size=patch_size,
        is_train=True,
        transform=train_transform,
        num_patches_per_volume=num_patches_per_volume,
        ids=train_ids,
    )

    val_dataset = VesuviusDataset(
        data_dir=data_dir,
        csv_path=train_csv,
        patch_size=patch_size,
        is_train=True,  # Still extract patches for validation
        transform=None,  # No augmentation for validation
        num_patches_per_volume=2,  # Fewer patches for validation
        ids=val_ids,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
