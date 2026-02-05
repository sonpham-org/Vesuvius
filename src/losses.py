"""
Topology-preserving loss functions for 3D surface segmentation.

Includes:
- Dice Loss
- Combo Loss (Dice + CE)
- clDice Loss (centerline Dice for tubular structures)
- Boundary Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


class DiceLoss(nn.Module):
    """Standard Dice Loss for segmentation."""

    def __init__(self, smooth: float = 1e-5, ignore_index: int = 2):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, D, H, W) - logits or probabilities
            target: (B, D, H, W) - class indices
        Returns:
            Scalar loss
        """
        # Convert logits to probabilities
        pred = F.softmax(pred, dim=1)

        # Create mask for valid pixels (not ignore_index)
        valid_mask = (target != self.ignore_index).float()

        # One-hot encode target (excluding ignore class)
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(
            target.clamp(0, num_classes - 1),
            num_classes
        ).permute(0, 4, 1, 2, 3).float()

        # Apply valid mask
        pred = pred * valid_mask.unsqueeze(1)
        target_one_hot = target_one_hot * valid_mask.unsqueeze(1)

        # Calculate Dice for each class (skip background)
        dice_scores = []
        for c in range(1, num_classes):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        # Average Dice across foreground classes
        return 1.0 - torch.stack(dice_scores).mean()


class SoftSkeletonize3D(nn.Module):
    """
    Differentiable 3D skeletonization using iterative erosion.

    This approximates the medial axis/skeleton of a 3D volume using
    repeated morphological erosion with a small kernel.
    """

    def __init__(self, num_iterations: int = 10):
        super().__init__()
        self.num_iterations = num_iterations

        # 3D erosion kernel (3x3x3 ball)
        kernel = torch.ones(1, 1, 3, 3, 3)
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute soft skeleton of input volume.

        Args:
            x: (B, 1, D, H, W) binary or soft mask
        Returns:
            skeleton: (B, 1, D, H, W) soft skeleton
        """
        skeleton = torch.zeros_like(x)

        for i in range(self.num_iterations):
            # Erode
            eroded = F.conv3d(x, self.kernel, padding=1)
            eroded = (eroded > 0.5).float()

            # Skeleton points are where erosion removes voxels
            # but the point was in the original
            diff = x - eroded
            skeleton = torch.max(skeleton, diff * (i + 1) / self.num_iterations)

            x = eroded

            # Stop if fully eroded
            if x.sum() == 0:
                break

        return skeleton


class ClDiceLoss(nn.Module):
    """
    Centerline Dice Loss (clDice) for topology-preserving segmentation.

    clDice measures the overlap between the skeletons/centerlines of the
    prediction and ground truth, encouraging topologically correct predictions.

    Reference:
    "clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"
    https://arxiv.org/abs/2003.07311
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        ignore_index: int = 2,
        alpha: float = 0.5,  # Weight for clDice vs regular Dice
        num_iterations: int = 10
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.skeletonize = SoftSkeletonize3D(num_iterations)

    def soft_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice score."""
        intersection = (pred * target).sum()
        return (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, D, H, W) - logits
            target: (B, D, H, W) - class indices
        Returns:
            Scalar loss combining Dice and clDice
        """
        # Get probabilities for foreground class
        pred_prob = F.softmax(pred, dim=1)[:, 1:2]  # (B, 1, D, H, W)

        # Create binary target for foreground
        target_binary = (target == 1).float().unsqueeze(1)  # (B, 1, D, H, W)

        # Create valid mask
        valid_mask = (target != self.ignore_index).float().unsqueeze(1)
        pred_prob = pred_prob * valid_mask
        target_binary = target_binary * valid_mask

        # Regular Dice
        dice = self.soft_dice(pred_prob, target_binary)

        # Compute skeletons
        pred_skeleton = self.skeletonize(pred_prob)
        target_skeleton = self.skeletonize(target_binary)

        # Topology Precision: skeleton of prediction inside ground truth
        tprec = (pred_skeleton * target_binary).sum() / (pred_skeleton.sum() + self.smooth)

        # Topology Sensitivity: skeleton of ground truth inside prediction
        tsens = (target_skeleton * pred_prob).sum() / (target_skeleton.sum() + self.smooth)

        # clDice = harmonic mean of tprec and tsens
        cl_dice = (2.0 * tprec * tsens + self.smooth) / (tprec + tsens + self.smooth)

        # Combined loss
        combined_dice = self.alpha * cl_dice + (1 - self.alpha) * dice

        return 1.0 - combined_dice


class ComboLoss(nn.Module):
    """
    Combo Loss: Weighted combination of Dice and Cross-Entropy.

    This is the loss used by top solutions in the competition.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-5,
        ignore_index: int = 2
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, D, H, W) - logits
            target: (B, D, H, W) - class indices
        """
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce


class TopologyAwareLoss(nn.Module):
    """
    Combined loss with topology preservation.

    Combines:
    - Combo Loss (Dice + CE) for voxel accuracy
    - clDice Loss for topology preservation
    """

    def __init__(
        self,
        combo_weight: float = 0.7,
        cldice_weight: float = 0.3,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-5,
        ignore_index: int = 2,
        cldice_alpha: float = 0.5,
        cldice_iterations: int = 10
    ):
        super().__init__()
        self.combo_weight = combo_weight
        self.cldice_weight = cldice_weight

        self.combo_loss = ComboLoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            smooth=smooth,
            ignore_index=ignore_index
        )
        self.cldice_loss = ClDiceLoss(
            smooth=smooth,
            ignore_index=ignore_index,
            alpha=cldice_alpha,
            num_iterations=cldice_iterations
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        combo = self.combo_loss(pred, target)
        cldice = self.cldice_loss(pred, target)
        return self.combo_weight * combo + self.cldice_weight * cldice


# Convenience function to get loss by name
def get_loss(name: str, **kwargs) -> nn.Module:
    """Get loss function by name."""
    losses = {
        'dice': DiceLoss,
        'combo': ComboLoss,
        'cldice': ClDiceLoss,
        'topology': TopologyAwareLoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    return losses[name](**kwargs)
