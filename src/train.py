"""
Training script for Vesuvius Challenge Surface Detection.

Usage:
    python src/train.py --config configs/default.yaml
    python src/train.py --model swinunetr --loss topology --epochs 100
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from models import get_model, print_model_summary
from losses import get_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train Vesuvius Surface Detection')

    # Data
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--train-csv', type=str, default='./data/train.csv',
                        help='Path to train.csv')

    # Model
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'unetr', 'swinunetr', 'attention_unet'],
                        help='Model architecture')
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--out-channels', type=int, default=3)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patch-size', type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument('--num-patches', type=int, default=4,
                        help='Patches per volume per epoch')

    # Loss
    parser.add_argument('--loss', type=str, default='topology',
                        choices=['dice', 'combo', 'cldice', 'topology'],
                        help='Loss function')

    # Optimization
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'])
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision')

    # Misc
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')

    args = parser.parse_args()

    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 2) -> Dict[str, float]:
    """Compute segmentation metrics."""
    with torch.no_grad():
        # Get predictions
        pred_classes = pred.argmax(dim=1)

        # Create mask for valid pixels
        valid_mask = target != ignore_index

        # Accuracy
        correct = (pred_classes == target) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()

        # Dice for foreground class (class 1)
        pred_fg = (pred_classes == 1) & valid_mask
        target_fg = (target == 1) & valid_mask

        intersection = (pred_fg & target_fg).sum().float()
        dice = (2 * intersection) / (pred_fg.sum() + target_fg.sum() + 1e-8)

        # IoU for foreground
        union = (pred_fg | target_fg).sum().float()
        iou = intersection / (union + 1e-8)

    return {
        'accuracy': accuracy.item(),
        'dice': dice.item(),
        'iou': iou.item(),
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_metrics = {'accuracy': 0, 'dice': 0, 'iou': 0}
    num_batches = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')

    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass with optional AMP
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Compute metrics
        metrics = compute_metrics(outputs, labels)

        # Update totals
        total_loss += loss.item()
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{metrics["dice"]:.4f}',
        })

    # Average metrics
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()

    total_loss = 0
    total_metrics = {'accuracy': 0, 'dice': 0, 'iou': 0}
    num_batches = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')

    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        metrics = compute_metrics(outputs, labels)

        total_loss += loss.item()
        for k in total_metrics:
            total_metrics[k] += metrics[k]
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{metrics["dice"]:.4f}',
        })

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: Path,
    is_best: bool = False,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }

    # Save latest
    torch.save(checkpoint, output_dir / 'latest.pth')

    # Save best
    if is_best:
        torch.save(checkpoint, output_dir / 'best.pth')

    # Save periodic checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(checkpoint, output_dir / f'epoch_{epoch+1}.pth')


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'{args.model}_{args.loss}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    # Tensorboard
    writer = SummaryWriter(output_dir / 'logs')

    # Data
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        batch_size=args.batch_size,
        patch_size=tuple(args.patch_size),
        num_patches_per_volume=args.num_patches,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Model
    print(f"\nCreating {args.model} model...")
    model = get_model(
        name=args.model,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=tuple(args.patch_size),
    )
    model = model.to(device)
    print_model_summary(model, args.model)

    # Loss
    print(f"Using {args.loss} loss...")
    criterion = get_loss(args.loss)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=1e-7,
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )
    else:
        scheduler = None

    # Warmup scheduler
    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=args.warmup_epochs,
        )

    # AMP scaler
    scaler = GradScaler() if args.amp else None

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0

    if args.resume:
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['metrics'].get('dice', 0)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}\n")

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        # Update scheduler
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        elif scheduler:
            scheduler.step()

        # Log to tensorboard
        for name, value in train_metrics.items():
            writer.add_scalar(f'train/{name}', value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f'val/{name}', value, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            print(f"  New best Dice: {best_dice:.4f}")

        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics, output_dir, is_best
        )

    # Training complete
    print(f"\nTraining complete!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Model saved to: {output_dir}")

    writer.close()


if __name__ == '__main__':
    main()
