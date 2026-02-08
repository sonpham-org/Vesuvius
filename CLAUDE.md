# Vesuvius Challenge - Surface Detection

Competition: https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection
Deadline: Feb 13, 2026
Prize: $200,000

## Task
Detect papyrus surfaces in 3D CT scans of ancient Herculaneum scrolls.

## Evaluation Metric
Topology-aware weighted average of:
- Surface Dice (voxel accuracy)
- TopoScore (surface connectivity)
- VOI (Variation of Information)


## Project Structure

```
Vesuvius/
├── data/                    # Competition data (gitignored)
│   ├── train_images/        # Training 3D TIFF volumes
│   ├── train_labels/        # Training masks
│   ├── test_images/         # Test volumes
│   ├── train.csv
│   └── test.csv
├── src/                     # Source code
│   ├── dataset.py           # Data loading & augmentation
│   ├── models.py            # Model architectures (MONAI)
│   ├── losses.py            # Loss functions (Dice, clDice, Topology)
│   ├── inference.py         # Sliding window + TTA + post-processing
│   └── train.py             # Training script
├── configs/                 # Training configs
│   ├── default.yaml         # UNet + topology loss
│   └── swinunetr.yaml       # SwinUNETR config
├── notebooks/               # Downloaded Kaggle notebooks for reference
├── outputs/                 # Training outputs (gitignored)
├── submissions/             # Submission files (gitignored)
├── submit.py                # Submission script
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default config (UNet + topology loss)
python src/train.py --config configs/default.yaml

# Train SwinUNETR
python src/train.py --config configs/swinunetr.yaml

# Create submission
python submit.py --checkpoint outputs/best.pth

# Submit to Kaggle
python submit.py --checkpoint outputs/best.pth --submit --message "My submission"
```

## Key Implementation Details

### Loss Functions (src/losses.py)
- **TopologyAwareLoss**: Combines ComboLoss (Dice + CE) with clDice for topology preservation
- **clDice**: Centerline Dice that measures skeleton overlap, penalizes gaps/holes

### Post-Processing (src/inference.py)
- 3D Hysteresis thresholding (T_low=0.30, T_high=0.80)
- Anisotropic morphological closing (z=3, xy=2)
- Dust removal (min_size=100)

### Test-Time Augmentation
- 8 views: original + 3 flips + 3 rotations
- Logit averaging before argmax

## Best Practices from Top Solutions

1. **Single model often beats ensembles** - aggressive post-processing compensates
2. **Combo loss** (Dice + CE) is the baseline
3. **Topology preservation** is crucial due to the metric
4. **TransUNet with SEResNeXt50** encoder is the current best architecture
5. **Patch size 160x160x160** with 40-50% overlap for sliding window
