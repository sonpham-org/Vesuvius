"""
Model architectures for Vesuvius Challenge Surface Detection.

Includes:
- MONAI UNet (baseline)
- MONAI UNETR (transformer-based)
- MONAI SwinUNETR (Swin Transformer-based)
"""

from typing import Optional, Tuple, Sequence
import torch
import torch.nn as nn

try:
    from monai.networks.nets import UNet, UNETR, SwinUNETR, AttentionUnet
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not available. Install with: pip install monai")


def get_model(
    name: str,
    in_channels: int = 1,
    out_channels: int = 3,
    spatial_dims: int = 3,
    img_size: Tuple[int, int, int] = (160, 160, 160),
    **kwargs
) -> nn.Module:
    """
    Get model by name.

    Args:
        name: Model name ('unet', 'unetr', 'swinunetr', 'attention_unet')
        in_channels: Number of input channels
        out_channels: Number of output classes
        spatial_dims: Number of spatial dimensions (3 for 3D)
        img_size: Input image size (required for transformer models)
        **kwargs: Additional model-specific arguments

    Returns:
        PyTorch model
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required. Install with: pip install monai")

    name = name.lower()

    if name == 'unet':
        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=kwargs.get('channels', (32, 64, 128, 256, 512)),
            strides=kwargs.get('strides', (2, 2, 2, 2)),
            num_res_units=kwargs.get('num_res_units', 2),
            dropout=kwargs.get('dropout', 0.0),
        )

    elif name == 'unetr':
        model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=kwargs.get('feature_size', 16),
            hidden_size=kwargs.get('hidden_size', 768),
            mlp_dim=kwargs.get('mlp_dim', 3072),
            num_heads=kwargs.get('num_heads', 12),
            dropout_rate=kwargs.get('dropout_rate', 0.0),
            spatial_dims=spatial_dims,
        )

    elif name == 'swinunetr':
        model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=kwargs.get('feature_size', 48),
            depths=kwargs.get('depths', (2, 2, 2, 2)),
            num_heads=kwargs.get('num_heads', (3, 6, 12, 24)),
            drop_rate=kwargs.get('drop_rate', 0.0),
            attn_drop_rate=kwargs.get('attn_drop_rate', 0.0),
            dropout_path_rate=kwargs.get('dropout_path_rate', 0.0),
            spatial_dims=spatial_dims,
        )

    elif name == 'attention_unet':
        model = AttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=kwargs.get('channels', (32, 64, 128, 256, 512)),
            strides=kwargs.get('strides', (2, 2, 2, 2)),
            dropout=kwargs.get('dropout', 0.0),
        )

    else:
        raise ValueError(
            f"Unknown model: {name}. "
            f"Available: unet, unetr, swinunetr, attention_unet"
        )

    return model


class EnsembleModel(nn.Module):
    """Ensemble of multiple models with weighted averaging."""

    def __init__(
        self,
        models: Sequence[nn.Module],
        weights: Optional[Sequence[float]] = None
    ):
        super().__init__()
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model, weight in zip(self.models, self.weights):
            out = model(x)
            outputs.append(out * weight)
        return sum(outputs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, name: str = "Model"):
    """Print model summary."""
    total_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Total trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"{'='*60}\n")


# Test models
if __name__ == "__main__":
    if MONAI_AVAILABLE:
        # Test each model
        x = torch.randn(1, 1, 160, 160, 160)

        for name in ['unet', 'unetr', 'swinunetr', 'attention_unet']:
            print(f"\nTesting {name}...")
            try:
                model = get_model(name)
                with torch.no_grad():
                    y = model(x)
                print(f"  Input: {x.shape} -> Output: {y.shape}")
                print_model_summary(model, name)
            except Exception as e:
                print(f"  Error: {e}")
