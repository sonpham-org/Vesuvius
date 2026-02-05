#!/usr/bin/env python3
"""
Create and submit inference for Vesuvius Challenge.

Usage:
    python submit.py --checkpoint outputs/best.pth --message "My submission"
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import yaml

from src.models import get_model
from src.inference import run_inference


def main():
    parser = argparse.ArgumentParser(description='Create Vesuvius submission')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--test-csv', type=str, default='./data/test.csv',
                        help='Path to test.csv')
    parser.add_argument('--output-dir', type=str, default='./submissions',
                        help='Output directory for submission')
    parser.add_argument('--message', type=str, default='',
                        help='Submission message/description')
    parser.add_argument('--submit', action='store_true',
                        help='Actually submit to Kaggle')

    # Inference settings
    parser.add_argument('--patch-size', type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation')
    parser.add_argument('--no-postprocess', action='store_true',
                        help='Disable post-processing')

    # Post-processing params
    parser.add_argument('--t-low', type=float, default=0.30)
    parser.add_argument('--t-high', type=float, default=0.80)
    parser.add_argument('--z-radius', type=int, default=3)
    parser.add_argument('--xy-radius', type=int, default=2)
    parser.add_argument('--dust-size', type=int, default=100)

    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Try to load config from checkpoint directory
    checkpoint_dir = Path(args.checkpoint).parent
    config_path = checkpoint_dir / 'config.yaml'

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config.get('model', 'unet')
        out_channels = config.get('out_channels', 3)
        patch_size = config.get('patch_size', args.patch_size)
    else:
        print("Warning: No config found, using default model settings")
        model_name = 'unet'
        out_channels = 3
        patch_size = args.patch_size

    # Create model
    print(f"Creating {model_name} model...")
    model = get_model(
        name=model_name,
        in_channels=1,
        out_channels=out_channels,
        img_size=tuple(patch_size),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / f'submission_{timestamp}.zip'

    # Run inference
    print("\nRunning inference...")
    run_inference(
        model=model,
        data_dir=args.data_dir,
        test_csv=args.test_csv,
        output_path=str(submission_path),
        roi_size=tuple(patch_size),
        overlap=args.overlap,
        use_tta=not args.no_tta,
        post_process=not args.no_postprocess,
        device=device,
        t_low=args.t_low,
        t_high=args.t_high,
        z_radius=args.z_radius,
        xy_radius=args.xy_radius,
        dust_min_size=args.dust_size,
    )

    print(f"\nSubmission created: {submission_path}")

    # Submit to Kaggle if requested
    if args.submit:
        print("\nSubmitting to Kaggle...")
        message = args.message or f"Submission {timestamp}"

        # Load API token
        kaggle_token = os.environ.get('KAGGLE_API_TOKEN')
        if kaggle_token:
            os.environ['KAGGLE_KEY'] = kaggle_token

        import subprocess
        result = subprocess.run([
            'kaggle', 'competitions', 'submit',
            '-c', 'vesuvius-challenge-surface-detection',
            '-f', str(submission_path),
            '-m', message
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        else:
            print("Submission successful!")
    else:
        print("\nTo submit to Kaggle, run:")
        print(f"  kaggle competitions submit -c vesuvius-challenge-surface-detection "
              f"-f {submission_path} -m 'Your message'")


if __name__ == '__main__':
    main()
