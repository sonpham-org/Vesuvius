"""
================================================================================
   VESUVIUS V12 - PRO SUBMISSION

   Based on top-performing kernels (LB 0.537+)
   - TransUNet SEResNeXt50 encoder
   - 4x Rotation TTA
   - Class index 2 for foreground
   - Optimized hysteresis thresholds
================================================================================
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================================
# SETUP PACKAGES
# ============================================================================
var = "/kaggle/input/vesuvius25-packages-offline-installer-v20251226/whls"
if os.path.exists(var):
    print(f"Installing packages from: {var}")
    import subprocess
    subprocess.run([
        "pip", "install", "--quiet",
        f"{var}/keras_nightly-3.12.0.dev2025100703-py3-none-any.whl",
        f"{var}/tifffile-2025.10.16-py3-none-any.whl",
        f"{var}/imagecodecs-2025.11.11-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        f"{var}/medicai-0.0.3-py3-none-any.whl",
        "--no-index",
        "--find-links", var
    ], check=False, capture_output=True)
else:
    print(f"Package directory not found: {var}")

# Protobuf patch
try:
    from google.protobuf import message_factory as _message_factory
    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
        from google.protobuf.message_factory import GetMessageClass
        def _GetPrototype(self, descriptor):
            return GetMessageClass(descriptor)
        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except:
    pass

import keras
from medicai.transforms import Compose, ScaleIntensityRange
from medicai.models import TransUNet
from medicai.utils.inference import SlidingWindowInference

import numpy as np
import pandas as pd
import zipfile
import tifffile
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects

print("="*60)
print("VESUVIUS V12 - PRO SUBMISSION")
print("="*60)
print(f"Keras backend: {keras.config.backend()}, version: {keras.version()}")

# ============================================================================
# CONFIG
# ============================================================================
root_dir = "/kaggle/input/vesuvius-challenge-surface-detection"
test_dir = f"{root_dir}/test_images"
output_dir = "/kaggle/working/submission_masks"
zip_path = "/kaggle/working/submission.zip"
os.makedirs(output_dir, exist_ok=True)

# Model config
NUM_CLASSES = 3
PATCH_SIZE = (160, 160, 160)
OVERLAP = 0.50

# Post-processing config (from baseline kernel LB 0.537)
T_LOW = 0.45
T_HIGH = 0.85
Z_RADIUS = 1
XY_RADIUS = 0
DUST_MIN_SIZE = 100

# ============================================================================
# MODEL
# ============================================================================
def get_model():
    """Load TransUNet with fine-tuned weights."""
    model = TransUNet(
        input_shape=(160, 160, 160, 1),
        encoder_name='seresnext50',
        classifier_activation='softmax',
        num_classes=NUM_CLASSES,
    )

    # Try multiple model paths
    model_paths = [
        "/kaggle/input/train-transunet-baseline-lb-0-537/fine_tuning_epoch_20.weights.h5",
        "/kaggle/input/train-vesuvius-surface-3d-detection-on-tpu/model.weights.h5",
        "/kaggle/input/colab-a-162v4-gpu-transunet-seresnext101-x160/model.weights.h5"
    ]

    for path in model_paths:
        if os.path.exists(path):
            print(f"Loading weights from: {path}")
            model.load_weights(path)
            break

    print(f"Model params: {model.count_params() / 1e6:.1f}M")
    return model

# ============================================================================
# TRANSFORMS
# ============================================================================
def val_transformation(image):
    """Normalize image to [0, 1]."""
    data = {"image": image}
    pipeline = Compose([
        ScaleIntensityRange(
            keys=["image"],
            a_min=0,
            a_max=255,
            b_min=0,
            b_max=1,
            clip=True,
        ),
    ])
    result = pipeline(data)
    return result["image"]

# ============================================================================
# HELPERS
# ============================================================================
def load_volume(path):
    """Load 3D TIFF."""
    vol = tifffile.imread(path)
    vol = vol.astype(np.float32)
    vol = vol[None, ..., None]  # (1, D, H, W, 1)
    return vol

def rot90_volume(vol, k):
    """Rotate volume k times 90° clockwise in HW plane."""
    if vol.ndim == 5:
        return np.rot90(vol, k=-k, axes=(2, 3))
    else:
        return np.rot90(vol, k=-k, axes=(1, 2))

def unrot90_volume(vol, k):
    """Inverse rotation."""
    return rot90_volume(vol, (4 - k) % 4)

def build_anisotropic_struct(z_radius: int, xy_radius: int):
    """Build anisotropic structure for morphology."""
    z, r = z_radius, xy_radius

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

# ============================================================================
# POST-PROCESSING
# ============================================================================
def topo_postprocess(probs, T_low=0.50, T_high=0.90, z_radius=1, xy_radius=0, dust_min_size=100):
    """
    Topology-aware post-processing with 3D hysteresis thresholding.
    """
    # Step 1: 3D Hysteresis
    strong = probs >= T_high
    weak = probs >= T_low

    if not strong.any():
        return np.zeros_like(probs, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(probs, dtype=np.uint8)

    # Step 2: 3D Anisotropic Closing
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    # Step 3: Dust Removal
    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)

# ============================================================================
# INFERENCE WITH TTA
# ============================================================================
print("\nLoading model...")
model = get_model()

pred = SlidingWindowInference(
    model,
    roi_size=PATCH_SIZE,
    num_classes=NUM_CLASSES,
    mode="gaussian",
    overlap=OVERLAP,
    sw_batch_size=1
)

def predict_probs_tta_rot(sample):
    """
    4x rotation TTA: 0°, 90°, 180°, 270°
    Returns averaged probabilities.
    """
    probs_accum = []

    for k in range(4):
        s_rot = rot90_volume(sample, k)
        out = pred(s_rot)  # (1, D, H, W, 3)
        out = np.asarray(out)

        # CLASS INDEX 1 for foreground (from baseline kernel LB 0.537)
        probs = out[0, ..., 1]  # (D, H, W)

        probs = unrot90_volume(probs, k)
        probs_accum.append(probs)

    return np.mean(probs_accum, axis=0)

def predict(sample):
    """Full prediction pipeline with TTA and post-processing."""
    # 4x Rotation TTA
    probs_fg = predict_probs_tta_rot(sample)

    # Post-processing
    final = topo_postprocess(
        probs_fg,
        T_low=T_LOW,
        T_high=T_HIGH,
        z_radius=Z_RADIUS,
        xy_radius=XY_RADIUS,
        dust_min_size=DUST_MIN_SIZE,
    )

    return final

# ============================================================================
# MAIN INFERENCE
# ============================================================================
print("\n" + "="*60)
print("V12 PRO INFERENCE")
print("="*60)
print(f"T_low={T_LOW}, T_high={T_HIGH}")
print(f"4x Rotation TTA enabled")
print(f"Class index: 1 (foreground - baseline approach)")

test_df = pd.read_csv(f"{root_dir}/test.csv")
print(f"\nTest samples: {len(test_df)}")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for idx, row in test_df.iterrows():
        image_id = row["id"]
        tif_path = f"{test_dir}/{image_id}.tif"

        print(f"\n[{idx+1}/{len(test_df)}] Processing: {image_id}.tif")

        # Load and transform
        volume = load_volume(tif_path)
        print(f"    Shape: {volume.shape[1:-1]}")
        volume = val_transformation(volume)

        # Predict
        output = predict(volume)
        print(f"    Positives: {output.sum():,} voxels ({output.sum()/output.size*100:.2f}%)")

        # Save
        out_path = f"{output_dir}/{image_id}.tif"
        tifffile.imwrite(out_path, output.astype(np.uint8))

        z.write(out_path, arcname=f"{image_id}.tif")
        os.remove(out_path)

print("\n" + "="*60)
print("V12 PRO COMPLETE!")
print(f"Submission: {zip_path}")
print("="*60)
