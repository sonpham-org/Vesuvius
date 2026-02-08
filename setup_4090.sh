#!/bin/bash
# Setup script for RTX 4090 machine
# Run this once to set up the environment

set -e

echo "=== Vesuvius Viz Setup for RTX 4090 ==="

# 1. Create venv
uv venv .venv --python 3.12
echo "Created venv"

# 2. Install JAX with CUDA 12
uv pip install --python .venv/bin/python "jax[cuda12]"
echo "Installed JAX CUDA"

# 3. Install exact Kaggle versions of keras + medicai
# Download the offline medicai wheel from Kaggle first:
#   source .env && export KAGGLE_API_TOKEN
#   kaggle kernels output ipythonx/vsdetection-packages-offline-installer-only -p /tmp/kaggle-whls
# Then install:
if [ -f /tmp/kaggle-whls/whls/medicai-0.0.3-py3-none-any.whl ]; then
    uv pip install --python .venv/bin/python \
        /tmp/kaggle-whls/whls/keras_nightly-3.12.0.dev2025100703-py3-none-any.whl \
        /tmp/kaggle-whls/whls/medicai-0.0.3-py3-none-any.whl
else
    echo "WARNING: Kaggle medicai wheel not found. Falling back to PyPI (may not load weights)."
    echo "Run: source .env && export KAGGLE_API_TOKEN && kaggle kernels output ipythonx/vsdetection-packages-offline-installer-only -p /tmp/kaggle-whls"
    uv pip install --python .venv/bin/python "keras-nightly==3.12.0.dev2025100703" medicai
fi

# 4. Install remaining deps
uv pip install --python .venv/bin/python tifffile scipy scikit-image matplotlib pandas tensorflow-cpu
echo "Installed dependencies"

# 5. Download model weights
if [ ! -f weights/transunet.seresnext50.160px.comboloss.weights.h5 ]; then
    echo "Downloading model weights..."
    source .env && export KAGGLE_API_TOKEN
    kaggle models instances versions download ipythonx/vsd-model/Keras/transunet/3 -p weights/ --untar
fi
echo "Weights ready"

echo ""
echo "=== Setup complete! ==="
echo "Run visualization:"
echo "  KERAS_BACKEND=jax .venv/bin/python src/local_inference_viz.py"
echo ""
echo "NOTE: If weights fail to load, the medicai version might not match."
echo "The model weights require the EXACT medicai wheel from Kaggle's offline installer."
echo "Make sure /tmp/kaggle-whls/whls/medicai-0.0.3-py3-none-any.whl exists before running setup."
