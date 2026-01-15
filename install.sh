#!/bin/bash
# VoiceHub Backend Installation Script
# Sets up conda environment and installs all dependencies

set -e  # Exit on error

echo "=================================================="
echo "VoiceHub Backend Installation"
echo "=================================================="

# Configuration
CONDA_ENV_NAME="voicehub"
PYTHON_VERSION="3.10"
MODEL_NAME="Fun-CosyVoice3-0.5B"
MODEL_DIR="CosyVoice/pretrained_models/${MODEL_NAME}"
MIRROR_URL="https://mirrors.aliyun.com/pypi/simple/"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 1. Create conda environment
echo ""
echo "[1/5] Creating conda environment..."
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "⚠️  Environment ${CONDA_ENV_NAME} already exists"
else
    conda create -n ${CONDA_ENV_NAME} -y python=${PYTHON_VERSION}
    echo "✅ Conda environment created"
fi

# Activate environment
echo ""
echo "[2/5] Activating environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

# 3. Install Python dependencies
echo ""
echo "[3/5] Installing Python dependencies..."
pip install -r requirements.txt -i ${MIRROR_URL} --trusted-host=mirrors.aliyun.com

# Fix ruamel.yaml compatibility
echo ""
echo "[4/5] Fixing dependency compatibility..."
pip install "ruamel.yaml<0.18" -i ${MIRROR_URL} --trusted-host=mirrors.aliyun.com -q
echo "✅ Dependencies fixed"

# 5. Setup CosyVoice as submodule
echo ""
echo "[5/5] Setting up CosyVoice..."
if [ -d "CosyVoice" ]; then
    echo "⚠️  CosyVoice directory already exists"
else
    echo "Adding CosyVoice as git submodule..."
    git submodule add https://github.com/FunAudioLLM/CosyVoice.git CosyVoice
    cd CosyVoice
    git submodule update --init --recursive
    cd ..
    echo "✅ CosyVoice submodule added"
fi

# Optional: Download model
echo ""
read -p "Download CosyVoice model now? (Model size: ~1GB) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading ${MODEL_NAME} model..."
    mkdir -p pretrained_models

    python3 << EOF
import sys
sys.path.append('CosyVoice/third_party/Matcha-TTS')
from modelscope import snapshot_download

print(f"Downloading {MODEL_NAME}...")
snapshot_download('FunAudioLLM/${MODEL_NAME}-2512', local_dir='${MODEL_DIR}')
print(f"✅ Model downloaded: {MODEL_DIR}")
EOF
else
    echo "⏭️  Skipping model download (will auto-download on first run)"
fi

# Installation complete
echo ""
echo "=================================================="
echo "✅ Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the environment:"
echo "     conda activate ${CONDA_ENV_NAME}"
echo ""
echo "  2. Start the backend server:"
echo "     python server.py --port 9880"
echo ""
echo "  3. Access API documentation:"
echo "     http://localhost:9880/docs"
echo ""
echo "=================================================="
