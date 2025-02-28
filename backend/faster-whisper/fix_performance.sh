#!/bin/bash

# Script to fix Faster Whisper performance issues
# This script will properly convert your Whisper model to CTranslate2 format

# Exit on any error
set -e

echo "===== Faster Whisper Performance Fix ====="
echo "This script will properly convert your Whisper model to CTranslate2 format"

# Get the script directory and workspace root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"

# Check if the virtual environment exists
if [ ! -d "$WORKSPACE_ROOT/venv" ]; then
    echo "Creating virtual environment..."
    python -m venv "$WORKSPACE_ROOT/venv"
fi

# Activate the virtual environment
source "$WORKSPACE_ROOT/venv/bin/activate" 2>/dev/null || source "$WORKSPACE_ROOT/venv/Scripts/activate"

# Install required packages
echo "Installing required packages..."
pip install -r "$SCRIPT_DIR/../requirements.txt"
pip install -r "$SCRIPT_DIR/requirements.txt"
pip install ctranslate2 transformers[torch] --upgrade

# Check if the models directory exists
if [ ! -d "$WORKSPACE_ROOT/models" ]; then
    echo "Creating models directory..."
    mkdir -p "$WORKSPACE_ROOT/models"
fi

# Check if the vanilla whisper model is available
WHISPER_MODEL_DIR="$WORKSPACE_ROOT/models/whisper-base.en"
if [ ! -d "$WHISPER_MODEL_DIR" ]; then
    echo "Downloading Whisper model (base.en)..."
    python -c "from transformers import WhisperForConditionalGeneration, WhisperProcessor; model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base.en'); processor = WhisperProcessor.from_pretrained('openai/whisper-base.en'); model.save_pretrained('$WHISPER_MODEL_DIR'); processor.save_pretrained('$WHISPER_MODEL_DIR')"
fi

# Create the output directory for the converted model
OUTPUT_MODEL_DIR="$WORKSPACE_ROOT/models/faster-whisper-base.en-converted"
if [ -d "$OUTPUT_MODEL_DIR" ]; then
    echo "Removing existing converted model..."
    rm -rf "$OUTPUT_MODEL_DIR"
fi

# Convert the model
echo "Converting Whisper model to CTranslate2 format with int8_float16 quantization..."
python "$SCRIPT_DIR/convert_model.py" --input "$WHISPER_MODEL_DIR" --output "$OUTPUT_MODEL_DIR" --quantization "int8_float16"

# Verify the converted model
if [ -f "$OUTPUT_MODEL_DIR/model.bin" ] && [ -f "$OUTPUT_MODEL_DIR/config.json" ]; then
    echo -e "\n✅ Success! Model converted successfully to CTranslate2 format."
    echo -e "\nYou can now use this model by setting:"
    echo "   MODEL_PATH=\"models/faster-whisper-base.en-converted\""
    echo -e "\nOr simply restart the Faster Whisper server, which will now automatically"
    echo "detect and use this optimized model."
else
    echo -e "\n❌ Error: Conversion failed. The model files are missing."
    exit 1
fi

echo -e "\nTo get maximum performance, make sure to:"
echo "1. Use beam_size=1 for faster transcription (default in updated code)"
echo "2. Keep the optimized computation types (already set in the code)"

echo -e "\nYou can restart the server now." 