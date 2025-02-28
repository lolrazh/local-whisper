"""
Script to convert a Whisper model to CTranslate2 format.
This is useful if you're experiencing issues with an existing model.
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger

def convert_model(input_model: str, output_dir: str, quantization: str = "float16"):
    """
    Convert a Whisper model to CTranslate2 format.
    
    Args:
        input_model: Path to the input Whisper model or model ID (e.g., "openai/whisper-base.en")
        output_dir: Path to save the converted model
        quantization: Quantization type (float16, int8, int8_float16)
    """
    logger.info(f"Converting model: {input_model} to CTranslate2 format")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Quantization: {quantization}")
    
    try:
        # Install required packages if not already installed
        import importlib
        if importlib.util.find_spec("transformers") is None:
            logger.info("Installing transformers...")
            os.system(f"{sys.executable} -m pip install transformers[torch]>=4.23")
        
        # Import after potential installation
        from ctranslate2.converters import TransformersConverter
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the converter
        converter = TransformersConverter("whisper", quantization=quantization)
        
        # Convert the model
        converter.convert(input_model, output_dir, copy_files=["tokenizer.json", "preprocessor_config.json"])
        
        logger.info(f"Model successfully converted to: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error converting model: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert a Whisper model to CTranslate2 format")
    parser.add_argument("--input", type=str, default="../../models/whisper-base.en", help="Input model path or ID")
    parser.add_argument("--output", type=str, default="../../models/faster-whisper-base.en-converted", help="Output directory")
    parser.add_argument("--quantization", type=str, default="int8_float16", help="Quantization type (float16, int8, int8_float16)")
    
    args = parser.parse_args()
    
    # Convert the model
    success = convert_model(args.input, args.output, args.quantization)
    
    if success:
        logger.info("Conversion completed successfully!")
        logger.info(f"Set MODEL_PATH=\"models/faster-whisper-base.en-converted\" in your environment or config")
    else:
        logger.error("Conversion failed!")
        sys.exit(1) 