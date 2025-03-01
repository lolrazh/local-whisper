import os
import tempfile
import subprocess
import time
from pathlib import Path
from typing import BinaryIO, Tuple
from loguru import logger


def preprocess_audio(file: BinaryIO, original_filename: str) -> Tuple[Path, float]:
    """
    Preprocess audio for optimal transcription.
    - Downsample to 16KHz
    - Convert to mono channel
    - Convert to FLAC format for lossless compression
    
    Args:
        file: Audio file binary data
        original_filename: Original filename with extension
    
    Returns:
        Tuple with the path to the processed file and processing time in seconds
    """
    start_time = time.time()
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Get the original file extension
    _, original_ext = os.path.splitext(original_filename)
    
    # Save the uploaded file temporarily
    temp_input_path = os.path.join(temp_dir, f"input{original_ext}")
    with open(temp_input_path, "wb") as temp_file:
        temp_file.write(file.read())
    
    # Create the output filename
    output_filename = f"processed_{os.path.basename(original_filename).split('.')[0]}.flac"
    temp_output_path = os.path.join(temp_dir, output_filename)
    
    # Convert audio to 16KHz mono FLAC
    try:
        cmd = [
            "ffmpeg",
            "-i", temp_input_path,
            "-ar", "16000",  # Set sample rate to 16kHz
            "-ac", "1",      # Set to mono channel
            "-map", "0:a",   # Map only audio stream
            "-c:a", "flac",  # Use FLAC codec
            temp_output_path,
            "-y",            # Overwrite if exists
            "-loglevel", "error"  # Minimize logging
        ]
        
        subprocess.run(cmd, check=True)
        logger.info(f"Audio preprocessing complete: {output_filename}")
        
        processing_time = time.time() - start_time
        return Path(temp_output_path), processing_time
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error preprocessing audio: {e}")
        raise RuntimeError(f"Audio preprocessing failed: {e}")


def is_valid_audio_format(filename: str) -> bool:
    """
    Check if the file format is a valid audio format.
    
    Args:
        filename: The name of the file to check
    
    Returns:
        Boolean indicating if the format is valid
    """
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.webm']
    _, ext = os.path.splitext(filename.lower())
    return ext in valid_extensions


def get_file_size_mb(file_path: str) -> float:
    """
    Get the file size in megabytes.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Size in MB
    """
    return os.path.getsize(file_path) / (1024 * 1024) 