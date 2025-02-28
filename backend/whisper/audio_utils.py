import os
import tempfile
import subprocess
from pathlib import Path
import inspect
from typing import BinaryIO, Tuple, Union, Any
from loguru import logger
from fastapi import UploadFile


async def preprocess_audio(file: Any, original_filename: str) -> Tuple[str, Path]:
    """
    Preprocess audio for optimal transcription.
    - Downsample to 16KHz
    - Convert to mono channel
    - Convert to FLAC format for lossless compression
    
    Args:
        file: Audio file binary data or object with async read() method
        original_filename: Original filename with extension
    
    Returns:
        Tuple with the new filename and the path to the processed file
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Get the original file extension
    _, original_ext = os.path.splitext(original_filename)
    
    # Generate temporary file paths
    temp_input_path = os.path.join(temp_dir, f"input{original_ext}")
    
    # Check if the file object has an async read method
    has_async_read = hasattr(file, 'read') and inspect.iscoroutinefunction(file.read)
    
    # Handle file content based on type
    if has_async_read:
        logger.debug(f"Processing file with async read method: {original_filename}")
        content = await file.read()
        with open(temp_input_path, "wb") as temp_file:
            temp_file.write(content)
    elif hasattr(file, 'read'):
        logger.debug(f"Processing file with sync read method: {original_filename}")
        with open(temp_input_path, "wb") as temp_file:
            temp_file.write(file.read())
    else:
        logger.error(f"Invalid file object: {type(file)}")
        raise ValueError("File object must have a read method")
    
    logger.debug(f"Saved input file to {temp_input_path}")
    
    # Create the output filename
    output_filename = f"processed_{os.path.basename(original_filename).split('.')[0]}.wav"
    temp_output_path = os.path.join(temp_dir, output_filename)
    
    # Convert audio to 16KHz mono WAV (better for Whisper)
    try:
        cmd = [
            "ffmpeg",
            "-i", temp_input_path,
            "-ar", "16000",  # Set sample rate to 16kHz
            "-ac", "1",      # Set to mono channel
            "-map", "0:a",   # Map only audio stream
            "-c:a", "pcm_s16le",  # Use 16-bit PCM for WAV
            temp_output_path,
            "-y",            # Overwrite if exists
            "-loglevel", "error"  # Minimize logging
        ]
        
        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info(f"Audio preprocessing complete: {output_filename}")
        
        return output_filename, Path(temp_output_path)
    
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