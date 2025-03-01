import os
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Any
from loguru import logger
from fastapi import UploadFile


async def preprocess_audio(file: UploadFile, original_filename: str) -> Tuple[Path, float]:
    """
    Simple and reliable audio preprocessing for transcription.
    
    Args:
        file: Audio file upload
        original_filename: Original filename with extension
        
    Returns:
        Path to processed audio file and processing time
    """
    import time
    start_time = time.time()
    
    # Create a temp directory
    temp_dir = tempfile.mkdtemp()
    
    # Save the uploaded file
    temp_input_path = os.path.join(temp_dir, original_filename)
    content = await file.read()
    with open(temp_input_path, "wb") as f:
        f.write(content)
    
    # Create output path
    output_filename = f"processed_{os.path.splitext(original_filename)[0]}.wav"
    output_path = os.path.join(temp_dir, output_filename)
    
    # Convert using FFmpeg (simple, direct call)
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output files
        "-i", temp_input_path,  # Input file
        "-ar", "16000",  # Sample rate
        "-ac", "1",  # Mono
        "-c:a", "pcm_s16le",  # Output codec
        output_path  # Output file
    ]
    
    # Run FFmpeg directly, with full error output
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Check if output exists
        if not os.path.exists(output_path):
            raise RuntimeError("FFmpeg failed to create output file")
            
        processing_time = time.time() - start_time
        return Path(output_path), processing_time
        
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error: {e.stderr}"
        if "Invalid data found when processing input" in e.stderr:
            error_msg = "Audio file appears to be corrupt or invalid format"
        
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        # Simple error handling
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise RuntimeError(f"Audio preprocessing failed: {str(e)}")


def is_valid_audio_format(filename: str) -> bool:
    """Check if file has valid audio extension."""
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