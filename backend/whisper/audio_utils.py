import os
import tempfile
import subprocess
from pathlib import Path
import inspect
import time
from typing import BinaryIO, Tuple, Union, Any
from loguru import logger
from fastapi import UploadFile


async def preprocess_audio(file: Any, original_filename: str) -> Tuple[str, Path]:
    """
    Preprocess audio for optimal transcription with CPU optimization.
    - Downsample to 16KHz
    - Convert to mono channel
    - Convert to WAV for compatibility
    
    Args:
        file: Audio file binary data or object with async read() method
        original_filename: Original filename with extension
    
    Returns:
        Tuple with the new filename and the path to the processed file
    """
    preproc_start = time.time()
    
    # Use RAM-based temp directory if available for faster I/O
    temp_dir = tempfile.mkdtemp(dir='/dev/shm' if os.path.exists('/dev/shm') else None)
    
    # Get the original file extension
    _, original_ext = os.path.splitext(original_filename)
    
    # Generate temporary file paths - use a more efficient naming scheme
    temp_input_path = os.path.join(temp_dir, f"in{original_ext}")
    
    # Handle file content in the most efficient way based on type
    try:
        if hasattr(file, 'file') and hasattr(file.file, 'read'):
            # FastAPI UploadFile optimization - use direct file access when possible
            # This avoids double buffering in memory
            with open(temp_input_path, "wb") as temp_file:
                temp_file.write(file.file.read())
        elif hasattr(file, 'read') and inspect.iscoroutinefunction(file.read):
            # For async file objects
            content = await file.read()
            with open(temp_input_path, "wb") as temp_file:
                temp_file.write(content)
        elif hasattr(file, 'read'):
            # For synchronous file-like objects
            with open(temp_input_path, "wb") as temp_file:
                temp_file.write(file.read())
        else:
            logger.error(f"Invalid file object: {type(file)}")
            raise ValueError("File object must have a read method")
    
        # Create the output filename - shorter for filesystem efficiency
        output_filename = f"proc_{os.path.basename(original_filename).split('.')[0]}.wav"
        temp_output_path = os.path.join(temp_dir, output_filename)
    
        # Convert audio to 16KHz mono WAV with optimized ffmpeg settings
        # Performance-optimized ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", temp_input_path,
            "-ar", "16000",     # Set sample rate to 16kHz
            "-ac", "1",         # Set to mono channel
            "-c:a", "pcm_s16le", # Use 16-bit PCM for WAV
            "-map_metadata", "-1", # Strip metadata to reduce processing time
            "-fflags", "+bitexact", # Ensure deterministic output
            "-flags:v", "+bitexact", # More bitexact flags
            "-flags:a", "+bitexact", # For audio bitexactness
            "-nostdin",         # Avoid stdin checks (faster)
            "-threads", "1",    # Single thread is often faster for small files
            temp_output_path,
            "-y",               # Overwrite if exists
            "-loglevel", "error" # Minimize logging
        ]
    
        # Run ffmpeg with high priority
        start_time = time.time()
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            bufsize=10**7  # Larger buffer size for better performance
        )
        _, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"ffmpeg error: {stderr.decode()}")
            raise RuntimeError(f"Audio preprocessing failed: {stderr.decode()}")
            
        preproc_time = time.time() - preproc_start
        logger.info(f"Audio preprocessing completed in {preproc_time*1000:.2f}ms")
        
        # Delete input file immediately to free up space
        try:
            os.unlink(temp_input_path)
        except Exception as e:
            logger.debug(f"Could not remove temp input file: {e}")
        
        return output_filename, Path(temp_output_path)
    
    except Exception as e:
        # Cleanup on error
        try:
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass
        
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