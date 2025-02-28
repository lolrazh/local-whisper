import os
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, BinaryIO, Tuple
from loguru import logger
import numpy as np
from scipy.io import wavfile
from fastapi import UploadFile
from faster_whisper import WhisperModel

from audio_utils import preprocess_audio, is_valid_audio_format

# Check if CUDA is available (through faster_whisper)
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using optimized compute types for each platform
    if device == "cuda":
        compute_type = "float16"  # Best for GPU
    else:
        compute_type = "int8"  # Best for CPU - INT8 quantization
    logger.info(f"Using device: {device}, compute_type: {compute_type}")
except ImportError:
    device = "cpu"
    compute_type = "int8"  # Best for CPU - INT8 quantization
    logger.info(f"Torch not found, using device: {device}, compute_type: {compute_type}")


class TranscriptionService:
    def __init__(self, model_size: str = "base", model_path: str = None):
        """
        Initialize the transcription service with Faster Whisper.
        
        Args:
            model_size: Size of the Whisper model ("tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3")
            model_path: Path to a local CTranslate2 model (overrides model_size if provided)
        """
        self.model_size = model_size
        self.model_path = model_path
        self.model_load_time = 0
        self._load_model()
        if model_path:
            logger.info(f"TranscriptionService initialized with model path: {model_path}")
        else:
            logger.info(f"TranscriptionService initialized with model size: {model_size}")
        
    def _load_model(self) -> None:
        """Load the Faster Whisper model"""
        start_time = time.time()
        
        try:
            # Get absolute path to workspace root (two levels up from this file)
            workspace_root = Path(__file__).parent.parent.parent
            
            if self.model_path:
                # Use local model path (relative to workspace root)
                model_path = os.path.join(workspace_root, self.model_path)
                logger.info(f"Loading Faster Whisper model from local path: {model_path}")
                
                # Check if the model exists at the specified path
                if not os.path.exists(model_path):
                    logger.error(f"Model not found at path: {model_path}")
                    raise FileNotFoundError(f"Model not found at: {model_path}")
                    
                # Check if it's a valid CTranslate2 model
                if not os.path.exists(os.path.join(model_path, "model.bin")):
                    logger.warning(f"The path {model_path} might not be a valid CTranslate2 model (no model.bin found)")
                
                logger.info(f"Using local model at absolute path: {model_path}")
                
                # Load the model with appropriate parameters based on device
                self.model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=min(os.cpu_count(), 4),  # Limit CPU threads to avoid excessive usage
                    num_workers=2  # Default is 4, slightly reduced for better stability
                )
            else:
                # Download and load the model by size
                logger.info(f"Loading Faster Whisper model of size: {self.model_size}")
                self.model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                    download_root=os.path.join(workspace_root, "models"),
                    cpu_threads=min(os.cpu_count(), 4),  # Limit CPU threads
                    num_workers=2  # Default is 4, slightly reduced for better stability
                )
            
            self.model_load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {self.model_load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    async def transcribe_file_upload(self, file: UploadFile, filename: str, return_metrics: bool = True) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Transcribe an uploaded audio file using Faster Whisper.
        
        Args:
            file: The uploaded audio file
            filename: Original filename
            return_metrics: Whether to return performance metrics
            
        Returns:
            Dictionary with transcription results and metrics
        """
        logger.info(f"Processing file: {filename}")
        
        start_time = time.time()
        
        # Verify file format
        if not is_valid_audio_format(filename):
            error_msg = f"Invalid audio format: {filename}. Supported formats: mp3, wav, m4a, flac, ogg, aac, webm"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Preprocess the audio file
            preprocess_start = time.time()
            _, audio_path = await preprocess_audio(file, filename)
            preprocess_time = time.time() - preprocess_start
            logger.info(f"Audio preprocessing completed in {preprocess_time:.2f}s")
            
            # Transcribe the processed audio
            transcribe_start = time.time()
            segments, info = self.model.transcribe(
                str(audio_path), 
                beam_size=5,  # Standard beam size for good accuracy/speed balance
                language="en",  # Specify English to avoid language detection overhead
                vad_filter=True,  # Enable VAD to skip silent parts
                vad_parameters=dict(min_silence_duration_ms=500),  # Optimize VAD params
                word_timestamps=True  # Keep word timestamps for detailed results
            )
            
            # Convert segments to a list format but collect them in memory first (avoid generator overhead)
            result_segments = []
            full_text = ""
            
            # Materialize the generator to avoid repeated processing
            segments_list = list(segments)
            
            for segment in segments_list:
                result_segments.append({
                    "id": len(result_segments),
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [{"word": word.word, "start": word.start, "end": word.end, "probability": word.probability} 
                             for word in (segment.words or [])],
                })
                full_text += segment.text + " "
                
            full_text = full_text.strip()
            transcribe_time = time.time() - transcribe_start
            
            # Prepare result data
            total_time = time.time() - start_time
            
            result = {
                "text": full_text,
                "segments": result_segments,
                "language": info.language,
                "language_probability": info.language_probability,
                "total_duration": info.duration if hasattr(info, "duration") else 0,
                "total_ms": int(total_time * 1000),
                "inference_ms": int(transcribe_time * 1000),
                "preprocessing_ms": int(preprocess_time * 1000),
                "model_info": {
                    "model_size": self.model_size,
                    "model_path": self.model_path,
                    "device": device,
                    "compute_type": compute_type
                }
            }
            
            logger.info(f"Transcription completed in {total_time:.2f}s (inference: {transcribe_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
        finally:
            # Cleanup temporary files
            try:
                if audio_path and audio_path.exists():
                    # Delete parent directory which contains all temp files
                    shutil.rmtree(audio_path.parent)
                    logger.debug(f"Cleaned up temporary files in {audio_path.parent}")
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available Faster Whisper models"""
        available_models = [
            {"id": "tiny", "name": "Tiny (39M parameters)"},
            {"id": "base", "name": "Base (74M parameters)"},
            {"id": "small", "name": "Small (244M parameters)"},
            {"id": "medium", "name": "Medium (769M parameters)"},
            {"id": "large-v1", "name": "Large v1 (1550M parameters)"},
            {"id": "large-v2", "name": "Large v2 (1550M parameters)"},
            {"id": "large-v3", "name": "Large v3 (1550M parameters)"}
        ]
        
        return {
            "models": available_models,
            "current_model": {
                "id": self.model_size,
                "path": self.model_path,
                "device": device,
                "compute_type": compute_type
            }
        } 