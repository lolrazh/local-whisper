import os
import asyncio
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union, BinaryIO, Tuple, List, Generator
from loguru import logger
import torch
from faster_whisper import WhisperModel
from fastapi import UploadFile

from audio_utils import preprocess_audio, is_valid_audio_format

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
logger.info(f"Using device: {device} with compute type: {compute_type}")

class TranscriptionService:
    def __init__(self, model_path: str = "models/faster-whisper-base.en"):
        """
        Initialize the transcription service with the Faster Whisper model.
        
        Args:
            model_path: Path to the Faster Whisper model
        """
        self.model_path = model_path
        self.model_load_time = 0
        self._load_model()
        logger.info(f"TranscriptionService initialized with model: {model_path}")
        
    def _load_model(self) -> None:
        """Load the Faster Whisper model"""
        logger.info(f"Loading Faster Whisper model from {self.model_path}")
        start_time = time.time()
        
        try:
            # Check if we're using a local path or a model size
            if Path(self.model_path).exists():
                logger.info(f"Loading model from local directory: {self.model_path}")
                # Load the model from a local directory
                self.model = WhisperModel(
                    self.model_path,
                    device=device,
                    compute_type=compute_type
                )
            else:
                # Use model size (e.g., "base", "small", "medium", "large-v3")
                logger.info(f"Using model size: {self.model_path}")
                self.model = WhisperModel(
                    self.model_path,
                    device=device,
                    compute_type=compute_type
                )
            
            self.model_load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {self.model_load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load Faster Whisper model: {e}")
    
    async def transcribe_file_upload(self, file: UploadFile, filename: str, return_metrics: bool = False) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Transcribe audio from an uploaded file with detailed performance metrics.
        
        Args:
            file: Audio file from FastAPI UploadFile
            filename: Original filename with extension
            return_metrics: Whether to return metrics as separate object
            
        Returns:
            Transcription result with performance metrics, and optionally a separate metrics dict
        """
        metrics = {
            "model_load": self.model_load_time,
            "validation": 0,
            "audio_preprocessing": 0,
            "audio_loading": 0,
            "model_inference": 0,
            "decoding": 0,
            "postprocessing": 0,
            "cleanup": 0,
            "overhead": 0,
            "total": 0
        }
        
        total_start = time.time()
        
        # Validate file format
        validation_start = time.time()
        if not is_valid_audio_format(filename):
            logger.error(f"Invalid audio format: {filename}")
            raise ValueError(f"Invalid audio format. Supported formats: mp3, wav, flac, m4a, ogg, aac, webm")
        metrics["validation"] = time.time() - validation_start
        logger.debug(f"File validation completed in {metrics['validation']*1000:.2f}ms")
        
        try:
            # Preprocess audio
            preprocessing_start = time.time()
            _, audio_path = await preprocess_audio(file, filename)
            metrics["audio_preprocessing"] = time.time() - preprocessing_start
            logger.debug(f"Audio preprocessing completed in {metrics['audio_preprocessing']*1000:.2f}ms")
            
            # Transcribe audio
            transcription_result, transcription_metrics = self._transcribe_audio_file(str(audio_path), return_metrics=True)
            
            # Update metrics from transcription
            metrics.update(transcription_metrics)
            
            # Total processing time
            metrics["total"] = time.time() - total_start
            
            # Format response
            response = {
                "text": transcription_result,
                "language": "en",  # Assuming English for now
                "segments": [],  # We'll add segments in a more complex implementation
                "total_ms": int(metrics["total"] * 1000),
                "metrics": {
                    "model_load_ms": int(metrics["model_load"] * 1000),
                    "preprocessing_ms": int(metrics["audio_preprocessing"] * 1000),
                    "inference_ms": int(metrics["model_inference"] * 1000),
                    "total_ms": int(metrics["total"] * 1000)
                }
            }
            
            if return_metrics:
                return response, metrics
            return response
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def _transcribe_audio_file(self, audio_path: str, return_metrics: bool = False) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        Transcribe an audio file using Faster Whisper.
        
        Args:
            audio_path: Path to the audio file
            return_metrics: Whether to return detailed metrics
            
        Returns:
            Transcription text and optionally metrics
        """
        metrics = {
            "audio_loading": 0,
            "model_inference": 0,
            "decoding": 0,
            "postprocessing": 0,
            "cleanup": 0
        }
        
        try:
            # Transcribe with the model
            inference_start = time.time()
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                language="en",  # Force English for now
                vad_filter=True,  # Voice activity detection to filter out silence
                vad_parameters=dict(min_silence_duration_ms=500)  # Minimum duration of silence to consider it a break
            )
            
            # Force execution of the generator
            segments_list = list(segments)
            metrics["model_inference"] = time.time() - inference_start
            
            # Process segments to create final text
            postprocessing_start = time.time()
            full_text = " ".join([segment.text.strip() for segment in segments_list])
            metrics["postprocessing"] = time.time() - postprocessing_start
            
            logger.info(f"Transcription completed with {len(segments_list)} segments")
            logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            
            # Cleanup temporary directories
            cleanup_start = time.time()
            # Note: We don't clean up audio_path here because it's managed by the caller
            metrics["cleanup"] = time.time() - cleanup_start
            
            if return_metrics:
                return full_text, metrics
            return full_text
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise RuntimeError(f"Transcription process failed: {e}")
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Get information about available models.
        
        Returns:
            Dictionary with model information
        """
        models = [
            {"id": "base.en", "name": "Faster Whisper Base (English)", "size_mb": 150},
            {"id": "small.en", "name": "Faster Whisper Small (English)", "size_mb": 500},
            {"id": "medium.en", "name": "Faster Whisper Medium (English)", "size_mb": 1500},
            {"id": "large-v3", "name": "Faster Whisper Large V3", "size_mb": 3000}
        ]
        
        return {
            "models": models,
            "current_model": {
                "id": os.path.basename(self.model_path),
                "name": f"Faster Whisper {os.path.basename(self.model_path)}",
                "device": device,
                "compute_type": compute_type
            }
        } 