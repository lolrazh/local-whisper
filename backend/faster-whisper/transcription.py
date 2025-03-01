import os
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Union, Tuple
from loguru import logger
import torch
from faster_whisper import WhisperModel
from fastapi import UploadFile

from audio_utils import preprocess_audio, is_valid_audio_format

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
logger.info(f"Using device: {device} with compute type: {compute_type}")

# CPU optimization settings
CPU_THREADS = min(os.cpu_count() or 4, 4)
logger.info(f"Using {CPU_THREADS} CPU threads for processing")

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
        """Load the Faster Whisper model with CPU optimizations"""
        logger.info(f"Loading Faster Whisper model from {self.model_path}")
        start_time = time.time()
        
        try:
            # Optimize for CPU performance with CTranslate2 settings
            cpu_options = {
                "cpu_threads": CPU_THREADS,
                "num_workers": 1,  # For predictable latency
            }
            
            # Check if we're using a local path or a model size
            if Path(self.model_path).exists():
                logger.info(f"Loading model from local directory: {self.model_path}")
                # Load the model from a local directory
                self.model = WhisperModel(
                    self.model_path,
                    device=device,
                    compute_type=compute_type,
                    **cpu_options
                )
            else:
                # Use model size (e.g., "base", "small", "medium", "large-v3")
                logger.info(f"Using model size: {self.model_path}")
                self.model = WhisperModel(
                    self.model_path,
                    device=device,
                    compute_type=compute_type,
                    **cpu_options
                )
            
            self.model_load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {self.model_load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load Faster Whisper model: {e}")
    
    async def transcribe_file_upload(self, file: UploadFile, filename: str) -> Dict[str, Any]:
        """
        Transcribe audio from an uploaded file with detailed performance metrics.
        
        Args:
            file: Audio file from FastAPI UploadFile
            filename: Original filename with extension
            
        Returns:
            Transcription result with performance metrics
        """
        metrics = {
            "model_load": self.model_load_time,
            "validation": 0,
            "audio_preprocessing": 0,
            "model_inference": 0,
            "total": 0
        }
        
        total_start = time.time()
        
        # Validate file format
        validation_start = time.time()
        if not is_valid_audio_format(filename):
            logger.error(f"Invalid audio format: {filename}")
            raise ValueError(f"Invalid audio format. Supported formats: mp3, wav, flac, m4a, ogg, aac, webm")
        metrics["validation"] = time.time() - validation_start
        
        try:
            # Preprocess audio
            preprocessing_start = time.time()
            logger.debug(f"Starting audio preprocessing for {filename}")
            processed_path, processing_time = await preprocess_audio(file, filename)
            metrics["audio_preprocessing"] = processing_time
            
            # Execute transcription in a separate thread to not block the event loop
            logger.debug(f"Starting Faster Whisper inference")
            
            # Run in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            transcription, inference_time = await loop.run_in_executor(
                None,
                lambda: self._transcribe_audio_file(str(processed_path))
            )
            
            metrics["model_inference"] = inference_time
            
            # Calculate total time
            metrics["total"] = time.time() - total_start
            
            # Create a simplified response
            response = {
                "text": transcription,
                "language": "en",
                "total_ms": round(metrics["total"] * 1000),
                "preprocessing_ms": round(metrics["audio_preprocessing"] * 1000),
                "model_inference_ms": round(metrics["model_inference"] * 1000),
            }
            
            return response
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise
    
    def _transcribe_audio_file(self, audio_path: str) -> Tuple[str, float]:
        """
        Transcribe an audio file using the Faster Whisper model.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcription text and inference time
        """
        inference_start = time.time()
        
        try:
            # Run inference
            segments, info = self.model.transcribe(
                audio_path,
                language="en",
                beam_size=5,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Collect all segments
            result = " ".join(segment.text for segment in segments)
            inference_time = time.time() - inference_start
            
            return result.strip(), inference_time
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        built_in_models = [
            "tiny", "tiny.en", 
            "base", "base.en", 
            "small", "small.en", 
            "medium", "medium.en", 
            "large-v1", "large-v2", "large-v3"
        ]
        
        return {
            "current_model": self.model_path,
            "available_models": built_in_models,
            "device": device,
            "compute_type": compute_type
        } 