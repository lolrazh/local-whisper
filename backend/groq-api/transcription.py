import os
import time
from pathlib import Path
from typing import Dict, Any, BinaryIO
from loguru import logger
from groq import Groq
from dotenv import load_dotenv

from audio_utils import preprocess_audio, is_valid_audio_format

# Load environment variables from .env file
load_dotenv()

class TranscriptionService:
    def __init__(self, api_key: str, model: str = "distil-whisper-large-v3-en"):
        """
        Initialize the transcription service with the Groq API.
        
        Args:
            api_key: Groq API key
            model: Model to use for transcription
        """
        self.api_key = api_key
        self.model = model
        self._validate_credentials()
        self.client = Groq(api_key=self.api_key)
        logger.info(f"TranscriptionService initialized with model: {model}")
        
    def _validate_credentials(self) -> None:
        """Validate that the API key is provided"""
        if not self.api_key:
            raise ValueError("Groq API key is required")
    
    def transcribe_file_upload(self, file: BinaryIO, filename: str) -> Dict[str, Any]:
        """
        Transcribe audio from an uploaded file with detailed performance metrics.
        
        Args:
            file: Audio file binary data
            filename: Original filename with extension
            
        Returns:
            Transcription result with performance metrics
        """
        metrics = {
            "validation_ms": 0,
            "preprocessing_ms": 0,
            "api_call_ms": 0,
            "total_ms": 0
        }
        
        total_start = time.time()
        
        # Validate file format
        validation_start = time.time()
        if not is_valid_audio_format(filename):
            logger.error(f"Invalid audio format: {filename}")
            raise ValueError(f"Invalid audio format. Supported formats: mp3, wav, flac, m4a, ogg, aac, webm")
        metrics["validation_ms"] = round((time.time() - validation_start) * 1000)
        
        try:
            # Preprocess audio
            preprocessing_start = time.time()
            logger.info(f"Starting audio preprocessing for {filename}")
            processed_file_path, processing_time = preprocess_audio(file, filename)
            metrics["preprocessing_ms"] = round(processing_time * 1000)
            
            # Execute transcription
            api_start = time.time()
            logger.info(f"Starting API call to Groq for transcription")
            
            with open(processed_file_path, "rb") as audio_file:
                params = {
                    "model": self.model,
                    "file": audio_file,
                    "response_format": "json",
                    "temperature": 0.0
                }
                
                transcription = self.client.audio.transcriptions.create(**params)
            
            metrics["api_call_ms"] = round((time.time() - api_start) * 1000)
            
            # Calculate total time
            metrics["total_ms"] = round((time.time() - total_start) * 1000)
            
            # Format response
            response = {
                "text": transcription.text,
                "language": "en",
                "total_ms": metrics["total_ms"],
                "preprocessing_ms": metrics["preprocessing_ms"],
                "model_inference_ms": metrics["api_call_ms"]
            }
            
            # Clean up temporary files
            try:
                if os.path.exists(processed_file_path):
                    os.unlink(processed_file_path)
                    parent_dir = os.path.dirname(processed_file_path)
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get available transcription models"""
        models = [
            "distil-whisper-large-v3-en",
            "whisper-large-v3"
        ]
        
        return {
            "current_model": self.model,
            "available_models": models
        } 