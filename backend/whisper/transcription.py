import os
import asyncio
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union, BinaryIO
from loguru import logger
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from scipy.io import wavfile
from fastapi import UploadFile

from audio_utils import preprocess_audio, is_valid_audio_format

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

class TranscriptionService:
    def __init__(self, model_path: str = "models/whisper-base.en"):
        """
        Initialize the transcription service with the Vanilla Whisper model.
        
        Args:
            model_path: Path to the Whisper model
        """
        self.model_path = model_path
        self._load_model()
        logger.info(f"TranscriptionService initialized with model: {model_path}")
        
    def _load_model(self) -> None:
        """Load the Whisper model and processor"""
        logger.info(f"Loading Whisper model from {self.model_path}")
        start_time = time.time()
        
        try:
            # Check if we're using a local path or a HuggingFace model ID
            if Path(self.model_path).exists():
                logger.info(f"Loading model from local directory: {self.model_path}")
                self.processor = WhisperProcessor.from_pretrained(self.model_path)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path).to(device)
            else:
                # Use standard HuggingFace model ID (e.g., "openai/whisper-base.en")
                logger.info(f"Local model not found, downloading from HuggingFace: openai/whisper-base.en")
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
                self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").to(device)
                
                # Save model to the local path for future use
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                logger.info(f"Saving model to: {self.model_path}")
                self.processor.save_pretrained(self.model_path)
                self.model.save_pretrained(self.model_path)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
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
            "validation_ms": 0,
            "preprocessing_ms": 0,
            "model_inference_ms": 0,
            "total_ms": 0
        }
        
        total_start = time.time()
        
        # Validate file format
        validation_start = time.time()
        if not is_valid_audio_format(filename):
            logger.error(f"Invalid audio format: {filename}")
            raise ValueError(f"Invalid audio format. Supported formats: mp3, wav, flac, m4a, ogg, aac, webm")
        metrics["validation_ms"] = round((time.time() - validation_start) * 1000)
        logger.info(f"File validation completed in {metrics['validation_ms']}ms")
        
        try:
            # Preprocess audio
            preprocessing_start = time.time()
            logger.info(f"Starting audio preprocessing for {filename}")
            output_filename, processed_file_path = await preprocess_audio(file, filename)
            metrics["preprocessing_ms"] = round((time.time() - preprocessing_start) * 1000)
            logger.info(f"Audio preprocessing completed in {metrics['preprocessing_ms']}ms")
            
            # Execute transcription in a separate thread to not block the event loop
            model_start = time.time()
            logger.info(f"Starting Whisper inference")
            
            # Run in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(
                None,
                lambda: self._transcribe_audio_file(str(processed_file_path))
            )
            
            metrics["model_inference_ms"] = round((time.time() - model_start) * 1000)
            logger.info(f"Whisper inference completed in {metrics['model_inference_ms']}ms")
            
            # Calculate total time
            metrics["total_ms"] = round((time.time() - total_start) * 1000)
            
            # Format response
            response = {
                "text": transcription,
                "duration": metrics["total_ms"] / 1000
            }
            
            # Add detailed performance metrics
            response["performance"] = metrics
            
            # Log performance summary
            logger.info(f"Transcription completed in {metrics['total_ms']}ms - " +
                       f"Validation: {metrics['validation_ms']}ms, " +
                       f"Preprocessing: {metrics['preprocessing_ms']}ms, " +
                       f"Model inference: {metrics['model_inference_ms']}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            try:
                if 'processed_file_path' in locals() and os.path.exists(processed_file_path):
                    os.unlink(processed_file_path)
                    parent_dir = os.path.dirname(processed_file_path)
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")
    
    def _transcribe_audio_file(self, audio_path: str) -> str:
        """
        Execute transcription on audio file using Whisper model.
        
        Args:
            audio_path: Path to the audio file to transcribe
            
        Returns:
            Transcription text
        """
        try:
            # Load audio
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float32 and normalize if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Process audio with the Whisper processor
            input_features = self.processor(
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features.to(device)
            
            # Generate token ids
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Decode token ids to text
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error during Whisper transcription: {e}")
            raise RuntimeError(f"Whisper transcription failed: {e}")
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Get a list of available transcription models.
        
        Returns:
            List of available models with details
        """
        return {
            "models": [
                {
                    "id": "whisper-base.en",
                    "name": "Whisper Base (English)",
                    "description": "Vanilla Whisper model for English transcription"
                }
            ]
        } 