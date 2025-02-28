import os
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, BinaryIO
from loguru import logger
from groq import AsyncGroq, Groq
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
        self.async_client = AsyncGroq(api_key=self.api_key)
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
        logger.info(f"File validation completed in {metrics['validation_ms']}ms")
        
        try:
            # Preprocess audio
            preprocessing_start = time.time()
            logger.info(f"Starting audio preprocessing for {filename}")
            output_filename, processed_file_path = preprocess_audio(file, filename)
            metrics["preprocessing_ms"] = round((time.time() - preprocessing_start) * 1000)
            logger.info(f"Audio preprocessing completed in {metrics['preprocessing_ms']}ms")
            
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
            logger.info(f"API call completed in {metrics['api_call_ms']}ms")
            
            # Calculate total time
            metrics["total_ms"] = round((time.time() - total_start) * 1000)
            
            # Format response
            response = self._format_response(transcription, metrics["total_ms"] / 1000)
            
            # Add detailed performance metrics
            response["performance"] = metrics
            
            # Log performance summary
            logger.info(f"Transcription completed in {metrics['total_ms']}ms - " +
                       f"Validation: {metrics['validation_ms']}ms, " +
                       f"Preprocessing: {metrics['preprocessing_ms']}ms, " +
                       f"API call: {metrics['api_call_ms']}ms")
            
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
        
    async def transcribe(
        self, 
        file_path: Path, 
        language: Optional[str] = "en",
        temperature: float = 0.0,
        prompt: Optional[str] = None,
        response_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file using the Groq API.
        
        Args:
            file_path: Path to the audio file
            language: Language code (optional)
            temperature: Temperature for generation (0-1)
            prompt: Context or specific vocabulary to help with transcription
            response_format: Format of the response
            
        Returns:
            Dictionary containing transcription results
        """
        metrics = {
            "validation_ms": 0,
            "api_call_ms": 0,
            "total_ms": 0
        }
        
        start_time = time.time()
        logger.info(f"Starting transcription for file: {file_path}")
        
        try:
            with open(file_path, "rb") as file:
                # Run the transcription in a separate thread to not block the event loop
                api_start_time = time.time()
                loop = asyncio.get_event_loop()
                transcription = await loop.run_in_executor(
                    None,
                    lambda: self._execute_transcription(
                        file, 
                        str(file_path), 
                        language, 
                        temperature, 
                        prompt, 
                        response_format
                    )
                )
                metrics["api_call_ms"] = round((time.time() - api_start_time) * 1000)
                
            metrics["total_ms"] = round((time.time() - start_time) * 1000)
            logger.info(f"Transcription completed in {metrics['total_ms']}ms (API call: {metrics['api_call_ms']}ms)")
            
            result = self._format_response(transcription, metrics["total_ms"] / 1000)
            result["performance"] = metrics
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    parent_dir = os.path.dirname(file_path)
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")
    
    def _execute_transcription(
        self, 
        file, 
        file_path: str,
        language: Optional[str],
        temperature: float,
        prompt: Optional[str],
        response_format: str
    ) -> Dict[str, Any]:
        """Execute the transcription call to Groq API"""
        try:
            with open(file_path, "rb") as audio_file:
                params = {
                    "model": self.model,
                    "file": audio_file,
                    "response_format": response_format,
                    "temperature": temperature
                }
                
                if language:
                    params["language"] = language
                    
                if prompt:
                    params["prompt"] = prompt
                    
                return self.client.audio.transcriptions.create(**params)
        except Exception as e:
            logger.error(f"Error in transcription execution: {str(e)}")
            raise
    
    def _format_response(self, transcription: Any, elapsed_time: float) -> Dict[str, Any]:
        """Format the transcription response"""
        if hasattr(transcription, "text"):
            text = transcription.text
        elif isinstance(transcription, dict):
            text = transcription.get("text", "")
        else:
            text = str(transcription)
            
        return {
            "text": text,
            "processing_time": elapsed_time,
            "model": self.model,
        }

    def get_available_models(self) -> Dict[str, Any]:
        """
        Get a list of available transcription models.
        
        Returns:
            List of available models with details
        """
        # In a real implementation, this would query the API for available models
        # For now, we'll return the default models we support
        return {
            "models": [
                {
                    "id": "distil-whisper-large-v3-en",
                    "name": "Distil Whisper Large v3 (English)",
                    "description": "Optimized for English transcription"
                },
                {
                    "id": "whisper-large-v3-turbo",
                    "name": "Whisper Large v3 Turbo",
                    "description": "Fast multi-language transcription"
                }
            ]
        } 