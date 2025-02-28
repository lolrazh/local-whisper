import os
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from loguru import logger
from groq import Groq
from groq.types.audio import Transcription


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
        start_time = time.time()
        logger.info(f"Starting transcription for file: {file_path}")
        
        try:
            with open(file_path, "rb") as file:
                # Run the transcription in a separate thread to not block the event loop
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
                
            elapsed_time = time.time() - start_time
            logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
            
            return self._format_response(transcription, elapsed_time)
            
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
    ) -> Transcription:
        """Execute the transcription call to Groq API"""
        params = {
            "file": (file_path, file.read()),
            "model": self.model,
            "response_format": response_format,
            "temperature": temperature
        }
        
        if language:
            params["language"] = language
            
        if prompt:
            params["prompt"] = prompt
            
        return self.client.audio.transcriptions.create(**params)
    
    def _format_response(self, transcription: Transcription, elapsed_time: float) -> Dict[str, Any]:
        """Format the transcription response"""
        return {
            "text": transcription.text,
            "processing_time": elapsed_time,
            "model": self.model,
            # Add any additional fields from the response if needed
        } 