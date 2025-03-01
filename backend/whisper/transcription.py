import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any
from loguru import logger
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from scipy.io import wavfile
from fastapi import UploadFile

from audio_utils import preprocess_audio, is_valid_audio_format

# Set torch settings
torch.set_num_threads(4)

# Device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

class TranscriptionService:
    def __init__(self, model_path: str = "models/whisper-base.en"):
        """Initialize the transcription service with the Whisper model."""
        self.model_path = model_path
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._load_model()
        logger.info(f"TranscriptionService initialized with model: {model_path}")
        
    def _load_model(self) -> None:
        """Load the Whisper model and processor."""
        logger.info(f"Loading Whisper model from {self.model_path}")
        
        try:
            # Check if we're using a local path or a HuggingFace model ID
            if Path(self.model_path).exists():
                # Load from local directory
                self.processor = WhisperProcessor.from_pretrained(self.model_path, local_files_only=True)
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_path,
                    local_files_only=True
                ).to(device)
            else:
                # Use standard HuggingFace model ID
                logger.info(f"Local model not found, downloading from HuggingFace: openai/whisper-base.en")
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    "openai/whisper-base.en"
                ).to(device)
                
                # Save model locally
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.processor.save_pretrained(self.model_path)
                self.model.save_pretrained(self.model_path)
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    async def transcribe_file_upload(self, file: UploadFile, filename: str) -> Dict[str, Any]:
        """Transcribe audio from an uploaded file."""
        start_time = time.time()
        
        # Basic validation
        if not is_valid_audio_format(filename):
            raise ValueError(f"Invalid audio format. Supported formats: mp3, wav, flac, m4a, ogg, aac, webm")
        
        # Preprocess audio
        processed_file_path, preprocessing_time = await preprocess_audio(file, filename)
        
        # Transcribe the audio
        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(
            self.executor,
            lambda: self._transcribe_audio_file(str(processed_file_path))
        )
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Clean up the file in the background
        try:
            if processed_file_path.exists():
                os.unlink(processed_file_path)
        except Exception:
            pass
        
        # Return the result
        return {
            "text": transcription,
            "language": "en",
            "total_ms": round(total_time * 1000)
        }
    
    def _transcribe_audio_file(self, audio_path: str) -> str:
        """Transcribe an audio file using Whisper model."""
        try:
            # Load audio
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float32 and normalize
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Process with Whisper
            input_features = self.processor(
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features.to(device)
            
            # Generate tokens
            with torch.no_grad():
                generated_ids = self.model.generate(input_features)
            
            # Decode the tokens to text
            transcription = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        return {
            "current_model": os.path.basename(self.model_path),
            "device": device,
            "available_models": ["whisper-base.en"]
        } 