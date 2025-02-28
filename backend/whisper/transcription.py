import os
import asyncio
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union, BinaryIO, Tuple
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
        self.model_load_time = 0
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
            
            self.model_load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {self.model_load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
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
            "feature_extraction": 0,
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
            logger.debug(f"Starting audio preprocessing for {filename}")
            output_filename, processed_file_path = await preprocess_audio(file, filename)
            metrics["audio_preprocessing"] = time.time() - preprocessing_start
            logger.debug(f"Audio preprocessing completed in {metrics['audio_preprocessing']*1000:.2f}ms")
            
            # Execute transcription in a separate thread to not block the event loop
            logger.debug(f"Starting Whisper inference")
            
            # Run in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            transcription, detailed_metrics = await loop.run_in_executor(
                None,
                lambda: self._transcribe_audio_file(str(processed_file_path), True)
            )
            
            # Update metrics with detailed inference metrics
            metrics.update(detailed_metrics)
            
            # Calculate total time
            metrics["total"] = time.time() - total_start
            
            # Calculate overhead (time not accounted for in the specific measurements)
            measured_time = sum(metrics.values()) - metrics["total"] - metrics["model_load"]
            metrics["overhead"] = metrics["total"] - measured_time
            
            # Log performance summary in a more visible format
            logger.info("\n" + "="*50)
            logger.info(f"WHISPER TRANSCRIPTION METRICS - {filename}")
            logger.info("="*50)
            logger.info(f"Transcription result: \"{transcription}\"")
            logger.info("Transcription completed successfully")
            
            # Detailed metrics are logged at debug level only
            logger.debug(f"Total time: {metrics['total']*1000:.2f}ms")
            
            # Group metrics by category for better readability
            preprocessing_metrics = {
                "Validation": metrics["validation"],
                "Audio Preprocessing": metrics["audio_preprocessing"],
                "Audio Loading": metrics["audio_loading"]
            }
            
            model_metrics = {
                "Feature Extraction": metrics["feature_extraction"],
                "Model Inference": metrics["model_inference"],
                "Decoding": metrics["decoding"],
                "Postprocessing": metrics["postprocessing"]
            }
            
            # Display preprocessing metrics at debug level
            logger.debug("\nPreprocessing Steps:")
            for name, value in preprocessing_metrics.items():
                percent = (value / metrics["total"]) * 100
                logger.debug(f"  {name}: {value*1000:.2f}ms ({percent:.1f}%)")
            
            # Display model metrics at debug level
            logger.debug("\nModel Inference Steps:")
            for name, value in model_metrics.items():
                percent = (value / metrics["total"]) * 100
                logger.debug(f"  {name}: {value*1000:.2f}ms ({percent:.1f}%)")
            
            # Calculate and display percentages of total time
            preprocessing_time = sum(preprocessing_metrics.values())
            model_time = sum(model_metrics.values())
            
            preprocessing_percent = (preprocessing_time / metrics["total"]) * 100
            model_percent = (model_time / metrics["total"]) * 100
            
            logger.debug("\nTime Distribution:")
            logger.debug(f"  Preprocessing: {preprocessing_time*1000:.2f}ms ({preprocessing_percent:.1f}%)")
            logger.debug(f"  Model Inference: {model_time*1000:.2f}ms ({model_percent:.1f}%)")
            logger.debug(f"  Overhead & Cleanup: {(metrics['overhead'] + metrics['cleanup'])*1000:.2f}ms ({(metrics['overhead'] + metrics['cleanup'])/metrics['total']*100:.1f}%)")
            logger.debug("="*50)
            
            # Create a simplified response that only includes the transcription text and total duration
            # This prevents the detailed metrics from appearing in the transcription box
            simplified_response = {
                "text": transcription,
                "duration": round(metrics["total"], 3),
                # Add detailed performance metrics for the UI display
                "total_ms": round(metrics["total"] * 1000),
                "preprocessing_ms": round(preprocessing_time * 1000),
                "model_inference_ms": round(model_time * 1000),
                "overhead_ms": round((metrics["overhead"] + metrics["cleanup"]) * 1000)
            }
            
            if return_metrics:
                return simplified_response, metrics
            return simplified_response
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            cleanup_start = time.time()
            try:
                if 'processed_file_path' in locals() and os.path.exists(processed_file_path):
                    os.unlink(processed_file_path)
                    parent_dir = os.path.dirname(processed_file_path)
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")
            finally:
                metrics["cleanup"] = time.time() - cleanup_start
    
    def _transcribe_audio_file(self, audio_path: str, return_metrics: bool = False) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        Execute transcription on audio file using Whisper model with detailed timing.
        
        Args:
            audio_path: Path to the audio file to transcribe
            return_metrics: Whether to return detailed metrics
            
        Returns:
            Transcription text and optionally metrics dictionary
        """
        metrics = {
            "audio_loading": 0,
            "feature_extraction": 0,
            "model_inference": 0,
            "decoding": 0,
            "postprocessing": 0
        }
        
        try:
            # Load audio
            audio_loading_start = time.time()
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float32 and normalize if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            metrics["audio_loading"] = time.time() - audio_loading_start
            
            # Process audio with the Whisper processor
            feature_extraction_start = time.time()
            input_features = self.processor(
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features.to(device)
            metrics["feature_extraction"] = time.time() - feature_extraction_start
            
            # Generate token ids
            inference_start = time.time()
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            metrics["model_inference"] = time.time() - inference_start
            
            # Decode token ids to text
            decoding_start = time.time()
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            metrics["decoding"] = time.time() - decoding_start
            
            # Post-processing
            postprocessing_start = time.time()
            transcription = transcription.strip()
            metrics["postprocessing"] = time.time() - postprocessing_start
            
            if return_metrics:
                return transcription, metrics
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