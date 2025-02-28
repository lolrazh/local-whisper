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
            _, audio_path = await preprocess_audio(file, filename)
            metrics["audio_preprocessing"] = time.time() - preprocessing_start
            logger.debug(f"Audio preprocessing completed in {metrics['audio_preprocessing']*1000:.2f}ms")
            
            # Execute transcription in a separate thread to not block the event loop
            logger.debug(f"Starting Faster Whisper inference")
            
            # Run in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            transcription, detailed_metrics = await loop.run_in_executor(
                None,
                lambda: self._transcribe_audio_file(str(audio_path), True)
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
            logger.info(f"FASTER WHISPER TRANSCRIPTION METRICS - {filename}")
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
                "Feature Extraction": metrics.get("feature_extraction", 0),
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
                "language": "en",  # Assuming English for now
                "segments": [],  # We'll add segments in a more complex implementation
                "total_ms": round(metrics["total"] * 1000),
                "preprocessing_ms": round(preprocessing_time * 1000),
                "model_inference_ms": round(model_time * 1000),
                "overhead_ms": round((metrics["overhead"] + metrics["cleanup"]) * 1000),
                "metrics": {
                    "model_load_ms": round(metrics["model_load"] * 1000),
                    "preprocessing_ms": round(preprocessing_time * 1000),
                    "inference_ms": round(model_time * 1000),
                    "total_ms": round(metrics["total"] * 1000)
                }
            }
            
            if return_metrics:
                return simplified_response, metrics
            return simplified_response
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
        finally:
            # Clean up temporary files
            cleanup_start = time.time()
            try:
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.unlink(audio_path)
                    parent_dir = os.path.dirname(audio_path)
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")
            finally:
                metrics["cleanup"] = time.time() - cleanup_start
    
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
            # Audio loading timing is included in the model_inference step for Faster Whisper
            # since the library handles it internally
            audio_loading_start = time.time()
            metrics["audio_loading"] = time.time() - audio_loading_start
            
            # Transcribe with the model
            inference_start = time.time()
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                language="en",  # Force English for now
                vad_filter=True,  # Voice activity detection to filter out silence
                vad_parameters=dict(min_silence_duration_ms=500)  # Minimum duration of silence to consider it a break
            )
            
            # Decoding timing
            decoding_start = time.time()
            # Force execution of the generator
            segments_list = list(segments)
            metrics["decoding"] = time.time() - decoding_start
            
            # Process segments to create final text
            postprocessing_start = time.time()
            full_text = " ".join([segment.text.strip() for segment in segments_list])
            metrics["postprocessing"] = time.time() - postprocessing_start
            
            # Calculate total inference time (including internal audio loading)
            metrics["model_inference"] = decoding_start - inference_start
            
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