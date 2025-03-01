import os
import asyncio
import time
import tempfile
import threading
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any, Union, BinaryIO, Tuple
from loguru import logger
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from scipy.io import wavfile
from fastapi import UploadFile

from audio_utils import preprocess_audio, is_valid_audio_format

# Determine optimal CPU settings
CPU_THREADS = min(max(os.cpu_count() or 4 - 1, 1), 6)  # Leave one core free for OS, max 6 for most systems
# Set torch thread settings for better CPU utilization
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(max(CPU_THREADS // 2, 1))  # Interop threads usually need fewer

# Check for GPU availability with better GPU detection
device = "cpu"
if torch.cuda.is_available():
    try:
        # Test if CUDA is actually working
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        device = "cuda"
        logger.info("CUDA is available and working")
    except Exception as e:
        logger.warning(f"CUDA appears available but failed to initialize: {e}")

logger.info(f"Using device: {device} with {CPU_THREADS} CPU threads")

class TranscriptionService:
    def __init__(self, model_path: str = "models/whisper-base.en"):
        """
        Initialize the transcription service with the Vanilla Whisper model.
        
        Args:
            model_path: Path to the Whisper model
        """
        self.model_path = model_path
        self.model_load_time = 0
        self.executor = ThreadPoolExecutor(max_workers=CPU_THREADS)
        self._load_model()
        logger.info(f"TranscriptionService initialized with model: {model_path}")
        
    def _load_model(self) -> None:
        """Load the Whisper model and processor with memory optimizations"""
        logger.info(f"Loading Whisper model from {self.model_path}")
        start_time = time.time()
        
        try:
            # Check if we're using a local path or a HuggingFace model ID
            if Path(self.model_path).exists():
                logger.info(f"Loading model from local directory: {self.model_path}")
                self.processor = WhisperProcessor.from_pretrained(
                    self.model_path,
                    local_files_only=True  # Avoid network checks
                )
                
                # Load the model with memory optimizations
                if device == "cpu":
                    # CPU optimizations: Use lower precision and optimize for CPU
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        self.model_path,
                        local_files_only=True,
                        low_cpu_mem_usage=True,  # Use less CPU memory
                        torch_dtype=torch.float32  # Standard precision for CPU
                    ).to(device)
                else:
                    # GPU optimizations: Use mixed precision
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        self.model_path,
                        local_files_only=True,
                        device_map="auto",  # Automatically map to best device 
                        torch_dtype=torch.float16  # Use FP16 for GPU
                    )
            else:
                # Use standard HuggingFace model ID (e.g., "openai/whisper-base.en")
                logger.info(f"Local model not found, downloading from HuggingFace: openai/whisper-base.en")
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
                
                # Load the model with optimizations
                if device == "cpu":
                    # CPU optimizations
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        "openai/whisper-base.en",
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32
                    ).to(device)
                else:
                    # GPU optimizations
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        "openai/whisper-base.en",
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                
                # Save model to the local path for future use
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                logger.info(f"Saving model to: {self.model_path}")
                self.processor.save_pretrained(self.model_path)
                self.model.save_pretrained(self.model_path)
            
            # Force memory cleanup after loading
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                
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
        
        # Fast validation - move this before any file processing
        validation_start = time.time()
        if not is_valid_audio_format(filename):
            logger.error(f"Invalid audio format: {filename}")
            raise ValueError(f"Invalid audio format. Supported formats: mp3, wav, flac, m4a, ogg, aac, webm")
        metrics["validation"] = time.time() - validation_start
        
        try:
            # Preprocess audio - now optimized in the preprocess_audio function
            preprocessing_start = time.time()
            output_filename, processed_file_path = await preprocess_audio(file, filename)
            metrics["audio_preprocessing"] = time.time() - preprocessing_start
            
            # Execute transcription in our managed thread pool executor
            loop = asyncio.get_event_loop()
            transcription, detailed_metrics = await loop.run_in_executor(
                self.executor,  # Use our dedicated thread pool
                lambda: self._transcribe_audio_file(str(processed_file_path), True)
            )
            
            # Update metrics with detailed inference metrics
            metrics.update(detailed_metrics)
            
            # Calculate total time
            metrics["total"] = time.time() - total_start
            
            # Calculate overhead (time not accounted for in the specific measurements)
            measured_time = sum(v for k, v in metrics.items() if k not in ["total", "model_load", "overhead"])
            metrics["overhead"] = metrics["total"] - measured_time - metrics["model_load"]
            
            # Create a simplified response optimized for API return
            simplified_response = {
                "text": transcription,
                "language": "en",  # Assuming English for now
                "segments": [],  # Simplified without segments
                # Add detailed performance metrics for the UI display
                "total_ms": round(metrics["total"] * 1000),
                "preprocessing_ms": round((metrics["validation"] + metrics["audio_preprocessing"]) * 1000),
                "model_inference_ms": round((metrics["audio_loading"] + metrics["feature_extraction"] + 
                                            metrics["model_inference"] + metrics["decoding"]) * 1000),
                "overhead_ms": round(metrics["overhead"] * 1000)
            }
            
            # Log performance summary briefly
            logger.info(f"Transcription complete: {len(transcription)} chars in {metrics['total']*1000:.2f}ms")
            
            # Detailed metrics are logged at debug level
            logger.debug(f"Preprocessing: {simplified_response['preprocessing_ms']}ms, " +
                        f"Inference: {simplified_response['model_inference_ms']}ms, " +
                        f"Overhead: {simplified_response['overhead_ms']}ms")
            
            if return_metrics:
                return simplified_response, metrics
            return simplified_response
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise
        finally:
            # Clean up temporary files - use a background task for this
            cleanup_start = time.time()
            try:
                if 'processed_file_path' in locals() and processed_file_path.exists():
                    os.unlink(processed_file_path)
                    parent_dir = processed_file_path.parent
                    if parent_dir.exists() and not list(parent_dir.iterdir()):
                        os.rmdir(parent_dir)
            except Exception as e:
                logger.debug(f"Error cleaning up temporary files: {e}")
            finally:
                metrics["cleanup"] = time.time() - cleanup_start
    
    def _transcribe_audio_file(self, audio_path: str, return_metrics: bool = False) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        Execute transcription on audio file using Whisper model with detailed timing.
        Optimized for CPU performance.
        
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
            # Load audio with optimized settings
            audio_loading_start = time.time()
            try:
                # Try memory-mapped loading for better performance with large files
                sample_rate, audio_data = wavfile.read(audio_path, mmap=True)
            except:
                # Fall back to regular loading
                sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float32 and normalize with optimized numpy operations
            if audio_data.dtype == np.int16:
                # Use .astype(np.float32, copy=False) where possible to avoid extra copy
                # The float32 division is faster than float64
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            metrics["audio_loading"] = time.time() - audio_loading_start
            
            # Process audio with the Whisper processor - optimized for memory usage
            feature_extraction_start = time.time()
            
            # Move audio data to correct device only once
            audio_tensor = torch.from_numpy(audio_data)
            
            # Use a more memory-efficient approach for feature extraction
            input_features = self.processor(
                audio_tensor, 
                sampling_rate=sample_rate, 
                return_tensors="pt",
                padding=False  # No padding needed for single sample
            ).input_features
            
            # Move to device only after processing to reduce memory copies
            input_features = input_features.to(device)
            metrics["feature_extraction"] = time.time() - feature_extraction_start
            
            # Generate token ids with optimized settings
            inference_start = time.time()
            with torch.no_grad():  # Ensure no gradients are calculated
                # Set more aggressive beam search parameters for speed
                predicted_ids = self.model.generate(
                    input_features,
                    num_beams=3,  # Reduced from default 5 for speed
                    do_sample=False,  # Deterministic for speed
                    max_new_tokens=256,  # Limit for shorter audio
                    temperature=1.0,  # No temperature (deterministic)
                    use_cache=True,  # Use KV cache for faster generation
                )
            metrics["model_inference"] = time.time() - inference_start
            
            # Decode the output
            decoding_start = time.time()
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0].strip()
            metrics["decoding"] = time.time() - decoding_start
            
            # Simple post-processing
            postprocessing_start = time.time()
            # Remove extra whitespace, normalize to single type of quotes, etc.
            transcription = " ".join(transcription.split())
            metrics["postprocessing"] = time.time() - postprocessing_start
            
            # Clean up memory
            del input_features, predicted_ids, audio_tensor
            if device == "cuda":
                torch.cuda.empty_cache()
            
            if return_metrics:
                return transcription, metrics
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}")
    
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