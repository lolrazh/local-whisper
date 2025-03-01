import os
import asyncio
import time
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Any, Union, BinaryIO, Tuple, List, Generator, AsyncGenerator
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
CPU_THREADS = min(os.cpu_count() or 4, 4)  # Limit to reasonable number to avoid oversubscription
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
            
            # Transcribe with the model - CPU optimized settings
            inference_start = time.time()
            
            # CPU-optimized transcription parameters (using only documented parameters)
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=3,  # Reduced from 5 for CPU performance
                language="en",  # Force English for now
                vad_filter=True,  # Voice activity detection to filter out silence
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Minimum duration of silence to consider it a break
                    threshold=0.5,  # Default threshold for VAD
                ),
                # CPU optimization options (only documented parameters)
                condition_on_previous_text=True,  # Use previous context to improve transcription
                initial_prompt=None,  # Can be set to a domain-specific prompt if known
                word_timestamps=False,  # Disable for speed (unless needed)
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
        
    async def batch_transcribe_files(self, file_paths: List[str], max_workers: Optional[int] = None) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Transcribe multiple audio files in parallel using CPU resources efficiently.
        
        Args:
            file_paths: List of paths to audio files
            max_workers: Maximum number of parallel workers (defaults to half of CPU cores)
            
        Returns:
            Dictionary mapping each file path to its transcription result
        """
        # Default to a reasonable number of workers based on CPU cores
        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 4) // 2)
            
        logger.info(f"Starting batch transcription of {len(file_paths)} files with {max_workers} workers")
        
        # Filter only valid audio files
        valid_files = []
        for file_path in file_paths:
            if is_valid_audio_format(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(f"Skipping invalid audio format: {file_path}")
        
        if not valid_files:
            return {"error": "No valid audio files provided"}
        
        # Use process pool for true parallelism with CPU-bound tasks
        results = {}
        batch_start_time = time.time()
        
        # Use a process pool for parallel transcription
        # Note: Each process will load its own copy of the model, which increases memory usage
        # but allows for true parallelism on multi-core CPUs
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create a single-process TranscriptionService in each worker
            # to handle the transcription
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path 
                for file_path in valid_files
            }
            
            # Collect results as they complete
            for future in asyncio.as_completed([asyncio.wrap_future(f) for f in future_to_file]):
                file_path = future_to_file[future.get_loop().futures._asyncio_future_blocking._fut]
                try:
                    result = await future
                    results[file_path] = result
                    logger.info(f"Completed transcription of {file_path}")
                except Exception as exc:
                    logger.error(f"Error processing {file_path}: {exc}")
                    results[file_path] = {"error": str(exc)}
        
        total_time = time.time() - batch_start_time
        logger.info(f"Batch transcription completed in {total_time:.2f} seconds")
        
        # Add summary metrics
        results["_batch_summary"] = {
            "total_files": len(valid_files),
            "successful_files": sum(1 for r in results.values() if not isinstance(r, dict) or "error" not in r),
            "total_time": total_time,
            "avg_time_per_file": total_time / len(valid_files) if valid_files else 0
        }
        
        return results
    
    def _process_single_file(self, file_path: str) -> Union[str, Dict[str, Any]]:
        """
        Helper method to process a single file in a separate process.
        This is used by the batch processing method.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Transcription text or error information
        """
        try:
            # Create a new transcription service in this process
            # This is needed because we're running in a separate process
            service = TranscriptionService(self.model_path)
            return service._transcribe_audio_file(file_path)
        except Exception as e:
            logger.error(f"Error in worker process: {e}")
            return {"error": str(e)}
            
    async def stream_transcribe(self, audio_path: str, segment_duration_ms: int = 10000) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream transcription results as they become available by processing audio in small segments.
        This provides incremental results for better user experience with long audio files.
        
        Args:
            audio_path: Path to the audio file
            segment_duration_ms: Duration of each segment in milliseconds
            
        Yields:
            Transcription results for each segment as they become available
        """
        # Import libraries for audio processing inside the function
        # to avoid issues if they're not installed yet
        try:
            import soundfile as sf
            import numpy as np
            
            # Try to import librosa but have fallback if it's not available
            try:
                import librosa
            except ImportError:
                logger.warning("Librosa not available, using basic audio processing instead")
                librosa = None
                
            # Proceed with audio processing
            if librosa:
                # Load audio file with librosa
                logger.info(f"Loading audio file for streaming transcription: {audio_path}")
                audio, sample_rate = librosa.load(audio_path, sr=None)
                duration = librosa.get_duration(y=audio, sr=sample_rate)
                logger.info(f"Audio duration: {duration:.2f} seconds")
            else:
                # Fallback to soundfile if librosa is not available
                logger.info(f"Loading audio file with soundfile: {audio_path}")
                audio, sample_rate = sf.read(audio_path)
                duration = len(audio) / sample_rate
                logger.info(f"Audio duration: {duration:.2f} seconds")
                
            # Calculate segment size in samples
            segment_size = int(sample_rate * segment_duration_ms / 1000)
            total_segments = int(np.ceil(len(audio) / segment_size))
            logger.info(f"Processing audio in {total_segments} segments of {segment_duration_ms}ms each")
            
            # Process each segment
            for i in range(total_segments):
                start_sample = i * segment_size
                end_sample = min((i + 1) * segment_size, len(audio))
                segment_audio = audio[start_sample:end_sample]
                
                # Save segment to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    segment_path = temp_file.name
                    sf.write(segment_path, segment_audio, sample_rate)
                
                try:
                    # Process the segment
                    logger.debug(f"Processing segment {i+1}/{total_segments}")
                    
                    # Run in a separate thread to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    text, metrics = await loop.run_in_executor(
                        None,
                        lambda: self._transcribe_audio_file(segment_path, return_metrics=True)
                    )
                    
                    # Calculate segment timing
                    start_time = i * segment_duration_ms / 1000
                    end_time = min((i + 1) * segment_duration_ms / 1000, duration)
                    
                    # Yield the segment result
                    yield {
                        "segment": i + 1,
                        "total_segments": total_segments,
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text,
                        "is_final": i == total_segments - 1,
                        "metrics": metrics
                    }
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(segment_path):
                        os.unlink(segment_path)
                        
        except Exception as e:
            logger.error(f"Error in streaming transcription: {e}")
            yield {"error": str(e)} 