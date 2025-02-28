import os
import sys
import time
import asyncio
import tempfile
import subprocess
from pathlib import Path
from loguru import logger
import io

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)

# Import the audio utils and transcription service
from audio_utils import preprocess_audio
from transcription import TranscriptionService

# Create a mock class that mimics the UploadFile interface for testing
class MockUploadFile:
    def __init__(self, filepath, filename):
        self.filepath = filepath
        self.filename = filename
        self._content = None
        
        # Read the file content once during initialization
        with open(filepath, "rb") as f:
            self._content = f.read()
    
    async def read(self):
        """Return the file content"""
        return self._content
    
    def close(self):
        """Cleanup method"""
        self._content = None

def create_test_audio():
    """Create a simple test audio file using ffmpeg"""
    temp_dir = tempfile.mkdtemp()
    test_audio_path = os.path.join(temp_dir, "test_audio.wav")
    
    try:
        # Generate a 3-second sine wave audio file
        cmd = [
            "ffmpeg",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=3",
            "-ar", "16000",
            "-ac", "1",
            test_audio_path,
            "-y",
            "-loglevel", "error"
        ]
        
        logger.info(f"Creating test audio file at {test_audio_path}")
        subprocess.run(cmd, check=True)
        logger.info(f"Test audio file created successfully")
        
        return test_audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating test audio: {e}")
        return None

async def test_preprocess(audio_path):
    """Test the audio preprocessing function directly"""
    logger.info(f"Testing audio preprocessing with {audio_path}")
    
    try:
        # Create a mock UploadFile
        mock_file = MockUploadFile(audio_path, os.path.basename(audio_path))
        
        # Call the preprocess_audio function
        start_time = time.time()
        output_filename, processed_path = await preprocess_audio(mock_file, mock_file.filename)
        processing_time = time.time() - start_time
        
        logger.info(f"Preprocessing successful in {processing_time:.2f}s")
        logger.info(f"Processed file: {processed_path}")
        
        # Close the mock file
        mock_file.close()
        
        return processed_path
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

async def test_transcription_service(audio_path):
    """Test the transcription service directly"""
    logger.info(f"Testing transcription service with {audio_path}")
    
    try:
        # Initialize the transcription service
        BASE_DIR = Path(__file__).parent.parent.parent
        MODEL_PATH = str(BASE_DIR / "models" / "whisper-base.en")
        
        model_load_start = time.time()
        service = TranscriptionService(model_path=MODEL_PATH)
        model_load_time = time.time() - model_load_start
        logger.info(f"Service initialization time: {model_load_time*1000:.2f}ms")
        
        # Create a mock UploadFile
        mock_file = MockUploadFile(audio_path, os.path.basename(audio_path))
        
        # Call the transcribe_file_upload method with return_metrics=True
        start_time = time.time()
        result, metrics = await service.transcribe_file_upload(
            mock_file, 
            mock_file.filename,
            return_metrics=True
        )
        transcription_time = time.time() - start_time
        
        logger.info(f"Transcription successful in {transcription_time:.2f}s")
        logger.info(f"Transcription result: {result['text']}")
        
        # Display detailed metrics summary
        logger.info("\n" + "-"*50)
        logger.info("PERFORMANCE METRICS SUMMARY:")
        logger.info("-"*50)
        logger.info(f"Total time: {metrics['total']*1000:.2f}ms")
        
        # Display metrics in a structured way, grouped by category
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
        
        other_metrics = {
            "Cleanup": metrics["cleanup"],
            "Overhead": metrics["overhead"]
        }
        
        # Display preprocessing metrics
        logger.info("\nPreprocessing Steps:")
        for name, value in preprocessing_metrics.items():
            percent = (value / metrics["total"]) * 100
            logger.info(f"  {name}: {value*1000:.2f}ms ({percent:.1f}%)")
        
        # Display model metrics
        logger.info("\nModel Inference Steps:")
        for name, value in model_metrics.items():
            percent = (value / metrics["total"]) * 100
            logger.info(f"  {name}: {value*1000:.2f}ms ({percent:.1f}%)")
        
        # Display other metrics
        logger.info("\nOther:")
        for name, value in other_metrics.items():
            percent = (value / metrics["total"]) * 100
            logger.info(f"  {name}: {value*1000:.2f}ms ({percent:.1f}%)")
        
        # Calculate and display percentages of total time
        logger.info("\nTime Distribution:")
        preprocessing_time = sum(preprocessing_metrics.values())
        model_time = sum(model_metrics.values())
        other_time = sum(other_metrics.values())
        
        preprocessing_percent = (preprocessing_time / metrics["total"]) * 100
        model_percent = (model_time / metrics["total"]) * 100
        other_percent = (other_time / metrics["total"]) * 100
        
        logger.info(f"  Preprocessing: {preprocessing_time*1000:.2f}ms ({preprocessing_percent:.1f}%)")
        logger.info(f"  Model Inference: {model_time*1000:.2f}ms ({model_percent:.1f}%)")
        logger.info(f"  Other: {other_time*1000:.2f}ms ({other_percent:.1f}%)")
        logger.info("-"*50)
        
        # Close the mock file
        mock_file.close()
        
        return result
    except Exception as e:
        logger.error(f"Error in transcription service: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

async def main_async():
    """Main async test function"""
    logger.info("Starting debug test")
    
    # Create a test audio file
    test_audio_path = create_test_audio()
    if not test_audio_path:
        logger.error("Failed to create test audio file")
        return
    
    try:
        # Test preprocessing directly
        processed_path = await test_preprocess(test_audio_path)
        if not processed_path:
            logger.error("Preprocessing test failed")
            return
        
        # Test transcription service directly
        result = await test_transcription_service(test_audio_path)
        if result:
            logger.info("Debug test completed successfully")
        else:
            logger.error("Transcription service test failed")
    finally:
        # Clean up
        try:
            if 'test_audio_path' in locals() and os.path.exists(test_audio_path):
                os.unlink(test_audio_path)
                parent_dir = os.path.dirname(test_audio_path)
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
            
            if 'processed_path' in locals() and os.path.exists(processed_path):
                os.unlink(processed_path)
                parent_dir = os.path.dirname(processed_path)
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")

def main():
    """Main test function"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 