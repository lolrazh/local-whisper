import os
import sys
import time
import asyncio
from pathlib import Path
import tempfile
import subprocess
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)

# Import the transcription service
from transcription import TranscriptionService
from audio_utils import preprocess_audio, is_valid_audio_format

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

async def test_audio_preprocessing(audio_path):
    """Test the audio preprocessing function"""
    logger.info(f"Testing audio preprocessing with {audio_path}")
    
    try:
        # Open the audio file
        with open(audio_path, "rb") as audio_file:
            # Call the preprocess_audio function
            start_time = time.time()
            output_filename, processed_file_path = await preprocess_audio(audio_file, os.path.basename(audio_path))
            processing_time = time.time() - start_time
            
            logger.info(f"Audio preprocessing successful in {processing_time:.2f}s")
            logger.info(f"Processed file: {processed_file_path}")
            
            return processed_file_path
    except Exception as e:
        logger.error(f"Error in audio preprocessing: {e}")
        return None

def test_transcription(audio_path, model_path="models/whisper-base.en"):
    """Test the transcription service with a given audio file"""
    logger.info(f"Testing transcription with {audio_path}")
    logger.info(f"Using model path: {model_path}")
    
    try:
        # Initialize the transcription service
        start_time = time.time()
        service = TranscriptionService(model_path=model_path)
        init_time = time.time() - start_time
        logger.info(f"TranscriptionService initialized in {init_time:.2f}s")
        
        # Open the audio file
        with open(audio_path, "rb") as audio_file:
            # Call the transcribe_file_upload method
            start_time = time.time()
            result = service._transcribe_audio_file(audio_path)
            transcription_time = time.time() - start_time
            
            logger.info(f"Transcription successful in {transcription_time:.2f}s")
            logger.info(f"Transcription result: {result}")
            
            return result
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

async def main_async():
    """Async version of main test function"""
    logger.info("Starting transcription test")
    
    # Get the absolute path to the model directory
    BASE_DIR = Path(__file__).parent.parent.parent
    MODEL_PATH = str(BASE_DIR / "models" / "whisper-base.en")
    logger.info(f"Using model path: {MODEL_PATH}")
    
    # Create a test audio file
    test_audio_path = create_test_audio()
    if not test_audio_path:
        logger.error("Failed to create test audio file")
        return
    
    # Test audio preprocessing
    processed_audio_path = await test_audio_preprocessing(test_audio_path)
    if not processed_audio_path:
        logger.error("Failed to preprocess audio")
        return
    
    # Test transcription
    transcription_result = test_transcription(processed_audio_path, MODEL_PATH)
    if transcription_result:
        logger.info("Test completed successfully")
    else:
        logger.error("Transcription test failed")
    
    # Clean up
    try:
        if os.path.exists(test_audio_path):
            os.unlink(test_audio_path)
            parent_dir = os.path.dirname(test_audio_path)
            if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
        
        if os.path.exists(processed_audio_path):
            os.unlink(processed_audio_path)
            parent_dir = os.path.dirname(processed_audio_path)
            if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {e}")

def main():
    """Main test function"""
    # Run the async function with asyncio
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 