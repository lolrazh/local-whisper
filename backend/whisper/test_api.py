import os
import sys
import time
import tempfile
import subprocess
import asyncio
import aiohttp
from pathlib import Path
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)

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

async def test_api_endpoint(audio_path, api_url="http://localhost:8001/transcribe"):
    """Test the API endpoint with a given audio file"""
    logger.info(f"Testing API endpoint {api_url} with {audio_path}")
    
    try:
        # Prepare the file for upload
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()
            
            # Make the API request using aiohttp
            start_time = time.time()
            logger.info(f"Sending request to {api_url}")
            
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field('file', 
                               content,
                               filename=os.path.basename(audio_path),
                               content_type="audio/wav")
                data.add_field('temperature', '0.0')
                
                async with session.post(api_url, data=data) as response:
                    request_time = time.time() - start_time
                    
                    # Log the response
                    logger.info(f"Request completed in {request_time:.2f}s")
                    logger.info(f"Response status code: {response.status}")
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Transcription: {result.get('text', '')}")
                        logger.info(f"Performance metrics: {result.get('performance', {})}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed with status code {response.status}")
                        logger.error(f"Response content: {error_text}")
                        return None
    except Exception as e:
        logger.error(f"Error in API request: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

async def main_async():
    """Async version of main test function"""
    logger.info("Starting API endpoint test")
    
    # Create a test audio file
    test_audio_path = create_test_audio()
    if not test_audio_path:
        logger.error("Failed to create test audio file")
        return
    
    # Test the API endpoint
    api_result = await test_api_endpoint(test_audio_path)
    if api_result:
        logger.info("API test completed successfully")
    else:
        logger.error("API test failed")
    
    # Clean up
    try:
        if os.path.exists(test_audio_path):
            os.unlink(test_audio_path)
            parent_dir = os.path.dirname(test_audio_path)
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