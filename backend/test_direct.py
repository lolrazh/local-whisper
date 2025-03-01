import asyncio
import subprocess
import os
import tempfile
from pathlib import Path

async def test_preprocessing():
    """Test direct audio preprocessing with FFmpeg."""
    # Create a temp directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp dir: {temp_dir}")
    
    # Create a simple test.wav file using ffmpeg
    test_wav = os.path.join(temp_dir, "test.wav")
    
    # Generate 1 second of audio
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", 
        "-i", "sine=frequency=1000:duration=1", 
        "-ar", "16000", "-ac", "1", 
        test_wav
    ]
    
    print(f"Creating test file with command: {' '.join(cmd)}")
    
    # Run with simple subprocess (not asyncio)
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error creating test file: {result.stderr}")
        return
    
    print(f"Created test file: {test_wav}")
    
    # Now try to convert it to standard format
    output_wav = os.path.join(temp_dir, "processed_test.wav")
    
    cmd = [
        "ffmpeg", "-y", "-i", test_wav,
        "-ar", "16000", "-ac", "1", 
        "-c:a", "pcm_s16le",
        output_wav
    ]
    
    print(f"Converting with command: {' '.join(cmd)}")
    
    # Run with simple subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error converting file: {result.stderr}")
    else:
        print(f"Conversion successful!")
        print(f"Output file exists: {os.path.exists(output_wav)}")
        print(f"Output file size: {os.path.getsize(output_wav) / 1024:.2f} KB")

if __name__ == "__main__":
    asyncio.run(test_preprocessing()) 