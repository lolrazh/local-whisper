import tempfile
import os
import subprocess
import asyncio
from pathlib import Path

async def test_ffmpeg():
    # Create a temp directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp dir: {temp_dir}")
    
    # Generate a test audio file
    test_wav = os.path.join(temp_dir, "test.wav")
    
    # Use FFmpeg to generate a sine wave audio file
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", test_wav
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the FFmpeg process
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    print(f"Exit code: {process.returncode}")
    if process.returncode != 0:
        print(f"Error: {stderr.decode('utf-8', errors='replace')}")
    else:
        print(f"Output file exists: {os.path.exists(test_wav)}")
        print(f"Output file size: {os.path.getsize(test_wav) if os.path.exists(test_wav) else 0}")
    
    # Now try to convert webm to wav
    test_webm = os.path.join(temp_dir, "recording.webm")
    # Create an empty webm file
    with open(test_webm, "wb") as f:
        f.write(b"WEBM" + b"\x00" * 100)  # Just a fake header
    
    print(f"Created fake webm file: {test_webm}")
    
    # Try to convert it
    output_wav = os.path.join(temp_dir, "processed_recording.wav")
    cmd = [
        "ffmpeg", "-y", "-i", test_webm,
        "-ar", "16000", "-ac", "1", 
        "-vn", "-sample_fmt", "s16",
        "-threads", str(os.cpu_count() or 1),
        output_wav
    ]
    
    print(f"Running conversion command: {' '.join(cmd)}")
    
    # Run the FFmpeg process
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    print(f"Conversion exit code: {process.returncode}")
    print(f"Conversion error: {stderr.decode('utf-8', errors='replace')}")

if __name__ == "__main__":
    asyncio.run(test_ffmpeg()) 