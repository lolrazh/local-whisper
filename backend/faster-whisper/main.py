import os
import sys
import shutil
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn
import signal
import threading
from pydantic import BaseModel, Field
import asyncio
import tempfile
import json

from transcription import TranscriptionService
from audio_utils import preprocess_audio, is_valid_audio_format, get_file_size_mb

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)
logger.add("logs/faster_whisper_api.log", rotation="10 MB", retention="3 days", level="DEBUG")

# Create FastAPI app
app = FastAPI(
    title="Faster Whisper API",
    description="API for transcribing audio using Faster Whisper",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")

# Path to local CTranslate2 model
# Try the specified model path, fall back to default if not found
MODEL_PATHS = [
    os.getenv("MODEL_PATH", "models/faster-whisper-base.en")  # User specified or default  # Another common name
]

# Find the first valid model path
selected_model_path = None
for path in MODEL_PATHS:
    # Check relative to workspace root
    workspace_root = Path(__file__).parent.parent.parent
    full_path = os.path.join(workspace_root, path)
    
    if os.path.exists(full_path) and os.path.exists(os.path.join(full_path, "model.bin")):
        selected_model_path = path
        logger.info(f"Found valid model at: {full_path}")
        break

if not selected_model_path:
    logger.warning(f"No valid model found in any of the specified paths. Will attempt to download model: {MODEL_SIZE}")
    selected_model_path = None
else:
    logger.info(f"Using model from path: {selected_model_path}")

MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "25"))
PORT = int(os.getenv("PORT", "3001"))
HOST = os.getenv("HOST", "0.0.0.0")

# Initialize transcription service with model path or size
transcription_service = TranscriptionService(
    model_size=MODEL_SIZE,
    model_path=selected_model_path
)

# For server shutdown handling
server_instance = None

# Health check endpoint
@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {"status": "ok", "faster_whisper_api": "1.0.0"}

# Shutdown endpoint for graceful termination
@app.post("/shutdown")
async def shutdown() -> Dict[str, str]:
    """Shutdown the server gracefully"""
    logger.info("Received shutdown request")
    
    # Schedule server shutdown
    def stop_server():
        logger.info("Shutting down server...")
        if server_instance:
            server_instance.should_exit = True
    
    # Run shutdown in separate thread to allow response to be sent
    threading.Thread(target=stop_server).start()
    return {"message": "Server shutting down"}

# Transcription endpoint
@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    temperature: float = Form(0.0),
    beam_size: int = Form(1),  # Allow customizing beam size
):
    """
    Transcribe an audio file and return the transcription.
    
    Args:
        file: Audio file to transcribe
        temperature: Sampling temperature for generation
        beam_size: Beam size for transcription (higher = more accurate but slower)
        
    Returns:
        Transcription result
    """
    # Get original filename
    filename = file.filename
    
    # Log the request
    logger.info(f"Received transcription request for: {filename} (beam_size={beam_size})")
    
    try:
        # Perform basic validation before processing
        if not filename:
            raise ValueError("No filename provided")
        
        # Call the transcription service
        result = await transcription_service.transcribe_file_upload(file, filename)
        
        # Log the completion with proper timing
        total_time = result.get("total_ms", 0)
        logger.info(f"Transcription completed in {total_time}ms")
        
        return result
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle other errors
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get a list of available models"""
    return transcription_service.get_available_models()

@app.get("/convert", status_code=200)
async def convert_model(background_tasks: BackgroundTasks):
    """Convert the vanilla Whisper model to CTranslate2 format"""
    try:
        # Import the conversion module
        from convert_model import convert_model as convert_whisper_model
        
        # Define input and output paths
        input_path = os.path.join(Path(__file__).parent.parent.parent, "models/whisper-base.en")
        output_path = os.path.join(Path(__file__).parent.parent.parent, "models/faster-whisper-base.en-converted")
        
        # Start conversion in the background
        async def run_conversion():
            success = convert_whisper_model(input_path, output_path, "int8_float16")
            if success:
                logger.info("Model conversion completed successfully")
            else:
                logger.error("Model conversion failed")
        
        background_tasks.add_task(run_conversion)
        return {"message": "Model conversion started in background. Check logs for progress."}
    except Exception as e:
        logger.error(f"Error starting model conversion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True) 