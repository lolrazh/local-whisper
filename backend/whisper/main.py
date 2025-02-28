import os
import sys
import shutil
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn
import signal
import threading
from pydantic import BaseModel

from transcription import TranscriptionService
from audio_utils import preprocess_audio, is_valid_audio_format, get_file_size_mb

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add("logs/whisper_api.log", rotation="10 MB", retention="3 days", level="DEBUG")

# Create FastAPI app
app = FastAPI(
    title="Whisper Transcription API",
    description="API for transcribing audio files using the Vanilla Whisper model",
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
# Get the absolute path to the model directory
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "whisper-base.en"))
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "25"))
PORT = int(os.getenv("PORT", "8001"))
HOST = os.getenv("HOST", "0.0.0.0")

logger.info(f"Using model path: {MODEL_PATH}")

# Initialize transcription service
transcription_service = TranscriptionService(
    model_path=MODEL_PATH
)

# For server shutdown handling
server_instance = None

# Health check endpoint
@app.get("/")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "message": "Whisper API is running",
        "version": "1.0.0"
    }

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
    temperature: float = Form(0.0)
) -> Dict[str, Any]:
    """
    Transcribe an audio file using the Vanilla Whisper model.
    
    Args:
        file: Audio file to transcribe
        temperature: Model temperature (0.0 for deterministic output)
        
    Returns:
        Transcription result with performance metrics
    """
    logger.info(f"Received transcription request: {file.filename}, size: {len(await file.read())/(1024*1024):.2f}MB")
    await file.seek(0)  # Reset file cursor after reading
    
    # Check file size
    content = await file.read()
    await file.seek(0)  # Reset file cursor
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size_mb:.2f}MB. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )
    
    try:
        # Process transcription
        result = await transcription_service.transcribe_file_upload(file, file.filename)
        return result
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@app.get("/models")
async def get_models() -> Dict[str, Any]:
    """Get available transcription models"""
    return transcription_service.get_available_models()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True) 