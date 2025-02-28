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
logger.add("logs/groq_api.log", rotation="10 MB", retention="3 days", level="DEBUG")

# Create FastAPI app
app = FastAPI(
    title="Groq Transcription API",
    description="API for transcribing audio files using the Groq API",
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "distil-whisper-large-v3-en")
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "25"))
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Initialize transcription service
transcription_service = TranscriptionService(
    api_key=GROQ_API_KEY,
    model=DEFAULT_MODEL
)

# For server shutdown handling
server_instance = None

@app.get("/")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {"status": "ok", "message": "Groq Transcription API is running"}

@app.post("/shutdown")
async def shutdown() -> Dict[str, Any]:
    """Shutdown the server gracefully"""
    logger.info("Shutdown request received, stopping server...")
    # Send SIGTERM to the current process
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "ok", "message": "Server is shutting down"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Transcribe an audio file
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Check if the transcribe_file_upload method exists (our new implementation)
        if hasattr(transcription_service, 'transcribe_file_upload'):
            # Use our new method with detailed performance metrics
            result = transcription_service.transcribe_file_upload(file.file, file.filename)
            
            # Return the complete result including performance metrics
            return result
        else:
            # Use the old async transcribe method (for backward compatibility)
            # First preprocess the file
            preprocessed_filename, preprocessed_path = preprocess_audio(file.file, file.filename)
            
            # Then transcribe it
            result = await transcription_service.transcribe(
                file_path=preprocessed_path
            )
            
            return result
            
    except ValueError as e:
        # Handle validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle other errors
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

@app.get("/models")
async def get_models() -> Dict[str, Any]:
    """Get available transcription models"""
    return transcription_service.get_available_models()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True) 