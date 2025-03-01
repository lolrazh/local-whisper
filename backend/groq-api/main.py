import os
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn

from transcription import TranscriptionService
from audio_utils import preprocess_audio, is_valid_audio_format

# Load environment variables
load_dotenv()

# Configure logging
logger.add("logs/groq_api.log", rotation="10 MB")

# Create FastAPI app
app = FastAPI(
    title="Groq Transcription API",
    description="API for transcribing audio files using the Groq API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "distil-whisper-large-v3-en")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Initialize transcription service
transcription_service = TranscriptionService(
    api_key=GROQ_API_KEY,
    model=DEFAULT_MODEL
)

@app.get("/")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {"status": "ok", "message": "Groq Transcription API is running"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Transcribe an audio file
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Use our transcribe_file_upload method
        result = transcription_service.transcribe_file_upload(file.file, file.filename)
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