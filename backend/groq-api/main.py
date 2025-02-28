import os
import sys
import shutil
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn

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


@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"status": "ok", "message": "Groq Transcription API is running"}


@app.post("/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form("en"),
    temperature: float = Form(0.0),
    prompt: Optional[str] = Form(None)
):
    """
    Transcribe an audio file using the Groq API
    
    - **file**: Audio file to transcribe
    - **model**: Model to use for transcription (defaults to environment variable)
    - **language**: Language code (optional, defaults to English)
    - **temperature**: Temperature for generation (0-1, defaults to 0)
    - **prompt**: Context or specific vocabulary to help with transcription
    """
    # Validate file format
    if not is_valid_audio_format(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Supported formats: mp3, wav, m4a, flac, ogg, aac"
        )
    
    try:
        # Preprocess audio file for optimal transcription
        preprocessed_filename, preprocessed_path = preprocess_audio(file.file, file.filename)
        
        # Check file size
        file_size_mb = get_file_size_mb(preprocessed_path)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
            )
        
        # Use the specified model or default
        active_model = model or DEFAULT_MODEL
        
        # Transcribe audio
        result = await transcription_service.transcribe(
            file_path=preprocessed_path,
            language=language,
            temperature=temperature,
            prompt=prompt
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models for transcription"""
    return {
        "models": [
            {
                "id": "distil-whisper-large-v3-en",
                "name": "Distil Whisper Large v3 (English)",
                "description": "Optimized for English transcription"
            },
            {
                "id": "whisper-large-v3-turbo",
                "name": "Whisper Large v3 Turbo",
                "description": "Fast multi-language transcription"
            }
        ]
    }


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True) 