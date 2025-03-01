import os
from typing import Dict, Any
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn

from transcription import TranscriptionService

# Configure logging
logger.add("logs/faster_whisper_api.log", rotation="10 MB")

# Create FastAPI app
app = FastAPI(
    title="Faster Whisper API",
    description="API for transcribing audio using Faster Whisper",
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
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "faster-whisper-base.en"))
PORT = int(os.getenv("PORT", "3001"))
HOST = os.getenv("HOST", "0.0.0.0")

# Initialize transcription service
transcription_service = TranscriptionService(model_path=MODEL_PATH)

# Health check endpoint
@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {"status": "ok"}

# Transcription endpoint
@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    temperature: float = Form(0.0),
):
    """Transcribe an audio file and return the transcription."""
    filename = file.filename
    logger.info(f"Received transcription request for: {filename}")
    
    try:
        if not filename:
            raise ValueError("No filename provided")
        
        result = await transcription_service.transcribe_file_upload(file, filename)
        logger.info(f"Transcription completed in {result.get('total_ms', 0)}ms")
        
        return result
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get a list of available models"""
    return transcription_service.get_available_models()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True) 