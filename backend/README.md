# Local Whisper Backend

This directory contains three separate backend implementations for speech-to-text transcription:

## Implementations

1. **Whisper** (`/whisper`): 
   - Standard OpenAI Whisper implementation
   - Uses the official Whisper model from OpenAI
   - Provides high-quality transcription with standard performance

2. **Faster-Whisper** (`/faster-whisper`):
   - High-performance implementation using the Faster-Whisper library
   - Optimized for speed while maintaining quality
   - Supports both CPU and GPU acceleration

3. **Groq API** (`/groq-api`):
   - Cloud-based implementation using Groq's API
   - Provides extremely fast transcription via Groq's LPU (Language Processing Unit)
   - Requires an API key and internet connection

## Common Features

All three implementations share:
- FastAPI-based REST API
- Audio preprocessing with FFmpeg
- Consistent API endpoints
- Cross-platform support (Windows, Linux, macOS)

## Setup Instructions

Each implementation has its own setup instructions in its respective directory. Generally:

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Install dependencies:
   - Windows: `.\install_deps.ps1`
   - Linux/Mac: `pip install -r requirements.txt`

3. Start the server:
   - Windows: `.\start.ps1`
   - Linux/Mac: `./start.sh`

## API Endpoints

All implementations provide these endpoints:
- `GET /` - Health check
- `POST /transcribe` - Transcribe audio file
- `GET /models` - List available models

## Recent Cleanup

The backend has been cleaned up to:
- Remove test files and unnecessary code
- Standardize audio processing across implementations
- Ensure consistent API behavior
- Improve error handling and logging
- Provide cross-platform startup scripts
- Update documentation

## Directory Structure

```
backend/
├── whisper/              # Standard Whisper implementation
├── faster-whisper/       # Faster-Whisper implementation
└── groq-api/             # Groq API implementation
```

Each implementation directory contains:
- `main.py` - FastAPI server and endpoints
- `transcription.py` - Transcription service
- `audio_utils.py` - Audio preprocessing utilities
- `requirements.txt` - Dependencies
- `start.sh` / `start.ps1` - Startup scripts
- `README.md` - Implementation-specific documentation 