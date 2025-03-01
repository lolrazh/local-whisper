# Whisper Transcription Backend

A FastAPI-based backend for audio transcription using OpenAI's Whisper model.

## Features

- High-quality transcription using the official Whisper model
- Audio preprocessing with FFmpeg
- REST API for transcription services
- Cross-platform support (Windows, Linux, macOS)

## Setup

### Prerequisites

- Python 3.8+
- FFmpeg installed and available in PATH

### Installation

#### Windows

1. Create a virtual environment (if not already created):
   ```
   python -m venv venv
   ```

2. Install dependencies:
   ```
   .\install_deps.ps1
   ```

3. Start the server:
   ```
   .\start.ps1
   ```

#### Linux/MacOS

1. Create a virtual environment (if not already created):
   ```
   python -m venv venv
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the server:
   ```
   ./start.sh
   ```

## API Endpoints

### Health Check
```
GET /
```
Returns a simple message to confirm the server is running.

### Transcribe Audio
```
POST /transcribe
```
Transcribes an audio file and returns the transcription.

**Parameters:**
- `file`: The audio file to transcribe (multipart/form-data)
- `model`: (Optional) The model to use for transcription (default: "base")

**Response:**
```json
{
  "text": "Transcribed text",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Segment text"
    }
  ],
  "language": "en",
  "processing_time": 1.23
}
```

### List Models
```
GET /models
```
Returns a list of available Whisper models.

## Environment Variables

- `MODEL_PATH`: Path to the Whisper model (default: "base")
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 25)
- `PORT`: Port to run the server on (default: 8000)
- `HOST`: Host to run the server on (default: "0.0.0.0")

## Directory Structure

```
whisper/
├── main.py            # FastAPI server and endpoints
├── transcription.py   # Transcription service
├── audio_utils.py     # Audio preprocessing utilities
├── requirements.txt   # Dependencies
├── start.sh           # Unix startup script
├── start.ps1          # Windows startup script
├── install_deps.ps1   # Windows dependency installation script
└── logs/              # Log files
``` 