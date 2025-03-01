# Groq API Transcription Backend

This is a FastAPI-based backend for transcribing audio files using the Groq API with the `distil-whisper-large-v3-en` model.

## Features

- **High-Performance Transcription**: Leverage Groq API for fast, accurate transcriptions.
- **Audio Preprocessing**: Automatically optimize audio files for best transcription results.
- **Error Handling**: Comprehensive error handling and logging.
- **Configurable**: Easy configuration via environment variables.

## Requirements

- Python 3.8+
- FFmpeg installed on your system (for audio preprocessing)
- Groq API key

## Setup

### Windows

1. Ensure FFmpeg is installed on your system and added to PATH.

2. Update the `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

3. Install dependencies:
   ```powershell
   .\install_deps.ps1
   ```

4. Start the server:
   ```powershell
   .\start.ps1
   ```

### Linux/MacOS

1. Ensure FFmpeg is installed on your system:
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

2. Update the `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Start the server:
   ```bash
   ./start.sh
   ```

## API Endpoints

### Health Check
- `GET /`: Check if the API is running.

### Transcribe Audio
- `POST /transcribe`: Transcribe an audio file.
  - Parameters:
    - `file`: Audio file (multipart/form-data)

### List Models
- `GET /models`: List available models for transcription.

## Audio Preprocessing

The API automatically preprocesses audio files for optimal transcription:
- Downsample to 16KHz
- Convert to mono channel
- Convert to FLAC format for lossless compression

This follows the Groq API best practices for audio transcription.

## Environment Variables

- `GROQ_API_KEY`: Your Groq API key
- `DEFAULT_MODEL`: Default model to use for transcription (defaults to "distil-whisper-large-v3-en")
- `PORT`: Port to run the server on (defaults to 8000)
- `HOST`: Host to run the server on (defaults to "0.0.0.0") 