# Groq API Transcription Backend

This is a FastAPI-based backend for transcribing audio files using the Groq API with the `distil-whisper-large-v3-en` model.

## Features

- **High-Performance Transcription**: Leverage Groq API for fast, accurate transcriptions.
- **Audio Preprocessing**: Automatically optimize audio files for best transcription results.
- **Asynchronous API**: Non-blocking design for handling multiple requests.
- **Error Handling**: Comprehensive error handling and logging.
- **Configurable**: Easy configuration via environment variables.

## Requirements

- Python 3.8+
- FFmpeg installed on your system (for audio preprocessing)
- Groq API key

## Setup

1. Ensure FFmpeg is installed on your system:
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

2. Update the `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Start the server:
   ```
   python main.py
   ```
   
   Or use the provided start script:
   ```
   ./start.sh
   ```

## API Endpoints

### Health Check
- `GET /`: Check if the API is running.

### Transcribe Audio
- `POST /transcribe`: Transcribe an audio file.
  - Parameters:
    - `file`: Audio file (multipart/form-data)
    - `model` (optional): Model to use for transcription
    - `language` (optional): Language code (defaults to "en")
    - `temperature` (optional): Temperature for generation (0-1, defaults to 0)
    - `prompt` (optional): Context or specific vocabulary to help with transcription

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
- `MAX_FILE_SIZE_MB`: Maximum allowed file size in MB (defaults to 25)
- `PORT`: Port to run the server on (defaults to 8000)
- `HOST`: Host to run the server on (defaults to "0.0.0.0") 