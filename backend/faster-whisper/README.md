# Faster Whisper Backend

This backend uses the [Faster Whisper](https://github.com/guillaumekln/faster-whisper) library for transcription, which is an optimized implementation of OpenAI's Whisper model using CTranslate2.

## Features

- Improved transcription speed compared to the original Whisper model
- Voice activity detection (VAD) to filter out silence
- Support for various model sizes (base, small, medium, large-v3)
- GPU acceleration when available, with INT8 quantization option for CPU

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux/MacOS: `source venv/bin/activate`

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the server:
   ```bash
   python main.py
   ```
   or use the provided script:
   ```bash
   ./start.sh
   ```

## API Endpoints

- `GET /`: Health check
- `POST /shutdown`: Gracefully shut down the server
- `POST /transcribe`: Transcribe an audio file
- `GET /models`: Get a list of available models

## Testing

Use the included test script to test the API:

```bash
python test_api.py --audio path/to/audio/file.mp3
```

## Configuration

The following environment variables can be set:

- `MODEL_PATH`: Path to the Faster Whisper model (default: `models/faster-whisper-base.en`)
- `MAX_FILE_SIZE_MB`: Maximum audio file size in MB (default: 25)
- `PORT`: Server port (default: 3001)
- `HOST`: Server host (default: 0.0.0.0)

## Model Information

- **base.en**: ~150MB, fastest, less accurate
- **small.en**: ~500MB, fast with reasonable accuracy
- **medium.en**: ~1.5GB, slower but more accurate
- **large-v3**: ~3GB, slower but most accurate 