@echo off
REM Activate virtual environment and run the Faster Whisper backend server

echo Starting Faster Whisper Backend...
echo Using local model at models/faster-whisper-base.en
cd /d "%~dp0"

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Activate virtual environment
call venv\Scripts\activate

REM Optional: Uncomment to use a specific model
REM set MODEL_PATH=models/faster-whisper-base.en

REM Start the FastAPI server
python main.py

pause 