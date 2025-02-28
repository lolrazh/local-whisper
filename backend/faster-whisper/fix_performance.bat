@echo off
echo ===== Faster Whisper Performance Fix =====
echo This script will properly convert your Whisper model to CTranslate2 format

:: Get the script directory and workspace root
set "SCRIPT_DIR=%~dp0"
set "WORKSPACE_ROOT=%SCRIPT_DIR%..\..\"

:: Remove trailing slash from paths
if %SCRIPT_DIR:~-1%==\ set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
if %WORKSPACE_ROOT:~-1%==\ set "WORKSPACE_ROOT=%WORKSPACE_ROOT:~0,-1%"

echo Script directory: %SCRIPT_DIR%
echo Workspace root: %WORKSPACE_ROOT%

:: Check if the virtual environment exists
if not exist "%WORKSPACE_ROOT%\venv" (
    echo Creating virtual environment...
    python -m venv "%WORKSPACE_ROOT%\venv"
)

:: Activate the virtual environment
call "%WORKSPACE_ROOT%\venv\Scripts\activate.bat"

:: Install required packages
echo Installing required packages...
pip install -r "%SCRIPT_DIR%\..\requirements.txt"
pip install -r "%SCRIPT_DIR%\requirements.txt"
pip install ctranslate2 transformers[torch] --upgrade

:: Check if the models directory exists
if not exist "%WORKSPACE_ROOT%\models" (
    echo Creating models directory...
    mkdir "%WORKSPACE_ROOT%\models"
)

:: Check if the vanilla whisper model is available
set "WHISPER_MODEL_DIR=%WORKSPACE_ROOT%\models\whisper-base.en"
if not exist "%WHISPER_MODEL_DIR%" (
    echo Downloading Whisper model (base.en)...
    python -c "from transformers import WhisperForConditionalGeneration, WhisperProcessor; model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base.en'); processor = WhisperProcessor.from_pretrained('openai/whisper-base.en'); model.save_pretrained(r'%WHISPER_MODEL_DIR%'); processor.save_pretrained(r'%WHISPER_MODEL_DIR%')"
)

:: Create the output directory for the converted model
set "OUTPUT_MODEL_DIR=%WORKSPACE_ROOT%\models\faster-whisper-base.en-converted"
if exist "%OUTPUT_MODEL_DIR%" (
    echo Removing existing converted model...
    rmdir /s /q "%OUTPUT_MODEL_DIR%"
)

:: Convert the model
echo Converting Whisper model to CTranslate2 format with int8_float16 quantization...
python "%SCRIPT_DIR%\convert_model.py" --input "%WHISPER_MODEL_DIR%" --output "%OUTPUT_MODEL_DIR%" --quantization "int8_float16"

:: Verify the converted model
if exist "%OUTPUT_MODEL_DIR%\model.bin" if exist "%OUTPUT_MODEL_DIR%\config.json" (
    echo.
    echo ✓ Success! Model converted successfully to CTranslate2 format.
    echo.
    echo You can now use this model by setting:
    echo    MODEL_PATH="models/faster-whisper-base.en-converted"
    echo.
    echo Or simply restart the Faster Whisper server, which will now automatically
    echo detect and use this optimized model.
) else (
    echo.
    echo ✗ Error: Conversion failed. The model files are missing.
    exit /b 1
)

echo.
echo To get maximum performance, make sure to:
echo 1. Use beam_size=1 for faster transcription (default in updated code)
echo 2. Keep the optimized computation types (already set in the code)

echo.
echo You can restart the server now.
pause 