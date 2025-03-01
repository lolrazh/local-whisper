# Script to install dependencies in the virtual environment
Write-Host "Installing dependencies for Faster Whisper backend..."

# Check if virtual environment exists
if (-Not (Test-Path ".\venv")) {
    Write-Host "Virtual environment not found. Creating a new one..."
    python -m venv venv
    if (-Not $?) {
        Write-Error "Failed to create virtual environment. Please check your Python installation."
        exit 1
    }
    Write-Host "Virtual environment created successfully."
}

# Activate virtual environment and install dependencies
Write-Host "Activating virtual environment and installing dependencies..."
& .\venv\Scripts\python.exe -m pip install --upgrade pip
if (-Not $?) {
    Write-Error "Failed to upgrade pip. Continuing anyway..."
}

Write-Host "Installing dependencies from requirements.txt..."
& .\venv\Scripts\python.exe -m pip install -r requirements.txt
if (-Not $?) {
    Write-Error "Failed to install dependencies."
    exit 1
}

# Verify installation by importing key packages
Write-Host "Verifying installation..."
$verification_script = @'
import torch
print(f"Torch installed: {torch.__version__}")
import faster_whisper
print(f"Faster Whisper installed: OK")
import ctranslate2
print(f"CTranslate2 installed: OK")
'@

# Save verification script to a temporary file
$verification_file = "verify_imports.py"
$verification_script | Out-File -FilePath $verification_file -Encoding utf8

# Run the verification script
& .\venv\Scripts\python.exe $verification_file

# Clean up
Remove-Item -Path $verification_file -Force

Write-Host "Installation completed!"