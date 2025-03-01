#!/bin/bash

# Detect OS and activate the appropriate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies if needed
pip install -r requirements.txt

# Start the FastAPI server
python main.py 