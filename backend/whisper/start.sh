#!/bin/bash

# Detect OS and activate the appropriate virtual environment
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate  # Windows
else
    source venv/bin/activate  # Linux/Mac
fi

# Start the FastAPI server
python main.py 