#!/bin/bash

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please create one using: python -m venv venv"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the FastAPI server
python main.py 