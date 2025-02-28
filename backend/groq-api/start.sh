#!/bin/bash

# Activate virtual environment
source venv/Scripts/activate  # For Windows
# source venv/bin/activate    # Uncomment for Linux/Mac

# Install dependencies if needed
pip install -r requirements.txt

# Start the FastAPI server
python main.py 