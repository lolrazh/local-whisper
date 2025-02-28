#!/bin/bash
# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/Scripts/activate || source venv/bin/activate
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the server
python main.py 