"""
Simple test script to verify the Faster Whisper API.
"""
import requests
import time
import os
import argparse
from pathlib import Path
import json

def test_health_check(base_url: str = "http://localhost:3001"):
    """Test the health check endpoint."""
    url = f"{base_url}/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        print(f"Health check passed: {response.json()}")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_models_endpoint(base_url: str = "http://localhost:3001"):
    """Test the models endpoint."""
    url = f"{base_url}/models"
    try:
        response = requests.get(url)
        response.raise_for_status()
        models_data = response.json()
        print(f"Models endpoint response: {json.dumps(models_data, indent=2)}")
        return True
    except Exception as e:
        print(f"Models endpoint test failed: {e}")
        return False

def test_transcription(audio_file_path: str, base_url: str = "http://localhost:3001"):
    """Test the transcription endpoint with the given audio file."""
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return False
    
    url = f"{base_url}/transcribe"
    
    try:
        with open(audio_file_path, 'rb') as f:
            start_time = time.time()
            
            # Create a file object with the original filename
            files = {'file': (os.path.basename(audio_file_path), f, 'audio/mpeg')}
            data = {'temperature': 0.0}
            
            print(f"Sending transcription request for {audio_file_path}...")
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            
            duration = time.time() - start_time
            result = response.json()
            
            print("\nTranscription result:")
            print(f"Text: {result['text']}")
            print(f"Total processing time: {result['total_ms']}ms (API call: {duration*1000:.2f}ms)")
            
            if 'metrics' in result:
                print("\nPerformance metrics:")
                for k, v in result['metrics'].items():
                    print(f"  {k}: {v}ms")
            
            return True
    except Exception as e:
        print(f"Transcription test failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the Faster Whisper API")
    parser.add_argument("--url", default="http://localhost:3001", help="Base URL of the API")
    parser.add_argument("--audio", default=None, help="Path to an audio file for testing transcription")
    
    args = parser.parse_args()
    
    print(f"Testing Faster Whisper API at {args.url}")
    
    health_ok = test_health_check(args.url)
    models_ok = test_models_endpoint(args.url)
    
    if args.audio:
        transcription_ok = test_transcription(args.audio, args.url)
        print(f"\nTest results: Health: {'✅' if health_ok else '❌'}, Models: {'✅' if models_ok else '❌'}, Transcription: {'✅' if transcription_ok else '❌'}")
    else:
        print(f"\nTest results: Health: {'✅' if health_ok else '❌'}, Models: {'✅' if models_ok else '❌'}")
        print("No audio file specified for transcription test.")

if __name__ == "__main__":
    main() 