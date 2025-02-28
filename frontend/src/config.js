/**
 * Backend Configuration
 * 
 * This file allows easy switching between different backend implementations.
 * To switch backends, just change the ACTIVE_BACKEND constant to the key of one
 * of the available backends.
 */

// Set this value to switch between backends
export const ACTIVE_BACKEND = 'mockBackend';

// Backend definitions
export const BACKENDS = {
  // Mock backend for testing/development
  mockBackend: {
    name: 'Mock Backend',
    url: null,
    transcribe: async (audioBlob) => {
      // Mock response after a delay to simulate processing
      await new Promise(resolve => setTimeout(resolve, 800));
      return "This is a mock transcription from the mock backend.";
    }
  },
  
  // Local Whisper backend (example for when you implement it)
  localWhisper: {
    name: 'Local Whisper',
    url: 'http://localhost:3000',
    transcribe: async (audioBlob) => {
      const formData = new FormData();
      formData.append('audio', audioBlob);
      
      const response = await fetch(`${BACKENDS.localWhisper.url}/transcribe`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Transcription failed: ${response.status}`);
      }
      
      const data = await response.json();
      return data.text;
    }
  },
  
  // Faster-Whisper backend (example for when you implement it)
  fasterWhisper: {
    name: 'Faster Whisper',
    url: 'http://localhost:3001',
    transcribe: async (audioBlob) => {
      const formData = new FormData();
      formData.append('audio', audioBlob);
      
      const response = await fetch(`${BACKENDS.fasterWhisper.url}/transcribe`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Transcription failed: ${response.status}`);
      }
      
      const data = await response.json();
      return data.text;
    }
  },
  
  // InsanelyFastWhisper backend (example for when you implement it)
  insanelyFastWhisper: {
    name: 'Insanely Fast Whisper',
    url: 'http://localhost:3002',
    transcribe: async (audioBlob) => {
      const formData = new FormData();
      formData.append('audio', audioBlob);
      
      const response = await fetch(`${BACKENDS.insanelyFastWhisper.url}/transcribe`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Transcription failed: ${response.status}`);
      }
      
      const data = await response.json();
      return data.text;
    }
  }
}; 