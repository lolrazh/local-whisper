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
    initialize: async (addLog) => {
      addLog('Mock backend initialized');
      return true;
    },
    transcribe: async (audioBlob) => {
      // Mock response after a delay to simulate processing
      await new Promise(resolve => setTimeout(resolve, 800));
      return "This is a mock transcription from the mock backend.";
    }
  },
  
  // Groq API backend
  groqApi: {
    name: 'Groq API',
    url: 'http://localhost:8000',
    initialize: async (addLog) => {
      addLog('Initializing Groq API backend...');
      try {
        const response = await fetch(`${BACKENDS.groqApi.url}/`, {
          method: 'GET',
        });
        
        if (!response.ok) {
          throw new Error(`API health check failed: ${response.status}`);
        }
        
        const data = await response.json();
        addLog(`Groq API backend initialized: ${data.message}`);
        return true;
      } catch (error) {
        addLog(`Failed to initialize Groq API: ${error.message}`);
        return false;
      }
    },
    transcribe: async (audioBlob) => {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');
      
      const response = await fetch(`${BACKENDS.groqApi.url}/transcribe`, {
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
  
  // Local Whisper backend (example for when you implement it)
  localWhisper: {
    name: 'Local Whisper',
    url: 'http://localhost:3000',
    initialize: async (addLog) => {
      addLog('Initializing Local Whisper backend...');
      // Simulate initialization
      await new Promise(resolve => setTimeout(resolve, 600));
      addLog('Local Whisper model loaded');
      return true;
    },
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
    initialize: async (addLog) => {
      addLog('Initializing Faster Whisper backend...');
      // Simulate initialization
      await new Promise(resolve => setTimeout(resolve, 1000));
      addLog('Faster Whisper model loaded successfully');
      return true;
    },
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
    initialize: async (addLog) => {
      addLog('Initializing Insanely Fast Whisper backend...');
      // Simulate initialization
      await new Promise(resolve => setTimeout(resolve, 1200));
      addLog('Insanely Fast Whisper model loaded successfully');
      return true;
    },
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