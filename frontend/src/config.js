/**
 * Backend Configuration
 * 
 * This file allows easy switching between different backend implementations.
 * To switch backends, just change the ACTIVE_BACKEND constant to the key of one
 * of the available backends.
 */

// Set this value to switch between backends
export const ACTIVE_BACKEND = 'mockBackend';

// Store for active backend server process
let activeServerProcess = null;

// Function to start the backend server
const startBackendServer = async (serverType, addLog) => {
  try {
    // Only shut down a previous server if it's different from the one we're starting
    if (activeServerProcess && activeServerProcess.type !== serverType) {
      addLog(`Shutting down previous ${activeServerProcess.type} backend server...`);
      await fetch(`${activeServerProcess.url}/shutdown`, { method: 'POST' }).catch(() => {});
      activeServerProcess = null;
    }
    
    // If we already have an active server of the same type, don't restart it
    if (activeServerProcess && activeServerProcess.type === serverType) {
      addLog(`${serverType} server is already running`);
      return true;
    }
    
    // For now, we only have implementation for Groq API
    if (serverType === 'groqApi') {
      addLog('Using existing Groq API backend server...');
      // We'll assume the server is already running externally
      // and we'll just check if it's accessible
      activeServerProcess = {
        type: 'groqApi',
        url: 'http://localhost:8000'
      };
      
      return true;
    }
    
    return false;
  } catch (error) {
    addLog(`Error starting backend server: ${error.message}`);
    return false;
  }
};

// Backend definitions
export const BACKENDS = {
  // Default "Select backend" option
  null: {
    name: 'Select backend',
    url: null,
    initialize: async (addLog) => {
      addLog('Please select a backend to initialize');
      return false;
    },
    transcribe: async (audioBlob) => {
      throw new Error('No backend selected');
    }
  },
  
  // Mock backend for testing/development
  mockBackend: {
    name: 'Mock Backend',
    url: null,
    initialize: async (addLog) => {
      addLog('Mock backend initialized');
      
      // If there's an active Groq API server, shut it down
      if (activeServerProcess && activeServerProcess.type === 'groqApi') {
        addLog('Shutting down Groq API backend server...');
        await fetch(`${activeServerProcess.url}/shutdown`, { method: 'POST' }).catch(() => {});
        activeServerProcess = null;
      }
      
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
      
      // Start the backend server - this won't shut down an existing Groq API server
      const serverStarted = await startBackendServer('groqApi', addLog);
      if (!serverStarted) {
        addLog('Failed to start Groq API server. Make sure it\'s running at http://localhost:8000');
      }
      
      try {
        // Check if the server is running with a health check
        const response = await fetch(`${BACKENDS.groqApi.url}/`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json'
          }
        });
        
        if (!response.ok) {
          throw new Error(`API health check failed: ${response.status}`);
        }
        
        const data = await response.json();
        addLog(`Groq API backend initialized: ${data.message}`);
        
        // Test models endpoint to ensure API is fully functional
        try {
          const modelsResponse = await fetch(`${BACKENDS.groqApi.url}/models`);
          if (modelsResponse.ok) {
            const modelsData = await modelsResponse.json();
            addLog(`Available models: ${modelsData.models.map(m => m.id).join(', ')}`);
          }
        } catch (error) {
          addLog(`Warning: Could not fetch models list: ${error.message}`);
          // Continue anyway as this is not critical
        }
        
        return true;
      } catch (error) {
        addLog(`Failed to initialize Groq API: ${error.message}`);
        addLog('Please make sure the backend server is running at http://localhost:8000');
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