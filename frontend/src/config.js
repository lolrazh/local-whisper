/**
 * Backend Configuration
 * 
 * This file allows easy switching between different backend implementations.
 * To switch backends, just change the ACTIVE_BACKEND constant to the key of one
 * of the available backends.
 * 
 * NOTE: In the final Electron app, this file would use electron's child_process module
 * to actually spawn and manage the Python backend servers.
 */

// Set this value to switch between backends
export const ACTIVE_BACKEND = 'mockBackend';

// Store for active backend server process
let activeServerProcess = null;

// Function to manage backend server lifecycle
const manageBackendServer = async (action, serverType, addLog) => {
  try {
    // In a real Electron app, this would use electron.child_process to spawn Python processes
    
    if (action === 'start') {
      // If we already have this server running, don't start it again
      if (activeServerProcess && activeServerProcess.type === serverType) {
        addLog(`${serverType} server is already running`);
        return true;
      }
      
      // Simulate starting the server
      // In a real Electron app, we would:
      // 1. Spawn a child process for the appropriate Python backend
      // 2. Listen for stdout/stderr for logging
      // 3. Store the process reference for later management
      
      addLog(`Starting ${serverType} backend server...`);
      
      // Placeholder for the real server process
      // In a real implementation with Electron:
      // const process = require('child_process').spawn('python', ['path/to/backend/main.py']);
      // process.stdout.on('data', (data) => addLog(`Server: ${data}`));
      // process.stderr.on('data', (data) => addLog(`Server error: ${data}`));
      
      activeServerProcess = {
        type: serverType,
        url: serverType === 'groqApi' ? 'http://localhost:8000' : 
             serverType === 'whisper' ? 'http://localhost:8001' :
             `http://localhost:300${Math.floor(Math.random() * 9)}`,
        // In real implementation: process: process
      };
      
      // In development mode, we're assuming the server is already running
      // and just checking if it's accessible
      
      return true;
    } 
    else if (action === 'stop') {
      if (activeServerProcess && (serverType === 'all' || activeServerProcess.type === serverType)) {
        addLog(`Shutting down ${activeServerProcess.type} backend server...`);
        
        // In a real electron app, we would kill the child process
        // if (activeServerProcess.process) {
        //   activeServerProcess.process.kill();
        // }
        
        // For now, we'll try to call the shutdown endpoint if available
        try {
          await fetch(`${activeServerProcess.url}/shutdown`, { 
            method: 'POST',
            headers: { 'Accept': 'application/json' }
          }).catch(() => {});
        } catch (error) {
          // Ignore errors in development mode
        }
        
        activeServerProcess = null;
        return true;
      }
      return false;
    }
    
    return false;
  } catch (error) {
    addLog(`Error managing backend server: ${error.message}`);
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
      
      // When "null" is selected, shut down any running server
      await manageBackendServer('stop', 'all', addLog);
      
      return false;
    },
    transcribe: async (audioBlob) => {
      throw new Error('No backend selected');
    }
  },
  
  // Groq API backend
  groqApi: {
    name: 'Groq API',
    url: 'http://localhost:8000',
    initialize: async (addLog) => {
      addLog('Initializing Groq API backend...');
      
      // In development mode, we'll assume the backend server is already running
      // and just check if it's accessible.
      // In the Electron app, this would actually start the server.
      try {
        // Check if the server is running with a health check
        const response = await fetch(`${BACKENDS.groqApi.url}/`, {
          method: 'GET',
          headers: { 'Accept': 'application/json' }
        });
        
        if (!response.ok) {
          throw new Error(`API health check failed: ${response.status}`);
        }
        
        const data = await response.json();
        addLog(`Groq API backend initialized: ${data.message}`);
        
        // Update the active server process
        activeServerProcess = {
          type: 'groqApi',
          url: 'http://localhost:8000'
        };
        
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
        
        // In development mode, we'll just fail initialization
        // In the real app, we'd try to restart the server or give more diagnostics
        return false;
      }
    },
    transcribe: async (audioBlob) => {
      // Make sure our server is still running
      if (!activeServerProcess || activeServerProcess.type !== 'groqApi') {
        throw new Error('Groq API server is not running. Please reinitialize the backend.');
      }
      
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');
      
      const response = await fetch(`${BACKENDS.groqApi.url}/transcribe`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Transcription failed: ${response.status}`);
      }
      
      // Return the full response data including performance metrics
      return await response.json();
    }
  },
  
  // Vanilla Whisper backend
  whisper: {
    name: 'Vanilla Whisper',
    url: 'http://localhost:8001',
    initialize: async (addLog) => {
      // Shut down any existing server
      await manageBackendServer('stop', 'all', addLog);
      
      addLog('Initializing Vanilla Whisper backend...');
      
      // In the real app, this would start the Whisper Python server
      const serverStarted = await manageBackendServer('start', 'whisper', addLog);
      
      try {
        // Check if the server is running with a health check
        const response = await fetch(`${BACKENDS.whisper.url}/`, {
          method: 'GET',
          headers: { 'Accept': 'application/json' }
        });
        
        if (!response.ok) {
          throw new Error(`API health check failed: ${response.status}`);
        }
        
        const data = await response.json();
        addLog(`Vanilla Whisper backend initialized: ${data.message}`);
        
        // Test models endpoint to ensure API is fully functional
        try {
          const modelsResponse = await fetch(`${BACKENDS.whisper.url}/models`);
          if (modelsResponse.ok) {
            const modelsData = await modelsResponse.json();
            addLog(`Available model: ${modelsData.models.map(m => m.id).join(', ')}`);
          }
        } catch (error) {
          addLog(`Warning: Could not fetch models list: ${error.message}`);
          // Continue anyway as this is not critical
        }
        
        return true;
      } catch (error) {
        addLog(`Failed to initialize Vanilla Whisper backend: ${error.message}`);
        addLog('Please make sure the backend server is running at http://localhost:8001');
        return false;
      }
    },
    transcribe: async (audioBlob) => {
      // Make sure our server is still running
      if (!activeServerProcess || activeServerProcess.type !== 'whisper') {
        throw new Error('Vanilla Whisper server is not running. Please reinitialize the backend.');
      }
      
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');
      
      const response = await fetch(`${BACKENDS.whisper.url}/transcribe`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Transcription failed: ${response.status}`);
      }
      
      // Return the full response data including performance metrics
      return await response.json();
    }
  },
  
  // Faster-Whisper backend (example for when you implement it)
  fasterWhisper: {
    name: 'Faster-Whisper',
    url: 'http://localhost:3001',
    initialize: async (addLog) => {
      // Shut down any existing server
      await manageBackendServer('stop', 'all', addLog);
      
      addLog('Initializing Faster-Whisper backend...');
      // In the real app, this would start the Faster Whisper Python server
      const serverStarted = await manageBackendServer('start', 'fasterWhisper', addLog);
      
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
}; 