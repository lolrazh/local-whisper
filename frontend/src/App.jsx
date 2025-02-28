import { useState, useRef, useEffect } from 'react';
import { BACKENDS } from './config';

// Smaller microphone button component
function MicrophoneButton({ isRecording, onClick, disabled }) {
  return (
    <button
      className={`flex justify-center items-center w-12 h-12 rounded-full transition duration-300 ${
        isRecording 
          ? 'bg-red-500 hover:bg-red-600' 
          : 'bg-black hover:bg-gray-800'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={isRecording ? 'Stop recording' : 'Start recording'}
    >
      <svg 
        xmlns="http://www.w3.org/2000/svg" 
        viewBox="0 0 24 24" 
        fill="white" 
        className="w-5 h-5"
      >
        <path fillRule="evenodd" d="M13 6a1 1 0 1 0-2 0v4a1 1 0 1 0 2 0V6zm-1 8a1 1 0 0 1 1 1v3a1 1 0 1 1-2 0v-3a1 1 0 0 1 1-1z" clipRule="evenodd"/>
        <path fillRule="evenodd" d="M12 2a4 4 0 0 0-4 4v4a4 4 0 0 0 8 0V6a4 4 0 0 0-4-4zm-2 4a2 2 0 1 1 4 0v4a2 2 0 1 1-4 0V6z" clipRule="evenodd"/>
        <path d="M7 12a1 1 0 0 1 1 1 4 4 0 0 0 8 0 1 1 0 1 1 2 0 6 6 0 0 1-5 5.92V21a1 1 0 1 1-2 0v-2.08A6 6 0 0 1 6 13a1 1 0 0 1 1-1z"/>
      </svg>
    </button>
  );
}

// Backend selector component
function BackendSelector({ backends, activeBackend, onChange, disabled }) {
  return (
    <select
      className="p-2 bg-white border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-black"
      value={activeBackend}
      onChange={e => onChange(e.target.value)}
      disabled={disabled}
    >
      {Object.keys(backends).map(backendKey => (
        <option key={backendKey} value={backendKey}>
          {backends[backendKey].name}
        </option>
      ))}
    </select>
  );
}

// Audio recording logic
function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [logs, setLogs] = useState([]);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  
  const addLog = (message) => {
    setLogs(prevLogs => [...prevLogs, { 
      id: Date.now(), 
      message, 
      timestamp: new Date().toLocaleTimeString() 
    }]);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      addLog('Recording started');
    } catch (error) {
      console.error('Error starting recording:', error);
      addLog(`Error: ${error.message}`);
    }
  };

  const stopRecording = () => {
    return new Promise((resolve) => {
      if (!mediaRecorderRef.current || mediaRecorderRef.current.state === 'inactive') {
        resolve(null);
        return;
      }

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const tracks = mediaRecorderRef.current.stream.getTracks();
        tracks.forEach(track => track.stop());
        
        setIsRecording(false);
        addLog('Recording stopped');
        resolve(audioBlob);
      };

      mediaRecorderRef.current.stop();
    });
  };

  return {
    isRecording,
    startRecording,
    stopRecording,
    logs,
    addLog
  };
}

function App() {
  const [transcription, setTranscription] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeBackend, setActiveBackend] = useState(Object.keys(BACKENDS)[0]);
  const [backendInitialized, setBackendInitialized] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const { isRecording, startRecording, stopRecording, logs, addLog } = useAudioRecorder();
  const [logPanelWidth, setLogPanelWidth] = useState(224);
  const isDraggingRef = useRef(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);

  const handleMicrophoneClick = async () => {
    if (isRecording) {
      setIsProcessing(true);
      const audioBlob = await stopRecording();
      
      if (audioBlob) {
        try {
          const transcriptionResult = await BACKENDS[activeBackend].transcribe(audioBlob);
          setTranscription(transcriptionResult);
          addLog(`Transcription received from ${BACKENDS[activeBackend].name}`);
        } catch (error) {
          console.error('Transcription error:', error);
          addLog(`Transcription error: ${error.message}`);
        }
      }
      setIsProcessing(false);
    } else {
      if (!backendInitialized) {
        addLog(`Please select and initialize a backend first`);
        return;
      }
      setTranscription('');
      await startRecording();
    }
  };
  
  const handleBackendChange = async (newBackend) => {
    if (isRecording || isProcessing || isInitializing) return;
    
    setActiveBackend(newBackend);
    setBackendInitialized(false);
    setTranscription('');
    addLog(`Backend changed to ${BACKENDS[newBackend].name}`);
    
    // Auto-initialize the backend
    await initializeBackend(newBackend);
  };
  
  const initializeBackend = async (backend) => {
    setIsInitializing(true);
    addLog(`Initializing ${BACKENDS[backend].name}...`);
    
    try {
      const success = await BACKENDS[backend].initialize(addLog);
      setBackendInitialized(success);
      if (success) {
        addLog(`${BACKENDS[backend].name} ready to use`);
      } else {
        addLog(`Failed to initialize ${BACKENDS[backend].name}`);
      }
    } catch (error) {
      addLog(`Error initializing backend: ${error.message}`);
      setBackendInitialized(false);
    } finally {
      setIsInitializing(false);
    }
  };
  
  const handleDragStart = (e) => {
    isDraggingRef.current = true;
    startXRef.current = e.clientX;
    startWidthRef.current = logPanelWidth;
    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
    
    // Prevent text selection during drag
    e.preventDefault();
  };
  
  useEffect(() => {
    const handleDragMove = (e) => {
      if (!isDraggingRef.current) return;
      
      const deltaX = startXRef.current - e.clientX;
      const newWidth = Math.max(180, Math.min(500, startWidthRef.current + deltaX));
      setLogPanelWidth(newWidth);
    };
    
    const handleDragEnd = () => {
      if (isDraggingRef.current) {
        isDraggingRef.current = false;
        document.body.style.cursor = 'default';
        document.body.style.userSelect = 'auto';
      }
    };
    
    document.addEventListener('mousemove', handleDragMove);
    document.addEventListener('mouseup', handleDragEnd);
    
    return () => {
      document.removeEventListener('mousemove', handleDragMove);
      document.removeEventListener('mouseup', handleDragEnd);
    };
  }, []);

  // Auto-initialize the first backend on mount
  useEffect(() => {
    initializeBackend(activeBackend);
  }, []);

  return (
    <div className="min-h-screen flex">
      <div className="flex-1 flex flex-col h-screen relative">
        {/* App Title in top left */}
        <div className="absolute top-4 left-4 app-title">
          LocalWhisper
        </div>
        
        {/* Centered Content */}
        <div className="flex-1 flex flex-col justify-center items-center px-8">
          <div className="w-full max-w-md flex flex-col items-center">
            <div className="w-full border border-gray-300 rounded-md bg-white transcription-box mb-4 relative">
              {/* Transcription area */}
              <div className="p-4" style={{ minHeight: "180px" }}>
                {isProcessing ? (
                  <p className="text-gray-500">Processing audio...</p>
                ) : transcription ? (
                  <p>{transcription}</p>
                ) : (
                  <p className="text-gray-500">Transcription will appear here...</p>
                )}
              </div>
              
              {/* Control bar at the bottom */}
              <div className="flex items-center justify-between p-2 border-t border-gray-200 bg-gray-50 rounded-b-md">
                {/* Backend selector on the left */}
                <BackendSelector 
                  backends={BACKENDS} 
                  activeBackend={activeBackend}
                  onChange={handleBackendChange}
                  disabled={isRecording || isProcessing || isInitializing}
                />
                
                {/* Status indicator in the middle */}
                <div className="text-sm text-gray-500">
                  {isInitializing ? (
                    <span className="flex items-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-black" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Initializing...
                    </span>
                  ) : isProcessing ? (
                    <span className="flex items-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-black" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Processing...
                    </span>
                  ) : isRecording ? (
                    <span className="flex items-center">
                      <span className="h-2 w-2 bg-red-500 rounded-full mr-2 animate-pulse"></span>
                      Recording...
                    </span>
                  ) : backendInitialized ? (
                    <span className="flex items-center">
                      <span className="h-2 w-2 bg-green-500 rounded-full mr-2"></span>
                      Ready
                    </span>
                  ) : (
                    <span className="flex items-center">
                      <span className="h-2 w-2 bg-yellow-500 rounded-full mr-2"></span>
                      Not initialized
                    </span>
                  )}
                </div>
                
                {/* Microphone button on the right */}
                <MicrophoneButton 
                  isRecording={isRecording} 
                  onClick={handleMicrophoneClick}
                  disabled={isProcessing || isInitializing || (!backendInitialized && !isRecording)}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Resizable Log Panel */}
      <div className="relative h-screen">
        {/* Drag Handle */}
        <div 
          className="absolute top-0 bottom-0 left-0 w-1 cursor-ew-resize z-10 hover:bg-gray-400"
          onMouseDown={handleDragStart}
        ></div>
        
        <div 
          className="h-full border-l border-gray-200 p-4 overflow-auto bg-gray-50 logs-panel" 
          style={{ width: `${logPanelWidth}px` }}
        >
          <h2 className="font-medium mb-2">Logs</h2>
          <div className="space-y-1">
            {logs.length === 0 ? (
              <p className="text-gray-500 text-sm">No logs yet...</p>
            ) : (
              logs.map((log) => (
                <div key={log.id} className="text-sm">
                  <span className="text-gray-500">[{log.timestamp}]</span> {log.message}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
