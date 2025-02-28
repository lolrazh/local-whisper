import { useState, useRef, useEffect } from 'react';
import { BACKENDS } from './config';

// Microphone button component with square icon when recording
function MicrophoneButton({ isRecording, onClick, disabled }) {
  return (
    <button
      className={`flex justify-center items-center w-10 h-10 rounded-md transition duration-300 ${
        isRecording 
          ? 'bg-red-500 hover:bg-red-600' 
          : 'bg-black hover:bg-gray-800'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={isRecording ? 'Stop recording' : 'Start recording'}
    >
      {isRecording ? (
        // Square icon when recording
        <div className="w-4 h-4 bg-white rounded-sm"></div>
      ) : (
        // Microphone icon when not recording
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
      )}
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

// Performance metrics display component
function PerformanceMetrics({ metrics }) {
  if (!metrics) return null;
  
  return (
    <div className="mt-2 text-xs text-gray-500 border-t border-gray-200 pt-2">
      <div className="font-medium mb-1">Performance Metrics:</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <div>Total time:</div>
        <div className="text-right font-mono">{metrics.total_ms || "-"}ms</div>
        
        {metrics.preprocessing_ms && (
          <>
            <div>Pre-processing:</div>
            <div className="text-right font-mono">{metrics.preprocessing_ms}ms</div>
          </>
        )}
        
        {metrics.model_inference_ms && (
          <>
            <div>Model inference:</div>
            <div className="text-right font-mono">{metrics.model_inference_ms}ms</div>
          </>
        )}
        
        {metrics.api_call_ms && (
          <>
            <div>API call:</div>
            <div className="text-right font-mono">{metrics.api_call_ms}ms</div>
          </>
        )}
        
        {metrics.overhead_ms && (
          <>
            <div>Overhead:</div>
            <div className="text-right font-mono">{metrics.overhead_ms}ms</div>
          </>
        )}
      </div>
    </div>
  );
}

function App() {
  const [transcription, setTranscription] = useState('');
  const [performanceMetrics, setPerformanceMetrics] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeBackend, setActiveBackend] = useState('null');
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
          const result = await BACKENDS[activeBackend].transcribe(audioBlob);
          
          // Handle both simple string returns and object returns with performance data
          if (typeof result === 'string') {
            setTranscription(result);
            setPerformanceMetrics(null);
          } else {
            // Extract the transcription text only
            setTranscription(result.text || "");
            
            // Format performance metrics
            if (result.total_ms || (result.performance && result.performance.total_ms)) {
              // Initialize metrics object
              const formattedMetrics = {
                total_ms: result.total_ms || result.performance?.total_ms
              };
              
              // Process Whisper metrics - direct properties at the root level
              if (result.preprocessing_ms !== undefined) {
                formattedMetrics.preprocessing_ms = result.preprocessing_ms;
              }
              
              if (result.model_inference_ms !== undefined) {
                formattedMetrics.model_inference_ms = result.model_inference_ms;
              }
              
              if (result.overhead_ms !== undefined) {
                formattedMetrics.overhead_ms = result.overhead_ms;
              }
              
              // Process Groq API metrics from performance object
              if (result.performance) {
                if (result.performance.preprocessing_ms) {
                  formattedMetrics.preprocessing_ms = result.performance.preprocessing_ms;
                }
                
                if (result.performance.api_call_ms) {
                  formattedMetrics.model_inference_ms = result.performance.api_call_ms;
                }
                
                // Calculate overhead if not provided
                const measuredTime = 
                  (formattedMetrics.preprocessing_ms || 0) + 
                  (formattedMetrics.model_inference_ms || 0);
                
                if (formattedMetrics.total_ms > measuredTime) {
                  formattedMetrics.overhead_ms = formattedMetrics.total_ms - measuredTime;
                }
              }
              
              // Legacy format support (kept for backward compatibility)
              if (result.duration) {
                // Convert seconds to ms for consistency
                formattedMetrics.total_ms = Math.round(result.duration * 1000);
                
                // Extract any available performance metrics if they're in the response
                if (result.audio_preprocessing) {
                  formattedMetrics.preprocessing_ms = Math.round(result.audio_preprocessing * 1000);
                }
                
                if (result.model_inference) {
                  formattedMetrics.model_inference_ms = Math.round(result.model_inference * 1000);
                }
                
                if (result.overhead) {
                  formattedMetrics.overhead_ms = Math.round(result.overhead * 1000);
                }
              }
              
              setPerformanceMetrics(formattedMetrics);
            } else {
              setPerformanceMetrics(null);
            }
          }
          
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
      setPerformanceMetrics(null);
      await startRecording();
    }
  };
  
  const handleBackendChange = async (newBackend) => {
    if (isRecording || isProcessing || isInitializing) return;
    
    // Don't reinitialize if it's the same backend
    if (newBackend === activeBackend) return;
    
    setActiveBackend(newBackend);
    setBackendInitialized(false);
    setTranscription('');
    
    if (newBackend === 'null') {
      addLog('Please select a valid backend');
      return;
    }
    
    addLog(`Switching to ${BACKENDS[newBackend].name}`);
    
    // Initialize the new backend (this will also handle shutting down previous backends)
    await initializeBackend(newBackend);
  };
  
  const initializeBackend = async (backend) => {
    if (backend === 'null') return;
    
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

  // No auto-initialization for the null backend
  useEffect(() => {
    // We don't auto-initialize anymore since default is 'null'
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
                  <>
                    <p>{transcription}</p>
                    <PerformanceMetrics metrics={performanceMetrics} />
                  </>
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
