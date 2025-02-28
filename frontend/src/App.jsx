import { useState, useRef, useEffect } from 'react';
import { ACTIVE_BACKEND, BACKENDS } from './config';

// Audio recording component
function MicrophoneButton({ isRecording, onClick, disabled }) {
  return (
    <button
      className={`relative flex justify-center items-center w-20 h-20 rounded-xl transition duration-300 ${
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
        className="w-8 h-8"
      >
        <path fillRule="evenodd" d="M13 6a1 1 0 1 0-2 0v4a1 1 0 1 0 2 0V6zm-1 8a1 1 0 0 1 1 1v3a1 1 0 1 1-2 0v-3a1 1 0 0 1 1-1z" clipRule="evenodd"/>
        <path fillRule="evenodd" d="M12 2a4 4 0 0 0-4 4v4a4 4 0 0 0 8 0V6a4 4 0 0 0-4-4zm-2 4a2 2 0 1 1 4 0v4a2 2 0 1 1-4 0V6z" clipRule="evenodd"/>
        <path d="M7 12a1 1 0 0 1 1 1 4 4 0 0 0 8 0 1 1 0 1 1 2 0 6 6 0 0 1-5 5.92V21a1 1 0 1 1-2 0v-2.08A6 6 0 0 1 6 13a1 1 0 0 1 1-1z"/>
      </svg>
    </button>
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
  const [activeBackend] = useState(ACTIVE_BACKEND);
  const { isRecording, startRecording, stopRecording, logs } = useAudioRecorder();
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
        } catch (error) {
          console.error('Transcription error:', error);
        }
      }
      setIsProcessing(false);
    } else {
      setTranscription('');
      await startRecording();
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
            <div className="mb-8">
              <MicrophoneButton 
                isRecording={isRecording} 
                onClick={handleMicrophoneClick}
                disabled={isProcessing}
              />
            </div>
            
            <div 
              className="w-full border border-gray-300 rounded-md p-4 bg-white transcription-area" 
              style={{ minHeight: "180px" }}
            >
              {isProcessing ? (
                <p className="text-gray-500">Processing audio...</p>
              ) : transcription ? (
                <p>{transcription}</p>
              ) : (
                <p className="text-gray-500">Transcription will appear here...</p>
              )}
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
