import React, { useRef, useState, useEffect } from 'react';

const App = () => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const canvasRef = useRef(null);

  const [stream, setStream] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState("Waiting for camera and microphone access...");

  const AUDIO_RECORD_DURATION_MS = 5000; // 5 seconds

  // 1. Request Camera and Microphone Access 
  useEffect(() => {
    const getMedia = async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        setStream(mediaStream);
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          videoRef.current.play();
        }
        setStatus("Ready to record!");
      } catch (err) {
        console.error("Error accessing media devices:", err);
        setError("Failed to get camera/microphone access. Please ensure permissions are granted.");
        setStatus("Access Denied.");
      }
    };

    getMedia();

    // Cleanup function to stop media stream when component unmounts
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // 2. Start Recording Audio 
const startRecording = () => {
    if (!stream) {
      setError("No media stream available. Please grant camera/mic access.");
      return;
    }
    if (isRecording) return;

    audioChunksRef.current = [];
    try {
      const audioStream = new MediaStream(stream.getAudioTracks());

      mediaRecorderRef.current = new MediaRecorder(audioStream, { mimeType: 'audio/webm' }); 

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        processAndSend();
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setStatus(`Recording audio for ${AUDIO_RECORD_DURATION_MS / 1000} seconds...`);
      setPredictionResult(null);
      setError(null);

      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
        }
      }, AUDIO_RECORD_DURATION_MS);

    } catch (err) {
      console.error("Error starting media recorder:", err);
      setError("Failed to start audio recording. Check browser support or try again.");
      setIsRecording(false);
    }
  };

  // 3. Capture Photo & Process/Send to Backend 
  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) {
      setError("Video or canvas not ready.");
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        resolve(blob);
      }, 'image/jpeg', 0.9); // Capture as JPEG with quality 0.9
    });
  };

  const processAndSend = async () => {
    setLoading(true);
    setStatus("Processing and sending data to backend...");
    setError(null);

    try {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      const imageBlob = await capturePhoto();

      if (!imageBlob || !audioBlob) {
        throw new Error("Failed to capture both image and audio blobs.");
      }

      const formData = new FormData();
      formData.append('image_file', imageBlob, 'photo.jpeg');
      formData.append('audio_file', audioBlob, 'audio.webm');

      // --- 4. Send to Backend ---
      const response = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Backend error: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const data = await response.json();
      setPredictionResult(data);
      setStatus("Prediction received!");
      console.log("Prediction Result:", data);

    } catch (err) {
      console.error("Error during prediction:", err);
      setError(`Prediction failed: ${err.message || err}`);
      setStatus("Prediction failed.");
    } finally {
      setLoading(false);
      setIsRecording(false); // Ensure recording state is reset
    }
  };

  const reset = () => {
    setPredictionResult(null);
    setError(null);
    setStatus("Ready to record!");
    setIsRecording(false);
    audioChunksRef.current = [];
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  };

  return (
    <div className="app-container">
      <h1>Multi-Modal Person Identification</h1>
      
      <div className="video-container">
        <video ref={videoRef} autoPlay muted playsInline></video>
        <canvas ref={canvasRef}></canvas>
      </div>

      <div className="controls">
        <button onClick={startRecording} disabled={!stream || isRecording || loading}>
          {isRecording ? "Recording..." : "Start Recording"}
        </button>
        <button onClick={reset} disabled={loading}>
          Reset
        </button>
      </div>

      {loading && (
        <div className="loading-spinner"></div>
      )}

      {error && (
        <p className="status-message error-message">{error}</p>
      )}

      {!loading && !error && (
        <p className="status-message">{status}</p>
      )}

      {predictionResult && (
        <div className="result-container">
          <h3>Prediction Result:</h3>
          <p><strong>Identified Person:</strong> {predictionResult.identified_person}</p>
          <p><strong>Reason:</strong> {predictionResult.reason}</p>
          <p><strong>Confidence:</strong> {predictionResult.prediction_details?.fused_confidence}</p>
          <p><strong>Time Taken:</strong> {predictionResult.prediction_time_seconds} seconds</p>
        </div>
      )}
    </div>
  );
};

export default App;
