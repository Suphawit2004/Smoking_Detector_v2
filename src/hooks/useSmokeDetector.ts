import { useEffect, useRef, useState, useCallback } from "react";
import * as ort from "onnxruntime-web";

export interface DetectionLog {
  id: number;
  time: string;
  confidence: number;
  image: string;
}

interface BoundingBox {
  x1: number; y1: number; x2: number; y2: number; confidence: number;
}

const calculateIoU = (box1: BoundingBox, box2: BoundingBox) => {
  const xA = Math.max(box1.x1, box2.x1); const yA = Math.max(box1.y1, box2.y1);
  const xB = Math.min(box1.x2, box2.x2); const yB = Math.min(box1.y2, box2.y2);
  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  const box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  return interArea / (box1Area + box2Area - interArea);
};

const playAlertSound = () => {
  try {
    const AudioContextClass = (window as any).AudioContext || (window as any).webkitAudioContext;
    const audioCtx = new AudioContextClass();
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    oscillator.connect(gainNode); gainNode.connect(audioCtx.destination);
    oscillator.type = "square";
    oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.3);
    gainNode.gain.setValueAtTime(0.2, audioCtx.currentTime);
    oscillator.start(); oscillator.stop(audioCtx.currentTime + 0.3);
    setTimeout(() => { if (audioCtx.state !== 'closed') audioCtx.close(); }, 500);
  } catch (e) { console.error("Failed to play sound", e); }
};

export function useSmokeDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>(0);
  const lastLogTimeRef = useRef<number>(0);

  const [modelSession, setModelSession] = useState<ort.InferenceSession | null>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [loadingText, setLoadingText] = useState("⏳ INITIALIZING AI CORE...");
  const [isSmoking, setIsSmoking] = useState(false);
  const [logs, setLogs] = useState<DetectionLog[]>([]);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>("");

  useEffect(() => {
    const initSystem = async () => {
      try {
        setLoadingText("⏳ LOADING YOLO ONNX MODEL...");
        (ort.env.wasm as any).wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        const session = await ort.InferenceSession.create("/models/yolo_smoking.onnx", { executionProviders: ["wasm"] });
        setModelSession(session);
        setLoadingText("✨ SYSTEM READY");
      } catch (error) {
        console.error("Error loading files:", error);
        setLoadingText("❌ CRITICAL ERROR: MODEL LOAD FAILED");
      }

      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');
      setCameras(videoDevices);
      if (videoDevices.length > 0) setSelectedCameraId(videoDevices[0].deviceId);
    };
    initSystem();
  }, []);

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: selectedCameraId 
          ? { deviceId: { exact: selectedCameraId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsCameraReady(true);
          navigator.mediaDevices.enumerateDevices().then(devices => 
            setCameras(devices.filter(d => d.kind === 'videoinput'))
          );
        };
      }
    } catch (error) {
      alert("ACCESS DENIED: ไม่สามารถเข้าถึงกล้องได้ กรุณาตรวจสอบสิทธิ์");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    canvasRef.current?.getContext("2d")?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setIsCameraReady(false);
    setIsSmoking(false);
  };

  const clearLogs = () => {
    if (confirm("SYSTEM WARNING: คุณแน่ใจหรือไม่ว่าต้องการล้างประวัติทั้งหมด?")) {
      setLogs([]);
    }
  };

  const detectSmoking = useCallback(async () => {
    if (!modelSession || !videoRef.current || !canvasRef.current || !videoRef.current.srcObject) return;
    const inputSize = 640;
    const { videoWidth, videoHeight } = videoRef.current;

    if (videoWidth === 0 || videoHeight === 0) {
      requestRef.current = requestAnimationFrame(detectSmoking);
      return;
    }

    const offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = inputSize; offscreenCanvas.height = inputSize;
    const ctx = offscreenCanvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    ctx.drawImage(videoRef.current, 0, 0, inputSize, inputSize);
    const pixels = ctx.getImageData(0, 0, inputSize, inputSize).data;
    const float32Data = new Float32Array(3 * inputSize * inputSize);
    for (let i = 0; i < inputSize * inputSize; i++) {
      float32Data[i] = pixels[i * 4] / 255.0;
      float32Data[inputSize * inputSize + i] = pixels[i * 4 + 1] / 255.0;
      float32Data[2 * inputSize * inputSize + i] = pixels[i * 4 + 2] / 255.0;
    }

    try {
      const tensor = new ort.Tensor("float32", float32Data, [1, 3, inputSize, inputSize]);
      const results = await modelSession.run({ [modelSession.inputNames[0]]: tensor });
      const output = results[modelSession.outputNames[0]].data as Float32Array;
      
      const numCols = 8400; let boxes: BoundingBox[] = [];
      for (let col = 0; col < numCols; col++) {
        const confidence = output[4 * numCols + col];
        if (confidence > 0.5) {
          const cx = output[0 * numCols + col]; const cy = output[1 * numCols + col];
          const w = output[2 * numCols + col]; const h = output[3 * numCols + col];
          boxes.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, confidence });
        }
      }

      boxes.sort((a, b) => b.confidence - a.confidence);
      const finalBoxes: BoundingBox[] = [];
      while (boxes.length > 0) {
        const current = boxes.shift();
        if (current) {
          finalBoxes.push(current);
          boxes = boxes.filter(box => calculateIoU(current, box) < 0.45);
        }
      }

      const displayCtx = canvasRef.current.getContext("2d");
      if (displayCtx) {
        canvasRef.current.width = videoWidth; canvasRef.current.height = videoHeight;
        displayCtx.clearRect(0, 0, videoWidth, videoHeight);
        const scaleX = videoWidth / inputSize; const scaleY = videoHeight / inputSize;

        if (finalBoxes.length > 0) {
          const topBox = finalBoxes[0];
          setIsSmoking(topBox.confidence >= 0.65);
          
          finalBoxes.forEach(box => {
            const actualX = box.x1 * scaleX; const actualY = box.y1 * scaleY;
            const actualW = (box.x2 - box.x1) * scaleX; const actualH = (box.y2 - box.y1) * scaleY;
            
            displayCtx.strokeStyle = box.confidence >= 0.65 ? "#ef4444" : "#f59e0b";
            displayCtx.lineWidth = 3;
            displayCtx.strokeRect(actualX, actualY, actualW, actualH);
            
            const cornerLength = 20; displayCtx.lineWidth = 5; displayCtx.beginPath();
            displayCtx.moveTo(actualX, actualY + cornerLength); displayCtx.lineTo(actualX, actualY); displayCtx.lineTo(actualX + cornerLength, actualY);
            displayCtx.moveTo(actualX + actualW - cornerLength, actualY); displayCtx.lineTo(actualX + actualW, actualY); displayCtx.lineTo(actualX + actualW, actualY + cornerLength);
            displayCtx.moveTo(actualX, actualY + actualH - cornerLength); displayCtx.lineTo(actualX, actualY + actualH); displayCtx.lineTo(actualX + cornerLength, actualY + actualH);
            displayCtx.moveTo(actualX + actualW - cornerLength, actualY + actualH); displayCtx.lineTo(actualX + actualW, actualY + actualH); displayCtx.lineTo(actualX + actualW, actualY + actualH - cornerLength);
            displayCtx.stroke();

            displayCtx.fillStyle = box.confidence >= 0.65 ? "#ef4444" : "#f59e0b";
            displayCtx.fillRect(actualX, actualY - 30, 200, 30);
            displayCtx.fillStyle = "white"; displayCtx.font = "bold 16px monospace";
            displayCtx.fillText(`[!] SMOKE DETECTED: ${(box.confidence * 100).toFixed(0)}%`, actualX + 8, actualY - 8);
          });

          const now = Date.now();
          if (topBox.confidence >= 0.65 && now - lastLogTimeRef.current > 5000) {
            playAlertSound();
            const snapCanvas = document.createElement("canvas");
            snapCanvas.width = videoWidth; snapCanvas.height = videoHeight;
            const snapCtx = snapCanvas.getContext("2d");
            snapCtx?.drawImage(videoRef.current, 0, 0);
            if(snapCtx) {
              snapCtx.fillStyle = "rgba(220, 38, 38, 0.8)"; snapCtx.fillRect(10, 10, 250, 40);
              snapCtx.fillStyle = "white"; snapCtx.font = "bold 18px monospace"; snapCtx.fillText(`INCIDENT LOGGED`, 20, 37);
            }
            setLogs(prev => [{ id: now, time: new Date().toLocaleTimeString('th-TH'), confidence: topBox.confidence, image: snapCanvas.toDataURL("image/jpeg", 0.7) }, ...prev].slice(0, 20)); 
            lastLogTimeRef.current = now;
          }
        } else { setIsSmoking(false); }
      }
    } catch (e) { console.error(e); }
    requestRef.current = requestAnimationFrame(detectSmoking);
  }, [modelSession, selectedCameraId]); // dependencies

  useEffect(() => {
    if (isCameraReady && modelSession) requestRef.current = requestAnimationFrame(detectSmoking);
    return () => { if (requestRef.current) cancelAnimationFrame(requestRef.current); };
  }, [isCameraReady, modelSession, detectSmoking]);

  return {
    videoRef,
    canvasRef,
    isCameraReady,
    loadingText,
    isSmoking,
    logs,
    cameras,
    selectedCameraId,
    setSelectedCameraId,
    startCamera,
    stopCamera,
    clearLogs,
    isModelReady: !!modelSession
  };
}