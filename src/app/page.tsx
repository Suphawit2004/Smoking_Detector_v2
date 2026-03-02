"use client";

import React, { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

interface DetectionLog {
  id: number;
  time: string;
  confidence: number;
  image: string;
}

interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
}

const calculateIoU = (box1: BoundingBox, box2: BoundingBox) => {
  const xA = Math.max(box1.x1, box2.x1);
  const yA = Math.max(box1.y1, box2.y1);
  const xB = Math.min(box1.x2, box2.x2);
  const yB = Math.min(box1.y2, box2.y2);
  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  const box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  return interArea / (box1Area + box2Area - interArea);
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [modelSession, setModelSession] = useState<ort.InferenceSession | null>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [loadingText, setLoadingText] = useState("⏳ กำลังโหลดระบบ AI...");
  
  const [isSmoking, setIsSmoking] = useState(false);
  const [logs, setLogs] = useState<DetectionLog[]>([]);

  const requestRef = useRef<number>(0);
  const lastLogTimeRef = useRef<number>(0);

  useEffect(() => {
    const loadModelAndClasses = async () => {
      try {
        setLoadingText("⏳ กำลังโหลดโมเดล YOLO ONNX...");
        (ort.env.wasm as any).wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        const session = await ort.InferenceSession.create("/models/yolo_smoking.onnx", {
          executionProviders: ["wasm"],
        });
        setModelSession(session);
        setLoadingText("✨ ระบบพร้อมใช้งาน");
      } catch (error) {
        console.error("Error loading files:", error);
        setLoadingText("❌ เกิดข้อผิดพลาดในการโหลดโมเดล");
      }
    };
    loadModelAndClasses();
  }, []);

  const startCamera = async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setIsCameraReady(true);
          };
        }
      } catch (error) {
        console.error("Error accessing the camera:", error);
        alert("ไม่สามารถเข้าถึงกล้องได้ กรุณาอนุญาตการใช้งานกล้อง");
      }
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setIsCameraReady(false);
    setIsSmoking(false);
  };

  const playAlertSound = () => {
    try {
      const AudioContextClass = (window as any).AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioContextClass();
      const oscillator = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioCtx.destination);
      oscillator.type = "square";
      oscillator.frequency.value = 800;
      gainNode.gain.setValueAtTime(0.15, audioCtx.currentTime);
      oscillator.start();
      oscillator.stop(audioCtx.currentTime + 0.3);

      setTimeout(() => { if (audioCtx.state !== 'closed') audioCtx.close(); }, 500);
    } catch (e) { console.error("Failed to play sound", e); }
  };

  const detectSmoking = async () => {
    if (!modelSession || !videoRef.current || !canvasRef.current || !videoRef.current.srcObject) return;

    const inputSize = 640;
    const videoWidth = videoRef.current.videoWidth;
    const videoHeight = videoRef.current.videoHeight;

    if (videoWidth === 0 || videoHeight === 0) {
      requestRef.current = requestAnimationFrame(detectSmoking);
      return;
    }

    const offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = inputSize;
    offscreenCanvas.height = inputSize;
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
    const tensor = new ort.Tensor("float32", float32Data, [1, 3, inputSize, inputSize]);

    try {
      const results = await modelSession.run({ [modelSession.inputNames[0]]: tensor });
      const output = results[modelSession.outputNames[0]].data as Float32Array;
      
      const numCols = 8400; 
      let boxes: BoundingBox[] = [];

      for (let col = 0; col < numCols; col++) {
        const confidence = output[4 * numCols + col];
        if (confidence > 0.5) {
          const cx = output[0 * numCols + col];
          const cy = output[1 * numCols + col];
          const w = output[2 * numCols + col];
          const h = output[3 * numCols + col];

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
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;
        displayCtx.clearRect(0, 0, videoWidth, videoHeight);

        const scaleX = videoWidth / inputSize;
        const scaleY = videoHeight / inputSize;

        if (finalBoxes.length > 0) {
          const topBox = finalBoxes[0];
          setIsSmoking(topBox.confidence >= 0.65);
          
          finalBoxes.forEach(box => {
            const actualX = box.x1 * scaleX;
            const actualY = box.y1 * scaleY;
            const actualW = (box.x2 - box.x1) * scaleX;
            const actualH = (box.y2 - box.y1) * scaleY;

            // วาดกรอบสไตล์ล้ำยุค
            displayCtx.strokeStyle = box.confidence >= 0.65 ? "#ef4444" : "#f59e0b";
            displayCtx.lineWidth = 3;
            displayCtx.strokeRect(actualX, actualY, actualW, actualH);
            
            displayCtx.fillStyle = box.confidence >= 0.65 ? "#ef4444" : "#f59e0b";
            displayCtx.fillRect(actualX, actualY - 30, 150, 30);
            displayCtx.fillStyle = "white";
            displayCtx.font = "bold 16px sans-serif";
            displayCtx.fillText(`🚬 Smoke ${(box.confidence * 100).toFixed(0)}%`, actualX + 8, actualY - 8);
          });

          const now = Date.now();
          if (topBox.confidence >= 0.65 && now - lastLogTimeRef.current > 3000) {
            playAlertSound();
            const snapshotCanvas = document.createElement("canvas");
            snapshotCanvas.width = videoWidth;
            snapshotCanvas.height = videoHeight;
            snapshotCanvas.getContext("2d")?.drawImage(videoRef.current, 0, 0);
            const snapshot = snapshotCanvas.toDataURL("image/jpeg", 0.7);

            setLogs(prev => [{
              id: now,
              time: new Date().toLocaleTimeString('th-TH'),
              confidence: topBox.confidence,
              image: snapshot
            }, ...prev].slice(0, 8)); // โชว์ 8 รายการล่าสุด
            lastLogTimeRef.current = now;
          }
        } else {
          setIsSmoking(false);
        }
      }
    } catch (e) { console.error(e); }

    requestRef.current = requestAnimationFrame(detectSmoking);
  };

  useEffect(() => {
    if (isCameraReady && modelSession) {
      requestRef.current = requestAnimationFrame(detectSmoking);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isCameraReady, modelSession]);

  return (
    <main className={`min-h-screen p-4 md:p-8 font-sans transition-all duration-700 ease-in-out ${isSmoking ? 'bg-gradient-to-br from-red-950 via-red-900 to-black' : 'bg-gradient-to-br from-slate-50 to-slate-200'}`}>
      
      <div className="max-w-7xl mx-auto">
        {/* Header สไตล์ Glassmorphism */}
        <header className={`flex flex-col md:flex-row items-center justify-between mb-8 p-6 rounded-3xl backdrop-blur-md shadow-lg border transition-all duration-500 ${isSmoking ? 'bg-red-900/40 border-red-500/30' : 'bg-white/60 border-white/40'}`}>
          <div className="text-center md:text-left">
            <h1 className={`text-4xl font-extrabold tracking-tight ${isSmoking ? 'text-white' : 'text-slate-800'}`}>
              Smoking<span className={isSmoking ? 'text-red-400' : 'text-blue-600'}>Detector</span>
            </h1>
            <p className={`mt-2 font-medium ${isSmoking ? 'text-red-200' : 'text-slate-500'}`}>
              ระบบเฝ้าระวังและตรวจจับการสูบบุหรี่อัจฉริยะด้วย AI
            </p>
          </div>
          <div className="mt-4 md:mt-0 flex items-center gap-3">
            <span className="relative flex h-4 w-4">
              {isCameraReady && <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isSmoking ? 'bg-red-400' : 'bg-green-400'}`}></span>}
              <span className={`relative inline-flex rounded-full h-4 w-4 ${isCameraReady ? (isSmoking ? 'bg-red-500' : 'bg-green-500') : 'bg-slate-400'}`}></span>
            </span>
            <span className={`font-bold px-4 py-2 rounded-full shadow-sm transition-colors ${isCameraReady ? (isSmoking ? 'bg-red-500/20 text-red-100 border border-red-500/50' : 'bg-green-500/20 text-green-700 border border-green-500/30') : 'bg-slate-200 text-slate-500 border border-slate-300'}`}>
              {isCameraReady ? (isSmoking ? '⚠️ พบความผิดปกติ' : '🛡️ ระบบเฝ้าระวังทำงาน') : '⚪ ระบบออฟไลน์'}
            </span>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* ส่วนแสดงกล้องวิดีโอ */}
          <div className="lg:col-span-2 space-y-6">
            <div className={`relative w-full aspect-video bg-slate-900 rounded-[2rem] overflow-hidden shadow-2xl transition-all duration-500 ${isSmoking ? 'ring-8 ring-red-500/50 shadow-red-500/40' : 'ring-4 ring-white/50'}`}>
              
              {!isCameraReady && (
                <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-slate-900/80 backdrop-blur-sm">
                  <span className="text-6xl mb-4 opacity-50">📷</span>
                  <p className="text-slate-200 font-medium px-6 py-3 rounded-full bg-slate-800/80 border border-slate-600 shadow-inner">
                    {loadingText}
                  </p>
                </div>
              )}
              
              <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover" playsInline muted />
              <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-contain z-20 pointer-events-none" />
              
              {isSmoking && (
                <div className="absolute top-6 left-1/2 -translate-x-1/2 bg-red-600/90 backdrop-blur-md text-white px-8 py-3 rounded-full font-bold text-xl z-30 flex items-center gap-2 shadow-[0_0_40px_rgba(220,38,38,0.8)] animate-bounce border border-red-400">
                  🚨 ตรวจพบการสูบบุหรี่!
                </div>
              )}
            </div>

            {/* แผงควบคุม (Control Panel) */}
            <div className={`p-6 rounded-3xl shadow-lg backdrop-blur-md border transition-colors duration-500 flex justify-between items-center ${isSmoking ? 'bg-red-900/40 border-red-500/30' : 'bg-white/80 border-white/50'}`}>
              <div className="flex flex-col">
                <span className={`text-sm font-bold mb-1 ${isSmoking ? 'text-red-300' : 'text-slate-500'}`}>การควบคุมกล้อง</span>
                <span className={`text-xs ${isSmoking ? 'text-red-200/70' : 'text-slate-400'}`}>กดเปิดเพื่อเริ่มรันโมเดล</span>
              </div>
              
              {!isCameraReady ? (
                <button onClick={startCamera} disabled={!modelSession} className="flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-2xl shadow-xl hover:shadow-2xl transition-all hover:-translate-y-1">
                  ▶️ เปิดระบบกล้อง
                </button>
              ) : (
                <button onClick={stopCamera} className="flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700 text-white font-bold rounded-2xl shadow-xl hover:shadow-2xl transition-all hover:-translate-y-1">
                  ⏹️ ปิดระบบ
                </button>
              )}
            </div>
          </div>

          {/* ส่วนแสดงประวัติการตรวจพบ */}
          <div className={`rounded-[2rem] shadow-xl flex flex-col h-[600px] lg:h-auto overflow-hidden border backdrop-blur-md transition-colors duration-500 ${isSmoking ? 'bg-red-950/60 border-red-500/30' : 'bg-white/80 border-white/50'}`}>
            <div className={`p-6 border-b flex items-center justify-between ${isSmoking ? 'border-red-500/30 bg-red-900/30' : 'border-slate-200 bg-slate-50/50'}`}>
              <h2 className={`text-xl font-bold flex items-center gap-2 ${isSmoking ? 'text-white' : 'text-slate-800'}`}>
                📋 ประวัติการตรวจพบ
              </h2>
              <span className={`text-sm font-bold px-3 py-1 rounded-full ${isSmoking ? 'bg-red-500/20 text-red-200' : 'bg-slate-200 text-slate-600'}`}>
                {logs.length} รายการ
              </span>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {logs.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center opacity-60">
                  <span className="text-6xl mb-4">✨</span>
                  <p className={`font-medium ${isSmoking ? 'text-red-200' : 'text-slate-500'}`}>
                    ยังไม่พบผู้กระทำผิด<br/>พื้นที่อยู่ในความปลอดภัย
                  </p>
                </div>
              ) : (
                logs.map(log => (
                  <div key={log.id} className={`p-4 rounded-2xl flex gap-4 shadow-sm relative overflow-hidden group border transition-all hover:shadow-md ${isSmoking ? 'bg-red-900/40 border-red-500/30' : 'bg-white border-slate-100'}`}>
                    <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-red-500"></div>
                    <img src={log.image} alt="หลักฐาน" className="w-20 h-20 object-cover rounded-xl border-2 border-slate-200/20 group-hover:scale-105 transition-transform" />
                    <div className="flex flex-col justify-center flex-1">
                      <span className={`font-bold text-sm flex items-center gap-1.5 ${isSmoking ? 'text-red-300' : 'text-red-600'}`}>
                        🔥 พบการสูบบุหรี่
                      </span>
                      <span className={`text-xs mt-1 font-medium ${isSmoking ? 'text-slate-300' : 'text-slate-500'}`}>
                        เวลา: {log.time} น.
                      </span>
                      <div className="mt-3 w-full bg-slate-200/30 rounded-full h-2">
                        <div className="bg-gradient-to-r from-red-500 to-orange-500 h-2 rounded-full" style={{ width: `${log.confidence * 100}%` }}></div>
                      </div>
                      <span className={`text-[11px] text-right mt-1 font-medium ${isSmoking ? 'text-red-200/70' : 'text-slate-400'}`}>
                        ความมั่นใจ {(log.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}