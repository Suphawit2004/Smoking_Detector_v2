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
  const [classes, setClasses] = useState<string[]>([]);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [loadingText, setLoadingText] = useState("กำลังโหลดข้อมูล...");
  
  const [isSmoking, setIsSmoking] = useState(false);
  const [logs, setLogs] = useState<DetectionLog[]>([]);

  const requestRef = useRef<number>();
  const lastLogTimeRef = useRef<number>(0);

  // 1. โหลดโมเดล
  useEffect(() => {
    const loadModelAndClasses = async () => {
      try {
        setLoadingText("กำลังโหลด classes.json...");
        const classRes = await fetch("/models/classes.json");
        const classData = await classRes.json();
        setClasses(classData);

        setLoadingText("กำลังโหลดโมเดล YOLO ONNX...");
        // บังคับ Type เป็น any เพื่อแก้บัค TypeScript
        (ort.env.wasm as any).wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        const session = await ort.InferenceSession.create("/models/yolo_smoking.onnx", {
          executionProviders: ["wasm"],
        });
        
        setModelSession(session);
        setLoadingText("โหลดโมเดลสำเร็จ พร้อมใช้งาน");
      } catch (error) {
        console.error("Error loading files:", error);
        setLoadingText("เกิดข้อผิดพลาดในการโหลดไฟล์ ตรวจสอบ Path อีกครั้ง");
      }
    };
    loadModelAndClasses();
  }, []);

  // 2. ฟังก์ชันเปิดกล้อง
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
        alert("ไม่สามารถเข้าถึงกล้องได้");
      }
    }
  };

  // 3. ฟังก์ชันปิดกล้อง
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    setIsCameraReady(false);
    setIsSmoking(false);
  };

  // 4. ฟังก์ชันเสียงแจ้งเตือน
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
      gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
      oscillator.start();
      oscillator.stop(audioCtx.currentTime + 0.3);

      setTimeout(() => {
        if (audioCtx.state !== 'closed') {
          audioCtx.close();
        }
      }, 500);
    } catch (e) {
      console.error("Failed to play sound", e);
    }
  };

  // 5. ระบบตรวจจับ
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

      // คัดกรองเบื้องต้นที่ 50% เพื่อตีกรอบ
      for (let col = 0; col < numCols; col++) {
        const confidence = output[4 * numCols + col];
        if (confidence > 0.5) {
          const cx = output[0 * numCols + col];
          const cy = output[1 * numCols + col];
          const w = output[2 * numCols + col];
          const h = output[3 * numCols + col];

          boxes.push({
            x1: cx - w / 2,
            y1: cy - h / 2,
            x2: cx + w / 2,
            y2: cy + h / 2,
            confidence: confidence
          });
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
          
          // เปลี่ยนสีจอและสถานะ ต่อเมื่อความมั่นใจ >= 65%
          setIsSmoking(topBox.confidence >= 0.65);
          
          finalBoxes.forEach(box => {
            const actualX = box.x1 * scaleX;
            const actualY = box.y1 * scaleY;
            const actualW = (box.x2 - box.x1) * scaleX;
            const actualH = (box.y2 - box.y1) * scaleY;

            displayCtx.strokeStyle = "#ef4444";
            displayCtx.lineWidth = 4;
            displayCtx.strokeRect(actualX, actualY, actualW, actualH);
            
            displayCtx.fillStyle = "#ef4444";
            displayCtx.fillRect(actualX, actualY - 30, 140, 30);
            displayCtx.fillStyle = "white";
            displayCtx.font = "bold 18px Arial";
            displayCtx.fillText(`Smoke ${(box.confidence * 100).toFixed(1)}%`, actualX + 5, actualY - 8);
          });

          // แจ้งเตือนเสียงและแคปภาพเมื่อมั่นใจ >= 65% เท่านั้น
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
            }, ...prev].slice(0, 5));
            lastLogTimeRef.current = now;
          }
        } else {
          setIsSmoking(false);
        }
      }
    } catch (e) {
      console.error(e);
    }

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
    <main className={`min-h-screen flex flex-col items-center py-10 transition-colors duration-300 ${isSmoking ? 'bg-red-900' : 'bg-gray-100'}`}>
      
      <header className="mb-6 text-center">
        <h1 className={`text-4xl font-bold mb-2 ${isSmoking ? 'text-white' : 'text-gray-800'}`}>
          Smoking Detector AI
        </h1>
        <p className={isSmoking ? 'text-red-200' : 'text-gray-500'}>ระบบตรวจจับและแจ้งเตือนการสูบบุหรี่เรียลไทม์</p>
      </header>

      <div className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-3 gap-6 px-4">
        
        <div className="lg:col-span-2 flex flex-col gap-4">
          <div className={`relative w-full aspect-video bg-black rounded-2xl overflow-hidden shadow-2xl border-4 transition-all ${isSmoking ? 'border-red-500 shadow-red-500/50 animate-pulse' : 'border-gray-300'}`}>
            {!isCameraReady && (
              <div className="absolute inset-0 flex items-center justify-center z-10">
                <p className="text-white text-lg bg-black/50 px-4 py-2 rounded-lg">{loadingText}</p>
              </div>
            )}
            
            <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover" playsInline muted />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-contain z-20 pointer-events-none" />
            
            {isSmoking && (
              <div className="absolute top-4 right-4 bg-red-600 text-white px-4 py-2 rounded-lg font-bold text-xl z-30 animate-bounce">
                ⚠️ ตรวจพบการสูบบุหรี่!
              </div>
            )}
          </div>

          <div className="flex justify-between items-center bg-white p-4 rounded-xl shadow">
            <div>
              <span className="block text-sm text-gray-500 font-medium mb-1">สถานะระบบ</span>
              <div className="flex items-center gap-2">
                <span className={`w-3 h-3 rounded-full ${isCameraReady ? 'bg-green-500' : 'bg-red-500'}`}></span>
                <span className="font-bold text-gray-700">{isCameraReady ? 'กล้องพร้อมใช้งาน' : 'กล้องปิดอยู่'}</span>
              </div>
            </div>
            
            {!isCameraReady ? (
              <button
                onClick={startCamera}
                disabled={!modelSession}
                className="px-6 py-3 bg-blue-600 disabled:bg-gray-400 hover:bg-blue-700 text-white font-bold rounded-lg shadow-md transition-all"
              >
                เปิดกล้อง 🎥
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-bold rounded-lg shadow-md transition-all"
              >
                ปิดกล้อง ⏹️
              </button>
            )}
            
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-5 flex flex-col h-[500px] lg:h-auto overflow-hidden">
          <h2 className="text-xl font-bold text-gray-800 mb-4 border-b pb-2">📋 ประวัติการตรวจพบ (ล่าสุด)</h2>
          
          <div className="flex-1 overflow-y-auto pr-2 space-y-4">
            {logs.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-400">
                <span className="text-4xl mb-2">✅</span>
                <p>ยังไม่พบการกระทำผิด</p>
              </div>
            ) : (
              logs.map(log => (
                <div key={log.id} className="bg-red-50 p-3 rounded-xl border border-red-100 flex gap-3 shadow-sm">
                  <img src={log.image} alt="หลักฐาน" className="w-20 h-20 object-cover rounded-lg border border-red-200" />
                  <div className="flex flex-col justify-center">
                    <span className="text-red-700 font-bold text-sm">เวลา: {log.time} น.</span>
                    <span className="text-gray-600 text-xs mt-1">
                      ความมั่นใจ: <span className="font-semibold text-gray-800">{(log.confidence * 100).toFixed(1)}%</span>
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

      </div>
    </main>
  );
}