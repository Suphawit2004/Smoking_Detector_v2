"use client";

import React, { useEffect, useState } from "react";
// นำเข้าระบบการทำงานจากไฟล์ที่เราแยกไว้ (ปรับ path ให้ตรงกับโฟลเดอร์ของคุณ)
import { useSmokeDetector, DetectionLog } from "../hooks/useSmokeDetector";

type Theme = "light" | "dark";

// Hook จัดการ Theme ย้ายมาไว้ฝั่ง UI เพราะเกี่ยวกับการแสดงผล
function useSystemTheme(): Theme {
  const [theme, setTheme] = useState<Theme>("light");
  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    setTheme(mediaQuery.matches ? "dark" : "light");
    const handleChange = (e: MediaQueryListEvent) => setTheme(e.matches ? "dark" : "light");
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);
  return theme;
}

export default function Home() {
  const theme = useSystemTheme();
  
  // 🔥 เรียกใช้ระบบทั้งหมดผ่าน Hook ตัวเดียว สั้นและสะอาดมาก!
  const system = useSmokeDetector();

  return (
    <main className={`min-h-screen p-4 md:p-8 font-sans transition-colors duration-500 ${theme === "dark" ? 'bg-slate-950 text-slate-300' : 'bg-slate-100 text-slate-800'}`}>
      <div className="max-w-7xl mx-auto">
        <SystemHeader isSmoking={system.isSmoking} isCameraReady={system.isCameraReady} theme={theme} />
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-6">
            <CameraMonitor theme={theme} system={system} />
            <ControlPanel theme={theme} system={system} />
          </div>
          <IncidentLogs logs={system.logs} theme={theme} onClearLogs={system.clearLogs} />
        </div>
      </div>
    </main>
  );
}

// ==========================================
// ส่วนประกอบของหน้าจอ (UI Components)
// ==========================================

const SystemHeader = ({ isSmoking, isCameraReady, theme }: { isSmoking: boolean, isCameraReady: boolean, theme: Theme }) => (
  <header className={`flex flex-col md:flex-row items-center justify-between mb-8 p-6 rounded-none border-l-8 transition-all shadow-md ${isSmoking ? 'border-l-red-600 bg-red-950/20' : (theme === "dark" ? 'border-l-slate-600 bg-slate-900 border-y border-r border-slate-800' : 'border-l-slate-400 bg-white border-y border-r border-slate-200')}`}>
    <div className="text-center md:text-left flex items-center gap-4">
      <span className={`text-5xl ${isSmoking ? 'text-red-500 animate-pulse' : 'text-slate-500'}`}>🚭</span>
      <div>
        <h1 className={`text-4xl font-extrabold tracking-tight uppercase ${theme === "dark" ? 'text-white' : 'text-slate-900'}`}>Smoke<span className="text-red-600">Guard</span></h1>
        <p className={`mt-1 font-mono text-sm tracking-widest uppercase ${theme === "dark" ? 'text-slate-400' : 'text-slate-500'}`}>AI Surveillance Command Center</p>
      </div>
    </div>
    <div className="mt-4 md:mt-0 flex items-center gap-4">
      <div className={`flex flex-col items-end font-mono text-xs uppercase ${theme === "dark" ? 'text-slate-500' : 'text-slate-400'}`}>
        <span>STATUS</span>
        <span className={`font-bold text-sm px-3 py-1 mt-1 border rounded-sm ${isCameraReady ? (isSmoking ? 'bg-red-950/50 text-red-500 border-red-500/50 animate-pulse' : 'bg-emerald-950/50 text-emerald-500 border-emerald-500/50') : 'bg-slate-800 text-slate-400 border-slate-700'}`}>
          {isCameraReady ? (isSmoking ? '🚨 CRITICAL: VIOLATION DETECTED' : '👁️ SYSTEM ACTIVE & MONITORING') : '⚪ SYSTEM OFFLINE'}
        </span>
      </div>
    </div>
  </header>
);

const CameraMonitor = ({ theme, system }: any) => (
  <div className={`relative w-full aspect-video bg-black rounded-none overflow-hidden transition-all duration-500 ${system.isSmoking ? 'border-2 border-red-500 shadow-[0_0_30px_rgba(239,68,68,0.4)]' : (theme === "dark" ? 'border border-slate-700 shadow-xl' : 'border border-slate-300 shadow-lg')}`}>
    <div className="absolute top-4 left-4 z-30 pointer-events-none font-mono text-white/80 text-xs drop-shadow-[0_1px_1px_rgba(0,0,0,1)] flex flex-col gap-1">
      <span>REC 🔴</span><span>CAM: {system.selectedCameraId ? 'EXTERNAL_01' : 'MAIN_FRONT'}</span><span>{new Date().toLocaleTimeString('en-US', { hour12: false })}</span>
    </div>
    {!system.isCameraReady && (
      <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-slate-900/95 backdrop-blur-sm border border-slate-800">
        <div className="relative flex items-center justify-center w-24 h-24 mb-6"><div className="absolute inset-0 border-4 border-red-500/30 rounded-full animate-ping"></div><span className="text-5xl">📡</span></div>
        <p className="text-red-500 font-mono tracking-widest uppercase text-sm mb-2 animate-pulse">NO SIGNAL / OFFLINE</p>
        <p className="text-slate-400 font-mono text-xs bg-slate-950 px-4 py-1 border border-slate-800">{system.loadingText}</p>
      </div>
    )}
    <video ref={system.videoRef} className="absolute inset-0 w-full h-full object-cover grayscale-[20%] contrast-125" playsInline muted />
    <canvas ref={system.canvasRef} className="absolute inset-0 w-full h-full object-contain z-20 pointer-events-none" />
    {system.isSmoking && <div className="absolute inset-0 z-10 pointer-events-none border-[8px] border-red-600/60 animate-pulse"></div>}
  </div>
);

const ControlPanel = ({ theme, system }: any) => (
  <div className={`p-6 rounded-none shadow-md border transition-colors flex flex-col md:flex-row justify-between items-start md:items-center gap-6 ${theme === "dark" ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
    <div className="flex flex-col flex-1 w-full">
      <span className={`font-mono text-xs tracking-wider uppercase mb-2 ${theme === "dark" ? 'text-slate-400' : 'text-slate-500'}`}>[ INPUT SOURCE CONFIGURATION ]</span>
      <div className="relative w-full max-w-sm">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><span className="text-slate-500">📹</span></div>
        <select 
          className={`block w-full pl-10 pr-10 py-3 text-sm font-medium rounded-none border-2 appearance-none cursor-pointer focus:outline-none ${theme === "dark" ? 'bg-slate-950 border-slate-700 text-slate-200 focus:border-red-500 disabled:bg-slate-900' : 'bg-slate-50 border-slate-300 text-slate-700 focus:border-red-500 disabled:bg-slate-100'} disabled:opacity-70`}
          value={system.selectedCameraId} onChange={(e) => system.setSelectedCameraId(e.target.value)} disabled={system.isCameraReady}
        >
          {system.cameras.length === 0 && <option>SCANNING FOR DEVICES...</option>}
          {system.cameras.map((c: any, i: number) => <option key={c.deviceId} value={c.deviceId}>{c.label ? c.label.toUpperCase() : `CAMERA DEVICE ${i + 1}`}</option>)}
        </select>
        <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none"><span className="text-xs text-slate-500">▼</span></div>
      </div>
    </div>
    {!system.isCameraReady ? (
      <button onClick={system.startCamera} disabled={!system.isModelReady} className="px-8 py-4 w-full md:w-auto bg-slate-800 hover:bg-slate-700 border border-slate-600 disabled:opacity-50 text-white font-mono uppercase text-sm rounded-none shadow-md transition-all">INITIALIZE CAMERA</button>
    ) : (
      <button onClick={system.stopCamera} className="px-8 py-4 w-full md:w-auto bg-red-700 hover:bg-red-800 border border-red-900 text-white font-mono uppercase text-sm rounded-none shadow-md transition-all">TERMINATE CONNECTION</button>
    )}
  </div>
);

const IncidentLogs = ({ logs, theme, onClearLogs }: { logs: DetectionLog[], theme: Theme, onClearLogs: () => void }) => (
  <div className={`shadow-md flex flex-col h-fit max-h-[700px] overflow-hidden border rounded-none ${theme === "dark" ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
    <div className={`p-4 border-b flex items-center justify-between font-mono uppercase text-sm ${theme === "dark" ? 'border-slate-800 bg-slate-950 text-slate-400' : 'border-slate-200 bg-slate-100 text-slate-600'}`}>
      <span className="flex items-center gap-2"><span className="text-red-500">▶</span> INCIDENT LOGS</span>
      <div className="flex items-center gap-2">
        <span className="bg-slate-800 text-slate-300 px-2 py-0.5 border border-slate-700">TOTAL: {logs.length}</span>
        {logs.length > 0 && (
          <button onClick={onClearLogs} className="bg-red-900/40 hover:bg-red-600 text-red-400 hover:text-white px-2 py-0.5 border border-red-800/50 hover:border-red-500 transition-colors text-xs" title="ล้างประวัติทั้งหมด">[ CLEAR ]</button>
        )}
      </div>
    </div>
    <div className="h-[580px] overflow-y-auto p-4 space-y-3 scrollbar-thin">
      {logs.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-full text-center opacity-30 font-mono text-sm">
          <span className="text-6xl mb-4 grayscale">🛡️</span><p>AREA SECURE<br/>NO VIOLATIONS DETECTED</p>
        </div>
      ) : (
        logs.map(log => (
          <div key={log.id} className={`p-3 flex gap-3 shadow-sm relative overflow-hidden border-l-4 rounded-none transition-all ${theme === "dark" ? 'bg-slate-950 border-l-red-600 border-y border-r border-slate-800 hover:bg-slate-800' : 'bg-white border-l-red-500 border-y border-r border-slate-200 hover:bg-slate-50'}`}>
            <img src={log.image} alt="Evidence" className="w-20 h-20 object-cover border border-slate-700 grayscale-[30%] contrast-125" />
            <div className="flex flex-col justify-center flex-1 font-mono">
              <span className="font-bold text-xs text-red-500 uppercase flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span> Rule Violation</span>
              <span className={`text-[10px] mt-1 ${theme === "dark" ? 'text-slate-400' : 'text-slate-500'}`}>T: {log.time}</span>
              <div className="mt-2 w-full bg-slate-800 border border-slate-700 h-1.5"><div className="bg-red-600 h-full" style={{ width: `${log.confidence * 100}%` }}></div></div>
              <span className="text-[10px] mt-1 text-slate-500 uppercase">CONF: {(log.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        ))
      )}
    </div>
  </div>
);