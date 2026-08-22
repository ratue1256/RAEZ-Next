import React from 'react';
import { Cpu, Wifi, WifiOff } from 'lucide-react';
import { BackendStatus } from '../types';

interface HeaderProps {
  status: BackendStatus;
  backendOnline: boolean | null;
}

export const Header: React.FC<HeaderProps> = ({ status, backendOnline }) => {
  return (
    <header className="flex flex-wrap justify-between items-center mb-6 glass p-5 rounded-2xl shadow-2xl gap-4">
      <div>
        <h1 className="text-2xl lg:text-3xl font-black bg-gradient-to-r from-sky-400 to-indigo-500 bg-clip-text text-transparent flex items-center gap-3">
          RAEZ Hand Bone Tracker
        </h1>
        <p className="text-slate-400 text-xs mt-1 font-mono">
          Deep Learning Control Center · MobileNetV3 + 3D Biomechanical FK Layer
        </p>
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-2 px-3 py-2 bg-slate-800/60 rounded-xl border border-white/5 text-sm">
          <Cpu className="text-sky-400 w-4 h-4 animate-pulse" />
          <span className="font-mono">{status.device || "..."}</span>
        </div>

        <div className={`flex items-center gap-2 px-3 py-2 rounded-xl border text-sm font-bold
          ${backendOnline === null ? 'bg-slate-800/50 border-white/5 text-slate-400' :
            backendOnline ? 'bg-green-500/10 border-green-500/20 text-green-400' :
              'bg-red-500/10 border-red-500/20 text-red-400'}`}
        >
          {backendOnline ? <Wifi size={14} /> : <WifiOff size={14} />}
          {backendOnline === null ? "Connecting..." : backendOnline ? "Backend Online" : "Backend Offline"}
        </div>

        <div className={`flex items-center gap-2 px-3 py-2 rounded-xl border text-sm font-bold
          ${status.is_running
            ? 'bg-green-500/10 border-green-500/20 text-green-400'
            : 'bg-slate-800/50 border-white/5 text-slate-400'}`}
        >
          <div className={`w-2 h-2 rounded-full ${status.is_running ? 'bg-green-500 animate-pulse' : 'bg-slate-500'}`} />
          {status.is_running ? 'TRAINING' : 'IDLE'}
        </div>
      </div>
    </header>
  );
};
