import React from 'react';
import { Play, Square, Terminal as TerminalIcon } from 'lucide-react';
import { BackendStatus } from '../types';

interface TrainingTabProps {
  stage: number;
  setStage: (s: number) => void;
  resumeTraining: boolean;
  setResumeTraining: (r: boolean) => void;
  status: BackendStatus;
  backendOnline: boolean | null;
  startTraining: () => void;
  stopTraining: () => void;
  logs: string[];
  logEndRef: React.RefObject<HTMLDivElement | null>;
}

export const TrainingTab: React.FC<TrainingTabProps> = ({
  stage,
  setStage,
  resumeTraining,
  setResumeTraining,
  status,
  backendOnline,
  startTraining,
  stopTraining,
  logs,
  logEndRef
}) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
      <div className="glass p-6 rounded-2xl flex flex-col gap-5">
        <h3 className="text-lg font-bold">Training Control</h3>

        <div className="flex gap-2">
          {[1, 2, 3].map((s) => (
            <button
              key={s}
              onClick={() => setStage(s)}
              className={`flex-1 py-2 rounded-xl text-sm font-bold transition-all ${
                stage === s ? 'bg-sky-500 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              Stage {s}
            </button>
          ))}
        </div>

        <div className="flex items-center justify-between px-3 py-3 bg-slate-900/40 rounded-xl border border-white/5">
          <span className="text-sm font-bold text-slate-300">Reprendre l'entraînement</span>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={resumeTraining}
              onChange={(e) => setResumeTraining(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-slate-800 rounded-full peer peer-focus:ring-2 peer-focus:ring-sky-500/20 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-slate-400 after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-sky-500"></div>
          </label>
        </div>

        <button
          onClick={status.is_running ? stopTraining : startTraining}
          disabled={backendOnline === false}
          className={`w-full py-4 rounded-xl flex items-center justify-center gap-3 font-bold text-base transition-all duration-200
            ${backendOnline === false ? 'bg-slate-700 opacity-50 cursor-not-allowed' :
              status.is_running
                ? 'bg-red-500 hover:bg-red-600 shadow-lg shadow-red-500/20'
                : 'bg-sky-500 hover:bg-sky-400 shadow-lg shadow-sky-500/20'
            }`}
        >
          {status.is_running ? <Square size={18} fill="white" /> : <Play size={18} fill="white" />}
          {status.is_running ? 'STOP TRAINING' : `START STAGE ${stage}`}
        </button>

        {backendOnline === false && (
          <p className="text-xs text-red-400 text-center">
            ⚠️ Backend hors ligne — Lance <code className="bg-slate-800 px-1 rounded">run_gui.bat</code>
          </p>
        )}

        <div className="p-4 rounded-xl bg-slate-900/60 border border-white/5 space-y-2 text-sm">
          {[
            ["Backbone", "MobileNetV3-Large"],
            ["Channels (feat)", "960 (global pool)"],
            ["Heads", "Heatmap (21x64x64) + Biomechanical 3D FK Layer"],
            ["Precision", "FP16 Mixed (AMP)"],
            ["Logger", "CSVLogger → logs/"],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between">
              <span className="text-slate-400">{k}</span>
              <span className="text-sky-400 font-mono text-xs">{v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Live Console Output */}
      <div className="glass rounded-2xl flex flex-col overflow-hidden" style={{ height: 560 }}>
        <div className="flex items-center gap-2 p-3 border-b border-white/5 text-slate-400 bg-slate-950/40">
          <TerminalIcon size={14} />
          <span className="text-xs font-mono uppercase tracking-widest">Live Output</span>
          <span className="ml-auto text-xs opacity-50">{logs.length} lines</span>
        </div>
        <div className="flex-1 overflow-y-auto font-mono text-[11px] p-3 space-y-0.5">
          {logs.map((log, i) => (
            <div
              key={i}
              className={`pl-2 border-l-2 leading-relaxed
                ${log.startsWith('❌') ? 'border-red-500 text-red-300' :
                  log.startsWith('✅') ? 'border-green-500 text-green-300' :
                  log.startsWith('[SYSTEM]') ? 'border-sky-700 text-slate-400 italic' :
                  'border-slate-800 text-slate-300'}`}
            >
              {log}
            </div>
          ))}
          <div ref={logEndRef as any} />
        </div>
      </div>
    </div>
  );
};
