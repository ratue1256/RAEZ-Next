import React from 'react';
import { RefreshCcw, CheckCircle2 } from 'lucide-react';
import { CheckpointInfo } from '../types';
import { formatBytes } from '../utils/metrics';

interface CheckpointsTabProps {
  checkpoints: CheckpointInfo[];
  fetchCheckpoints: () => void;
  loadModelStatus: string | null;
  handleLoadModel: (path: string) => void;
}

export const CheckpointsTab: React.FC<CheckpointsTabProps> = ({
  checkpoints,
  fetchCheckpoints,
  loadModelStatus,
  handleLoadModel
}) => {
  return (
    <div className="space-y-5">
      <div className="flex justify-between items-center">
        <h3 className="text-base font-bold">{checkpoints.length} Checkpoint(s) disponible(s)</h3>
        <button
          onClick={fetchCheckpoints}
          className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-xl text-sm transition-colors"
        >
          <RefreshCcw size={14} /> Rafraîchir
        </button>
      </div>

      {loadModelStatus && (
        <div className="p-3 bg-slate-900/80 border border-white/10 text-xs font-mono text-cyan-300 rounded-xl">
          {loadModelStatus}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {checkpoints.map((ckpt, i) => (
          <div
            key={i}
            className="glass p-5 rounded-2xl border-l-4 border-sky-500 hover:scale-[1.02] transition-transform flex flex-col justify-between"
          >
            <div>
              <div className="flex justify-between items-start mb-3">
                <CheckCircle2 className="text-emerald-400" size={18} />
                <span className="text-[10px] font-mono text-slate-400">{formatBytes(ckpt.size)}</span>
              </div>
              <h4 className="font-bold text-sm truncate mb-1" title={ckpt.name}>{ckpt.name}</h4>
              <p className="text-[10px] text-slate-500 truncate mb-4" title={ckpt.path}>{ckpt.path}</p>
            </div>
            <button
              onClick={() => handleLoadModel(ckpt.path)}
              className="w-full py-2 bg-slate-800 hover:bg-sky-500 hover:text-white rounded-lg text-xs font-bold transition-all border border-white/5 shadow-md"
            >
              LOAD WEIGHTS
            </button>
          </div>
        ))}
        {checkpoints.length === 0 && (
          <div className="col-span-3 text-center text-slate-500 italic py-16">
            Aucun checkpoint pour l'instant. Lance un entraînement !
          </div>
        )}
      </div>
    </div>
  );
};
