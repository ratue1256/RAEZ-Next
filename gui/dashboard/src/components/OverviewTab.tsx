import React from 'react';
import { BarChart3, RefreshCcw, Award } from 'lucide-react';
import { StatCard } from './StatCard';
import { LeaderboardEntry } from '../types';
import { safeNum } from '../utils/metrics';

interface OverviewTabProps {
  latestTrainLoss: any;
  latestValLoss: any;
  bestMPJPE: number | null;
  leaderboardData: LeaderboardEntry[];
}

export const OverviewTab: React.FC<OverviewTabProps> = ({
  latestTrainLoss,
  latestValLoss,
  bestMPJPE,
  leaderboardData
}) => {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatCard
          label="Latest Train Loss"
          value={latestTrainLoss !== null ? safeNum(latestTrainLoss) : "—"}
          icon={<BarChart3 />}
        />
        <StatCard
          label="Latest Val Loss"
          value={latestValLoss !== null ? safeNum(latestValLoss) : "—"}
          icon={<RefreshCcw />}
          color="text-yellow-400"
        />
        <StatCard
          label="Best MPJPE (3D)"
          value={bestMPJPE !== null && bestMPJPE < 900 ? safeNum(bestMPJPE * 1000, 1) + " mm" : "—"}
          icon={<Award />}
          color="text-emerald-400"
        />
      </div>

      {/* Dynamic Global SOTA Leaderboard */}
      <div className="glass p-5 rounded-2xl">
        <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
          <Award className="text-yellow-400 animate-bounce" size={18} /> Classement Mondial des Modèles (Live Benchmark)
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-white/10 text-slate-400 font-mono">
                <th className="py-2.5">Rang</th>
                <th className="py-2.5">Modèle</th>
                <th className="py-2.5 text-right">MPJPE 3D (Erreur)</th>
                <th className="py-2.5 text-right">Temps d'inférence GPU</th>
                <th className="py-2.5 text-right">Inférence FPS</th>
                <th className="py-2.5 text-right">Paramètres</th>
              </tr>
            </thead>
            <tbody>
              {leaderboardData.map((row: LeaderboardEntry, idx: number) => (
                <tr
                  key={row.name}
                  className={`border-b border-white/5 transition-colors hover:bg-white/5
                    ${row.isCustom ? 'bg-sky-500/10 font-bold text-sky-300' : 'text-slate-300'}`}
                >
                  <td className="py-3 font-mono">
                    {idx === 0 ? "🥇 #1" : idx === 1 ? "🥈 #2" : idx === 2 ? "🥉 #3" : `   #${idx + 1}`}
                  </td>
                  <td className="py-3 flex items-center gap-2">
                    {row.name}
                    {row.isCustom && <span className="px-1.5 py-0.5 text-[9px] bg-sky-500 text-white rounded font-mono">VOUS</span>}
                  </td>
                  <td className="py-3 text-right font-mono text-emerald-400">{row.mpjpe.toFixed(1)} mm</td>
                  <td className="py-3 text-right font-mono">{row.latency.toFixed(1)} ms</td>
                  <td className="py-3 text-right font-mono">{row.fps.toFixed(0)} FPS</td>
                  <td className="py-3 text-right font-mono text-slate-500">{row.params}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
