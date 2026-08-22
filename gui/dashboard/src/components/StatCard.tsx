import React from 'react';

interface StatCardProps {
  label: string;
  value: string;
  icon: React.ReactElement;
  color?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  icon,
  color = "text-sky-400"
}) => {
  return (
    <div className="glass p-5 rounded-2xl border-b-2 border-transparent hover:border-sky-500 transition-all duration-300 group">
      <div className="flex justify-between items-center mb-3">
        <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">{label}</span>
        {React.cloneElement(icon, {
          className: `${color} w-4 h-4 group-hover:scale-110 transition-transform`
        } as any)}
      </div>
      <div className="text-2xl font-black text-white">{value}</div>
    </div>
  );
};
