import React from 'react';
import { Activity, Zap, Database, Camera, Layers } from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'overview', label: 'Overview', icon: <Activity size={18} /> },
    { id: 'training', label: 'Training', icon: <Zap size={18} /> },
    { id: 'mydata', label: 'My Data', icon: <Database size={18} /> },
    { id: 'testing', label: 'Testing', icon: <Camera size={18} /> },
    { id: 'checkpoints', label: 'Models', icon: <Layers size={18} /> },
  ];

  return (
    <nav className="col-span-12 lg:col-span-2 flex lg:flex-col gap-2">
      {tabs.map((t) => {
        const active = activeTab === t.id;
        return (
          <button
            key={t.id}
            onClick={() => onTabChange(t.id)}
            className={`flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all duration-200 w-full text-left
              ${active
                ? 'bg-sky-500 text-white shadow-lg shadow-sky-500/30'
                : 'text-slate-400 hover:bg-white/5 hover:text-white'
              }`}
          >
            {t.icon}
            <span className="text-sm">{t.label}</span>
          </button>
        );
      })}
    </nav>
  );
};
