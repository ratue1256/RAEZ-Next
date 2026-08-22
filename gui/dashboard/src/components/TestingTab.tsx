import React from 'react';
import { Cpu, Zap, Camera, Image as ImageIcon, Database } from 'lucide-react';
import { InferenceResult } from '../types';

interface TestingTabProps {
  testMode: 'upload' | 'webcam';
  setTestMode: (m: 'upload' | 'webcam') => void;
  webcamActive: boolean;
  toggleWebcamTesting: () => void;
  stopAllWebcams: () => void;
  gpuBoost: boolean;
  setGpuBoost: (b: boolean) => void;
  inferenceMode: string;
  setInferenceMode: (m: string) => void;
  testImage: string | null;
  handleImageUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  testResults: InferenceResult | null;
  inferenceLoading: boolean;
  lastInferenceLatency: number | null;
  customFps: number;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}

export const TestingTab: React.FC<TestingTabProps> = ({
  testMode,
  setTestMode,
  webcamActive,
  toggleWebcamTesting,
  stopAllWebcams,
  gpuBoost,
  setGpuBoost,
  inferenceMode,
  setInferenceMode,
  testImage,
  handleImageUpload,
  testResults,
  inferenceLoading,
  lastInferenceLatency,
  customFps,
  videoRef,
  canvasRef
}) => {
  // Natural size of the uploaded image: model coords are relative to the full
  // frame, but the <img> uses object-contain (letterboxed) -> remap dot positions.
  const [naturalSize, setNaturalSize] = React.useState<{ w: number; h: number } | null>(null);

  const jointOverlayStyle = (kp: number[]): React.CSSProperties => {
    const fx = kp[0] / 256;
    const fy = kp[1] / 256;
    if (!naturalSize || naturalSize.w <= 0 || naturalSize.h <= 0) {
      return { left: `${fx * 100}%`, top: `${fy * 100}%` };
    }
    const { w, h } = naturalSize;
    const longSide = Math.max(w, h);
    return {
      left: `${50 + (fx - 0.5) * (w / longSide) * 100}%`,
      top: `${50 + (fy - 0.5) * (h / longSide) * 100}%`
    };
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
      <div className="xl:col-span-2 glass p-5 rounded-2xl flex flex-col items-center">
        <h3 className="text-base font-bold mb-4 flex items-center justify-between w-full">
          <span className="flex items-center gap-2 text-cyan-400">
            <Cpu size={18} /> Inférence Live (PyTorch / ONNX INT8)
          </span>
          {webcamActive && lastInferenceLatency && (
            <span className="text-xs font-mono text-cyan-300 bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded-lg">
              {lastInferenceLatency.toFixed(1)} ms ({customFps.toFixed(0)} FPS)
            </span>
          )}
        </h3>

        {/* Mode selector */}
        <div className="flex bg-slate-950 p-1 rounded-xl w-full max-w-sm mb-4 border border-white/5 text-xs">
          <button
            onClick={() => { stopAllWebcams(); setTestMode('upload'); }}
            className={`flex-1 py-2 rounded-lg font-bold transition-all ${testMode === 'upload' ? 'bg-sky-500 text-white' : 'text-slate-400 hover:text-white'}`}
          >
            📁 Fichier Image
          </button>
          <button
            onClick={() => { setTestMode('webcam'); }}
            className={`flex-1 py-2 rounded-lg font-bold transition-all ${testMode === 'webcam' ? 'bg-sky-500 text-white' : 'text-slate-400 hover:text-white'}`}
          >
            🎥 Caméra en direct
          </button>
        </div>

        {testMode === 'webcam' && (
          <div className="flex items-center justify-between px-3 py-2 bg-slate-900/40 rounded-xl border border-white/5 text-xs w-full max-w-sm mb-4">
            <span className="font-bold text-slate-300 flex items-center gap-1">
              <Zap className="text-yellow-400 w-3.5 h-3.5" /> Boost Inférence (Ultra low latency)
            </span>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={gpuBoost}
                onChange={(e) => setGpuBoost(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-9 h-5 bg-slate-800 rounded-full peer peer-focus:ring-2 peer-focus:ring-sky-500/20 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-slate-400 after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-sky-500"></div>
            </label>
          </div>
        )}

        <div className="flex flex-col gap-2 w-full max-w-sm mb-4 text-[10px]">
          <div className="flex justify-between items-center px-1 text-slate-400 font-bold uppercase tracking-wider">
            <span>Moteur d'inférence</span>
          </div>
          <div className="grid grid-cols-3 gap-1 bg-slate-950 p-1 rounded-xl border border-white/5">
            {(['pytorch', 'onnx_fp32', 'onnx_int8'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setInferenceMode(m)}
                className={`py-2 rounded-lg font-bold transition-all ${
                  inferenceMode === m
                    ? 'bg-sky-500 text-white shadow-md shadow-sky-500/10'
                    : 'text-slate-400 hover:text-white hover:bg-slate-900/40'
                }`}
              >
                {m === 'pytorch' ? 'PyTorch' : m === 'onnx_fp32' ? 'ONNX FP32' : 'ONNX INT8'}
              </button>
            ))}
          </div>
        </div>

        <div className="relative aspect-square w-full max-w-[400px] bg-slate-950 rounded-2xl overflow-hidden border border-white/10 shadow-2xl">
          {testMode === 'upload' ? (
            <>
              {testImage ? (
                <div className="relative w-full h-full">
                  <img
                    src={testImage}
                    className="w-full h-full object-contain"
                    alt="test"
                    onLoad={(e) => setNaturalSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                  />
                  {testResults && !testResults.error && testResults.keypoints?.map((kp: number[], i: number) => (
                    <div
                      key={i}
                      className="absolute w-2.5 h-2.5 bg-cyan-400 rounded-full -translate-x-1/2 -translate-y-1/2 shadow-[0_0_8px_rgba(34,211,238,0.9)] border border-white/30"
                      style={jointOverlayStyle(kp)}
                      title={`Joint ${i}`}
                    />
                  ))}
                </div>
              ) : (
                <label className="flex flex-col items-center justify-center h-full border-2 border-dashed border-slate-700 hover:border-sky-500 rounded-2xl cursor-pointer transition-all p-6">
                  <ImageIcon className="text-slate-500 mb-2" size={32} />
                  <span className="text-sm font-bold text-white">Sélectionner une image</span>
                  <span className="text-xs text-slate-400 mt-1">Glissez ou cliquez pour charger</span>
                  <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
                </label>
              )}
            </>
          ) : (
            <>
              <video
                ref={videoRef as any}
                className="hidden"
                width="320"
                height="320"
                playsInline
                muted
              />
              <canvas
                ref={canvasRef as any}
                className="w-full h-full object-cover"
                width="320"
                height="320"
              />
              {!webcamActive && (
                <div className="absolute inset-0 bg-slate-950/80 flex flex-col items-center justify-center text-center p-6 gap-3">
                  <Camera className="text-sky-400" size={32} />
                  <h4 className="font-bold text-white text-sm">Activer la caméra pour l'inférence</h4>
                  <p className="text-xs text-slate-400 max-w-xs">Calcule les joints 3D de la main en live avec contraintes biomécaniques.</p>
                  <button
                    onClick={toggleWebcamTesting}
                    className="px-4 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-lg text-xs font-bold transition-all shadow-lg"
                  >
                    Démarrer le flux live
                  </button>
                </div>
              )}
            </>
          )}

          {inferenceLoading && (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
              <div className="text-sky-400 font-bold animate-pulse text-xs">Exécution de l'inférence...</div>
            </div>
          )}
        </div>

        {testMode === 'webcam' && webcamActive && (
          <button
            onClick={toggleWebcamTesting}
            className="mt-4 px-4 py-2 bg-slate-800 hover:bg-red-500 hover:text-white rounded-xl text-xs font-bold transition-all"
          >
            Arrêter la caméra
          </button>
        )}
      </div>

      {/* Coordinates Inspector */}
      <div className="glass p-5 rounded-2xl flex flex-col">
        <h3 className="text-base font-bold mb-4 flex items-center gap-2">
          <Database className="text-sky-400" size={18} /> Coordonnées prédites (2D / 3D)
        </h3>
        {testResults && !testResults.error ? (
          <div className="flex-1 flex flex-col space-y-4">
            <div className="text-xs text-slate-400 font-mono flex justify-between bg-slate-950/60 p-3 rounded-xl border border-white/5">
              <span>Articulations : {testResults.keypoints?.length || 0} détectées</span>
              {testResults.latency_ms && <span className="text-cyan-400 font-bold">Latency : {testResults.latency_ms.toFixed(1)} ms</span>}
            </div>
            <div className="overflow-y-auto max-h-96 space-y-1 font-mono text-xs pr-1">
              {testResults.keypoints?.map((kp: number[], i: number) => {
                const kp3d = testResults.joints_3d?.[i] || [0, 0, 0];
                return (
                  <div key={i} className="text-slate-300 p-2 rounded-lg bg-slate-950/40 border border-white/5 space-y-1">
                    <div className="flex justify-between font-bold">
                      <span className="text-slate-400"># {i} Joint</span>
                    </div>
                    <div className="grid grid-cols-2 gap-1 text-[10px]">
                      <div>2D : x=<span className="text-cyan-400">{kp[0].toFixed(0)}</span> y=<span className="text-cyan-400">{kp[1].toFixed(0)}</span></div>
                      <div>3D : dx=<span className="text-indigo-400">{kp3d[0].toFixed(2)}</span> dy=<span className="text-indigo-400">{kp3d[1].toFixed(2)}</span> dz=<span className="text-indigo-400">{kp3d[2].toFixed(2)}</span></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-500 italic text-sm py-16">
            {testMode === 'upload' ? "Chargez une image pour voir les résultats." : "Activez le flux vidéo pour afficher les coordonnées."}
          </div>
        )}
      </div>
    </div>
  );
};
