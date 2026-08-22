import React from 'react';
import { Camera, Database, RefreshCcw, CheckCircle2, Square } from 'lucide-react';
import { TARGET_POSES } from '../utils/poseClassifier';
import { GuidedStep, TargetPose } from '../types';

interface MyDataTabProps {
  myDataActive: boolean;
  sessionActive: boolean;
  sessionComplete: boolean;
  currentStepIdx: number;
  sessionSteps: GuidedStep[];
  flashActive: boolean;
  currentHand: 'left' | 'right' | 'both';
  currentPose: TargetPose;
  detectedHand: string;
  detectedPose: string;
  detectedPoseId: string | null;
  transitionActive: boolean;
  transitionTimeLeft: number;
  nextStep: GuidedStep | null;
  nextPose: TargetPose | null;
  myHandDetected: boolean;
  lastCaptureTimeRef: React.RefObject<number>;
  autoCaptureProgress: number;
  mpLoading: boolean;
  targetHandType: 'left' | 'right' | 'both';
  setTargetHandType: (t: 'left' | 'right' | 'both') => void;
  totalSessionSteps: number;
  setTotalSessionSteps: (n: number) => void;
  startGuidedSession: () => void;
  stopAllWebcams: () => void;
  setSessionActive: (a: boolean) => void;
  setSessionComplete: (c: boolean) => void;
  setCurrentStepIdx: (i: number) => void;
  setMyDataPoseIdx: (i: number) => void;
  myDataPoseIdx: number;
  handleCaptureSample: () => void;
  captureStatus: string | null;
  collectedCounts: Record<string, number>;
  handleTabChange: (tab: string) => void;
  myDataVideoRef: React.RefObject<HTMLVideoElement | null>;
  myDataCanvasRef: React.RefObject<HTMLCanvasElement | null>;
}

export const MyDataTab: React.FC<MyDataTabProps> = ({
  myDataActive,
  sessionActive,
  sessionComplete,
  currentStepIdx,
  sessionSteps,
  flashActive,
  currentHand,
  currentPose,
  detectedHand,
  detectedPose,
  detectedPoseId,
  transitionActive,
  transitionTimeLeft,
  nextStep,
  nextPose,
  myHandDetected,
  lastCaptureTimeRef,
  autoCaptureProgress,
  mpLoading,
  targetHandType,
  setTargetHandType,
  totalSessionSteps,
  setTotalSessionSteps,
  startGuidedSession,
  stopAllWebcams,
  setSessionActive,
  setSessionComplete,
  setCurrentStepIdx,
  setMyDataPoseIdx,
  myDataPoseIdx,
  handleCaptureSample,
  captureStatus,
  collectedCounts,
  handleTabChange,
  myDataVideoRef,
  myDataCanvasRef
}) => {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
      {/* Webcam view */}
      <div className="xl:col-span-2 glass p-5 rounded-2xl flex flex-col items-center relative">
        <h3 className="text-base font-bold mb-4 flex items-center justify-between w-full">
          <span className="flex items-center gap-2 text-purple-400">
            <Camera size={18} /> Acquisition de Données en Direct (MediaPipe)
          </span>
          {sessionActive && (
            <span className="px-2.5 py-1 text-xs font-bold bg-purple-600 text-white rounded-lg font-mono animate-pulse">
              SESSION EN COURS · POSE {currentStepIdx + 1}/{sessionSteps.length}
            </span>
          )}
        </h3>

        <div className="relative aspect-square w-full max-w-[480px] bg-slate-950 rounded-2xl overflow-hidden border border-white/5 shadow-2xl">
          <video
            ref={myDataVideoRef as any}
            className="hidden"
            width="480"
            height="480"
            playsInline
            muted
          />
          <canvas
            ref={myDataCanvasRef as any}
            className="w-full h-full object-cover"
            width="480"
            height="480"
          />

          {/* Flash feedback */}
          {flashActive && (
            <div className="absolute inset-0 bg-white opacity-95 pointer-events-none z-50 transition-opacity duration-150 animate-flash" />
          )}

          {/* OVERLAY 1: Target Objective (Top-Left) */}
          {myDataActive && (
            <div className="absolute top-3 left-3 bg-slate-950/90 backdrop-blur-md border border-purple-500/30 p-3 rounded-2xl flex flex-col gap-1 max-w-[210px] shadow-2xl pointer-events-none z-10">
              <span className="text-[10px] font-black text-purple-400 tracking-wider font-mono uppercase">Main Attendue</span>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="text-3xl leading-none">
                  {currentHand === 'left' ? "🤚" : currentHand === 'right' ? "✋" : "👐"}
                </span>
                <span className="text-xs font-black text-white leading-tight uppercase tracking-wide">
                  {currentHand === 'left' ? "Main Gauche" : currentHand === 'right' ? "Main Droite" : "Gauche ou Droite"}
                </span>
              </div>
              <span className="text-[10px] text-slate-300 font-bold font-mono mt-1 pt-1 border-t border-white/5 flex items-center gap-1">
                Pose : {currentPose.icon} {currentPose.id === 'middle_finger' ? "Doigt d'honneur" : currentPose.label.split(" ")[0]}
              </span>
            </div>
          )}

          {/* OVERLAY 2: Detection Status (Top-Right) */}
          {myDataActive && (
            <div className="absolute top-3 right-3 bg-slate-950/90 backdrop-blur-md border border-cyan-500/30 p-3 rounded-2xl flex flex-col gap-1 max-w-[210px] shadow-2xl text-right pointer-events-none z-10">
              <span className="text-[10px] font-black text-cyan-400 tracking-wider font-mono uppercase">Détection</span>
              <div className="flex items-center justify-end gap-1.5 mt-0.5">
                <span className={`text-xs font-black leading-tight uppercase tracking-wide ${
                  detectedHand !== "Aucune" ? 'text-green-400 font-extrabold' : 'text-slate-500'
                }`}>
                  {detectedHand !== "Aucune" ? `${detectedHand}` : "Pas de main"}
                </span>
                <span className="text-xl leading-none">
                  {detectedHand === 'Gauche' ? '🤚' : detectedHand === 'Droite' ? '✋' : '❌'}
                </span>
              </div>
              <span className={`text-[10px] font-bold font-mono mt-1 pt-1 border-t border-white/5 ${
                detectedPoseId === currentPose.id ? 'text-green-400' : detectedPose !== 'Aucune' ? 'text-yellow-400' : 'text-slate-500'
              }`}>
                Pose : {detectedPose !== 'Aucune' ? (detectedPoseId === 'middle_finger' ? "🖕 D. d'honneur" : detectedPose.split(" ")[0]) : "Aucune"}
              </span>
            </div>
          )}

          {/* Floating Action Banner HUD (Center-Bottom) */}
          {myDataActive && (
            <div className="absolute bottom-4 left-4 right-4 bg-slate-950/90 border border-purple-500/20 p-4 rounded-2xl flex flex-col gap-2.5 backdrop-blur-md text-center shadow-2xl z-10 pointer-events-none transition-all">
              {(() => {
                if (sessionActive && transitionActive) {
                  const handLabel = nextStep
                    ? (nextStep.handType === 'left' ? 'Main Gauche 🤚' : nextStep.handType === 'right' ? 'Main Droite ✋' : 'Deux Mains 👐')
                    : '';
                  return (
                    <div className="flex flex-col items-center justify-center gap-1 py-1">
                      <span className="text-sm font-black text-emerald-400 flex items-center justify-center gap-2">
                        <CheckCircle2 size={16} className="text-emerald-400 animate-bounce" />
                        ✓ CAPTURÉ !
                      </span>
                      <span className="text-xs text-slate-300 font-semibold">
                        Suivant dans <span className="font-mono text-purple-400 font-bold">{transitionTimeLeft.toFixed(1)}s</span> :{" "}
                        {nextPose ? (
                          <span className="text-white font-extrabold bg-purple-500/20 border border-purple-500/30 px-2 py-0.5 rounded">
                            {nextPose.icon} {nextPose.label.split(" ")[0]} ({handLabel})
                          </span>
                        ) : "Fin de session"}
                      </span>
                    </div>
                  );
                }

                if (!myHandDetected) {
                  return (
                    <div className="flex flex-col items-center justify-center gap-1 py-1.5">
                      <span className="text-xs font-black text-slate-400 animate-pulse flex items-center justify-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-slate-500 animate-ping" />
                        ⚠️ Placez votre main en face de la caméra...
                      </span>
                      <span className="text-[9px] text-slate-500 uppercase tracking-widest font-mono font-bold">
                        En attente de détection
                      </span>
                    </div>
                  );
                }

                const poseMatches = (detectedPoseId === currentPose.id);
                const handMatches = (currentHand === 'both') ||
                                    (currentHand === 'left' && detectedHand === 'Gauche') ||
                                    (currentHand === 'right' && detectedHand === 'Droite');

                const now = Date.now();
                const inCooldown = !sessionActive && ((now - (lastCaptureTimeRef.current || 0)) < 1500);
                if (inCooldown) {
                  return (
                    <div className="flex flex-col items-center justify-center gap-1 py-1">
                      <span className="text-sm font-black text-emerald-400 flex items-center justify-center gap-2">
                        <CheckCircle2 size={16} className="text-emerald-400 animate-bounce" />
                        ✓ CAPTURÉ !
                      </span>
                      <span className="text-[10px] text-slate-300 font-semibold animate-pulse">
                        Tournez ou inclinez légèrement votre main...
                      </span>
                    </div>
                  );
                }

                if (poseMatches && !handMatches) {
                  return (
                    <div className="flex flex-col items-center justify-center gap-1 py-1 animate-pulse">
                      <span className="text-xs font-black text-rose-400 flex items-center justify-center gap-2">
                        ⚠️ MAUVAISE MAIN DÉTECTÉE
                      </span>
                      <span className="text-[10px] text-slate-300 font-medium">
                        Utilisez votre main <span className="font-extrabold text-yellow-400">{currentHand === 'left' ? 'GAUCHE 🤚' : 'DROITE ✋'}</span>
                      </span>
                    </div>
                  );
                }

                if (!poseMatches) {
                  const targetHandText = currentHand === 'left' ? 'MAIN GAUCHE 🤚' : currentHand === 'right' ? 'MAIN DROITE ✋' : 'MAIN GAUCHE ou DROITE 👐';
                  return (
                    <div className="flex flex-col items-center justify-center gap-1.5 py-1">
                      <span className="text-[10px] text-slate-400 uppercase tracking-widest font-black font-mono">Consigne de pose</span>
                      <span className="flex items-center justify-center flex-wrap gap-2 text-white text-xs font-extrabold bg-indigo-500/20 border border-indigo-500/40 px-3.5 py-1.5 rounded-xl">
                        👉 Faites : <span className="text-white bg-indigo-500 px-2 py-0.5 rounded-md font-black">{currentPose.label.split(" ")[0]} {currentPose.icon}</span>
                        <span className="text-[10px] opacity-75 font-mono">avec la {targetHandText}</span>
                      </span>
                    </div>
                  );
                }

                return (
                  <div className="flex flex-col gap-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-purple-300 font-black animate-pulse flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-purple-500 animate-ping" />
                        📸 ENREGISTREMENT EN COURS...
                      </span>
                      <span className="font-mono font-black text-purple-300 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">{autoCaptureProgress}%</span>
                    </div>
                    <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden border border-white/5 p-[1px]">
                      <div
                        className="bg-gradient-to-r from-purple-500 via-pink-500 to-rose-500 h-full rounded-full transition-all duration-75 shadow-[0_0_8px_rgba(236,72,153,0.5)]"
                        style={{ width: `${autoCaptureProgress}%` }}
                      />
                    </div>
                    <span className="text-[9px] text-purple-300 font-bold uppercase tracking-wider font-mono">Maintenez la pose immobile...</span>
                  </div>
                );
              })()}
            </div>
          )}

          {/* Startup setup / landing cover */}
          {!myDataActive && (
            <div className="absolute inset-0 bg-slate-950/95 flex flex-col items-center justify-center p-6 text-center gap-6 z-20">
              <Database className="text-purple-400 animate-bounce" size={48} />
              <div>
                <h4 className="font-black text-white text-lg">Acquisition Guidée "My Data"</h4>
                <p className="text-xs text-slate-400 max-w-xs mt-1.5 leading-relaxed">
                  Sélectionnez vos options de capture ci-dessous puis lancez la session. Le système enregistre vos images automatiquement sans clic.
                </p>
              </div>

              <div className="bg-purple-500/10 border border-purple-500/20 px-3.5 py-2.5 rounded-xl text-xs text-purple-300 font-medium max-w-xs shadow-inner">
                🖕 La session commencera par la pose : <span className="font-black text-white">Doigt d'honneur</span>
              </div>

              <div className="w-full max-w-xs p-4 bg-slate-900/60 border border-white/5 rounded-2xl text-left space-y-4 text-xs">
                <div className="space-y-1.5">
                  <label className="text-slate-400 font-semibold">Main ciblée pour l'acquisition :</label>
                  <div className="grid grid-cols-3 gap-1 bg-slate-950 p-1 rounded-xl">
                    {(['both', 'left', 'right'] as const).map(type => (
                      <button
                        key={type}
                        type="button"
                        onClick={() => setTargetHandType(type)}
                        className={`py-1.5 rounded-lg text-[10px] font-bold transition-all ${
                          targetHandType === type
                            ? 'bg-purple-600 text-white shadow-md'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        {type === 'both' ? 'Les deux 👐' : type === 'left' ? 'Gauche 🤚' : 'Droite ✋'}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between items-center">
                    <label className="text-slate-400 font-semibold">Nombre total de poses :</label>
                    <span className="font-bold text-purple-300 font-mono">{totalSessionSteps} poses</span>
                  </div>
                  <div className="grid grid-cols-4 gap-1">
                    {[20, 50, 100, 150].map(cnt => (
                      <button
                        key={cnt}
                        type="button"
                        onClick={() => setTotalSessionSteps(cnt)}
                        className={`py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                          totalSessionSteps === cnt
                            ? 'bg-purple-600 border border-purple-400 text-white shadow-md'
                            : 'bg-slate-950 border border-white/5 text-slate-400 hover:text-white'
                        }`}
                      >
                        {cnt}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <button
                onClick={startGuidedSession}
                disabled={mpLoading}
                className="w-full max-w-xs py-3.5 bg-purple-600 hover:bg-purple-500 disabled:bg-slate-700 text-white rounded-xl text-sm font-black transition-all shadow-lg shadow-purple-600/30 border border-purple-400 flex items-center justify-center gap-2"
              >
                {mpLoading ? (
                  <>
                    <RefreshCcw size={16} className="animate-spin" />
                    Démarrage de la caméra...
                  </>
                ) : (
                  <>
                    <Camera size={16} />
                    Activer la Caméra & Lancer
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {myDataActive && (
          <button
            onClick={stopAllWebcams}
            className="mt-4 px-4 py-2 bg-slate-800 hover:bg-red-500 hover:text-white rounded-xl text-xs font-bold transition-all"
          >
            Désactiver la caméra
          </button>
        )}
      </div>

      {/* Session Control and Pose List */}
      <div className="glass p-5 rounded-2xl flex flex-col justify-between min-h-[500px]">
        {sessionComplete ? (
          <div className="flex-1 flex flex-col justify-center items-center text-center p-6 space-y-5">
            <div className="text-6xl animate-bounce">🎉</div>
            <h3 className="text-xl font-black text-white">Session terminée !</h3>
            <p className="text-xs text-slate-400 leading-relaxed max-w-xs">
              Félicitations ! Vous avez capturé avec succès toutes les images pour vos poses personnalisées.
            </p>
            <div className="p-4 bg-slate-950/60 border border-white/5 rounded-xl text-xs font-mono text-purple-300 w-full text-left space-y-1">
              <div>• Mode : {targetHandType === 'left' ? "Main Gauche" : targetHandType === 'right' ? "Main Droite" : "Les Deux Mains"}</div>
              <div>• Total : {sessionSteps.length} images sauvegardées dans <code className="text-white bg-slate-800 px-1.5 py-0.5 rounded text-[10px]">data/raw/custom/</code></div>
            </div>
            <div className="flex gap-2 w-full">
              <button
                onClick={() => {
                  setSessionComplete(false);
                  setSessionActive(false);
                  setCurrentStepIdx(0);
                  setMyDataPoseIdx(0);
                }}
                className="flex-1 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold rounded-xl transition-all border border-white/5"
              >
                Recommencer
              </button>
              <button
                onClick={() => handleTabChange('training')}
                className="flex-1 py-2.5 bg-purple-600 hover:bg-purple-500 text-white text-xs font-bold rounded-xl transition-all shadow-lg shadow-purple-600/20"
              >
                Lancer l'entraînement
              </button>
            </div>
          </div>
        ) : (
          <div>
            <h3 className="text-base font-bold mb-4 flex items-center justify-between w-full">
              <span className="flex items-center gap-2 text-purple-400">
                <Database className="text-purple-400" size={18} /> Pose Demandée
              </span>
              {sessionActive && (
                <span className="text-xs font-bold bg-purple-600 text-white px-2 py-0.5 rounded font-mono">
                  POSE {currentStepIdx + 1}/{sessionSteps.length}
                </span>
              )}
            </h3>

            <div className="p-5 bg-purple-500/10 border border-purple-500/20 rounded-2xl text-center space-y-3 mb-5">
              <div className="text-5xl">{currentPose.icon}</div>
              <h4 className="text-lg font-black text-white">{currentPose.label}</h4>
              <p className="text-xs text-purple-200 leading-relaxed">{currentPose.desc}</p>

              {sessionActive && (
                <div className="pt-2 border-t border-purple-500/10 flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs text-slate-300">
                    <span>Progression :</span>
                    <span className="font-bold font-mono text-purple-300">
                      {currentStepIdx} / {sessionSteps.length}
                    </span>
                  </div>
                  <div className="w-full bg-slate-900/50 h-2 rounded-full overflow-hidden border border-purple-500/10">
                    <div
                      className="bg-purple-500 h-full transition-all duration-300"
                      style={{ width: `${(currentStepIdx / sessionSteps.length) * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {!sessionActive && (
              <div className="grid grid-cols-6 gap-1 mb-5">
                {TARGET_POSES.map((pose, idx) => (
                  <button
                    key={pose.id}
                    type="button"
                    onClick={() => setMyDataPoseIdx(idx)}
                    className={`py-2 text-lg rounded-lg border transition-all ${
                      myDataPoseIdx === idx
                        ? 'bg-purple-600 border-purple-400 text-white scale-105 shadow-md shadow-purple-600/20'
                        : 'bg-slate-900 border-white/5 hover:bg-slate-800 text-slate-400'
                    }`}
                    title={pose.label}
                  >
                    {pose.icon}
                  </button>
                ))}
              </div>
            )}

            <div className="space-y-3">
              {sessionActive ? (
                <button
                  onClick={() => setSessionActive(false)}
                  className="w-full py-3.5 bg-red-600 hover:bg-red-500 text-white font-bold rounded-xl flex items-center justify-center gap-2 text-sm transition-all border border-red-500 shadow-lg shadow-red-500/20"
                >
                  <Square size={16} fill="white" /> Suspendre la capture
                </button>
              ) : (
                <button
                  onClick={startGuidedSession}
                  className="w-full py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all text-sm bg-purple-600 hover:bg-purple-500 text-white shadow-lg shadow-purple-600/30 border border-purple-400"
                >
                  <Database size={16} /> Lancer la labellisation guidée
                </button>
              )}

              {!sessionActive && (
                <button
                  onClick={handleCaptureSample}
                  disabled={!myDataActive || !myHandDetected}
                  className={`w-full py-2.5 rounded-xl font-bold flex items-center justify-center gap-2 transition-all text-xs
                    ${(!myDataActive || !myHandDetected)
                      ? 'bg-slate-900/60 text-slate-600 cursor-not-allowed border border-white/5'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-white/10'}`}
                >
                  Capture manuelle unique
                </button>
              )}

              {captureStatus && (
                <div className="p-3 bg-slate-900/60 border border-white/5 rounded-xl text-center text-xs text-purple-300 font-mono leading-relaxed">
                  {captureStatus}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Collected Statistics */}
        <div className="border-t border-white/5 pt-4 space-y-2.5 mt-4">
          <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
            Statistiques du Dataset Personnel
          </h4>
          <div className="grid grid-cols-2 gap-1.5">
            {TARGET_POSES.map(pose => (
              <div
                key={pose.id}
                className={`flex justify-between items-center text-[10px] font-mono p-2 rounded-lg border transition-colors ${
                  !sessionComplete && currentPose.id === pose.id && sessionActive
                    ? 'bg-purple-500/10 border-purple-500/30 text-purple-300'
                    : 'bg-slate-950/40 border-white/5 text-slate-400'
                }`}
              >
                <span className="flex items-center gap-1.5 truncate">
                  <span>{pose.icon}</span>
                  <span className="truncate">{pose.id === "middle_finger" ? "D. d'honneur" : pose.label.split(" ")[0]}</span>
                </span>
                <span className="font-bold text-white shrink-0">{collectedCounts[pose.id] || 0}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
