export interface BackendStatus {
  is_running: boolean;
  device?: string;
  project?: string;
  pid?: number | null;
}

export interface MetricRow {
  step: number;
  epoch?: number;
  train_total?: number;
  val_total?: number;
  val_mpjpe_3d?: number;
  val_confidence?: number;
  [key: string]: any;
}

export interface CheckpointInfo {
  name: string;
  path: string;
  size: number;
}

export interface TargetPose {
  id: string;
  label: string;
  desc: string;
  icon: string;
}

export interface GuidedStep {
  poseId: string;
  handType: 'left' | 'right' | 'both';
}

export interface LeaderboardEntry {
  name: string;
  mpjpe: number;
  latency: number;
  fps: number;
  params: string;
  isCustom: boolean;
}

export interface InferenceResult {
  keypoints?: number[][];
  joints_3d?: number[][];
  latency_ms?: number;
  error?: string;
}
