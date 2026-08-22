export const BONE_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [0, 13], [13, 14], [14, 15], [15, 16],// Ring
  [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [5, 9], [9, 13], [13, 17]             // Palm Base
];

export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  keypoints: number[][],
  w: number,
  h: number
): void {
  if (!keypoints || keypoints.length < 21) return;

  // Draw bones
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#06b6d4'; // Cyan neon
  ctx.shadowBlur = 8;
  ctx.shadowColor = 'rgba(6, 182, 212, 0.8)';

  BONE_CONNECTIONS.forEach(([s, e]) => {
    const start = keypoints[s];
    const end = keypoints[e];
    if (start && end) {
      ctx.beginPath();
      ctx.moveTo((start[0] / 256) * w, (start[1] / 256) * h);
      ctx.lineTo((end[0] / 256) * w, (end[1] / 256) * h);
      ctx.stroke();
    }
  });

  // Draw joints
  ctx.fillStyle = '#10b981'; // Emerald neon
  ctx.shadowBlur = 10;
  ctx.shadowColor = 'rgba(16, 185, 129, 0.9)';
  keypoints.forEach((kp: number[]) => {
    ctx.beginPath();
    ctx.arc((kp[0] / 256) * w, (kp[1] / 256) * h, 5, 0, 2 * Math.PI);
    ctx.fill();
  });
  ctx.shadowBlur = 0;
}

export function drawMediaPipeSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: any[],
  w: number,
  h: number,
  mirror = true
): void {
  const getX = (lm: any) => mirror ? (1 - lm.x) * w : lm.x * w;
  const getY = (lm: any) => lm.y * h;

  ctx.lineWidth = 3;
  ctx.strokeStyle = '#a855f7'; // Purple neon
  ctx.shadowBlur = 6;
  ctx.shadowColor = 'rgba(168, 85, 247, 0.8)';

  BONE_CONNECTIONS.forEach(([s, e]) => {
    const start = landmarks[s];
    const end = landmarks[e];
    if (start && end) {
      ctx.beginPath();
      ctx.moveTo(getX(start), getY(start));
      ctx.lineTo(getX(end), getY(end));
      ctx.stroke();
    }
  });

  ctx.fillStyle = '#ec4899'; // Hot pink joints
  ctx.shadowBlur = 8;
  ctx.shadowColor = 'rgba(236, 72, 153, 0.9)';
  landmarks.forEach((lm: any) => {
    ctx.beginPath();
    ctx.arc(getX(lm), getY(lm), 4, 0, 2 * Math.PI);
    ctx.fill();
  });
  ctx.shadowBlur = 0;
}
