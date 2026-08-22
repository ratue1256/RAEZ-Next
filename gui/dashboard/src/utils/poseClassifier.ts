import { TargetPose, GuidedStep } from '../types';

export const TARGET_POSES: TargetPose[] = [
  { id: "middle_finger", label: "Doigt d'honneur (Middle Finger)", desc: "Levez uniquement le majeur, les autres doigts repliés dans la paume.", icon: "🖕" },
  { id: "fist", label: "Poing (Fist)", desc: "Fermez le poing. Tous les doigts repliés dans la paume.", icon: "✊" },
  { id: "open_hand", label: "Main ouverte (Open Hand)", desc: "Écartez bien les 5 doigts.", icon: "✋" },
  { id: "thumbs_up", label: "Pouce levé (Thumbs Up)", desc: "Fermez la main et pointez le pouce vers le haut.", icon: "👍" },
  { id: "peace", label: "Victoire (Peace)", desc: "Levez uniquement l'index et le majeur en forme de V.", icon: "✌️" },
  { id: "ok_sign", label: "Signe OK (OK Sign)", desc: "Touchez le pouce avec l'index pour former un cercle.", icon: "👌" },
];

export function getFingerStates(landmarks: any[]) {
  if (!landmarks || landmarks.length < 21) {
    return { thumb: false, index: false, middle: false, ring: false, pinky: false };
  }
  const indexExtended = landmarks[8].y < landmarks[6].y - 0.01;
  const middleExtended = landmarks[12].y < landmarks[10].y - 0.01;
  const ringExtended = landmarks[16].y < landmarks[14].y - 0.01;
  const pinkyExtended = landmarks[20].y < landmarks[18].y - 0.01;

  const dx = landmarks[4].x - landmarks[5].x;
  const dy = landmarks[4].y - landmarks[5].y;
  const thumbDist = Math.sqrt(dx * dx + dy * dy);

  const ix = landmarks[8].x - landmarks[5].x;
  const iy = landmarks[8].y - landmarks[5].y;
  const indexLength = Math.sqrt(ix * ix + iy * iy);

  const thumbExtended = thumbDist > indexLength * 0.65 && landmarks[4].y < landmarks[2].y + 0.08;

  return {
    thumb: thumbExtended,
    index: indexExtended,
    middle: middleExtended,
    ring: ringExtended,
    pinky: pinkyExtended
  };
}

export function classifyPose(landmarks: any[]): string | null {
  const { thumb, index, middle, ring, pinky } = getFingerStates(landmarks);

  // 1. Victory/Peace
  if (index && middle && !ring && !pinky) {
    return "peace";
  }
  // 2. Middle Finger
  if (!index && middle && !ring && !pinky) {
    return "middle_finger";
  }
  // 3. Fist
  if (!index && !middle && !ring && !pinky) {
    return "fist";
  }
  // 4. Open Hand
  if (index && middle && ring && pinky) {
    return "open_hand";
  }
  // 5. Thumbs Up
  if (thumb && !index && !middle && !ring && !pinky) {
    return "thumbs_up";
  }
  // 6. OK Sign
  const d48x = landmarks[4].x - landmarks[8].x;
  const d48y = landmarks[4].y - landmarks[8].y;
  const d48 = Math.sqrt(d48x * d48x + d48y * d48y);
  if (d48 < 0.06 && middle && ring && pinky) {
    return "ok_sign";
  }

  return null;
}

export function generateSteps(totalSteps: number, configuredHand: 'left' | 'right' | 'both'): GuidedStep[] {
  const steps: GuidedStep[] = [];
  const handOptions: ('left' | 'right' | 'both')[] = ['left', 'right', 'both'];

  // Step 1 is always Middle Finger
  steps.push({
    poseId: 'middle_finger',
    handType: configuredHand
  });

  for (let i = 1; i < totalSteps; i++) {
    const randomPose = TARGET_POSES[Math.floor(Math.random() * TARGET_POSES.length)];
    const targetHand = configuredHand === 'both'
      ? handOptions[Math.floor(Math.random() * handOptions.length)]
      : configuredHand;
    steps.push({
      poseId: randomPose.id,
      handType: targetHand
    });
  }

  return steps;
}
