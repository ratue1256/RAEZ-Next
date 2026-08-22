import { describe, it, expect } from 'vitest';
import { TARGET_POSES, classifyPose, generateSteps } from '../src/utils/poseClassifier';

type Landmark = { x: number; y: number; z?: number };

/** Build a 21-point hand. Fingers listed as extended stay straight up (y decreases). */
function makeHand(extended: Array<'thumb' | 'index' | 'middle' | 'ring' | 'pinky'>): Landmark[] {
  const lm: Landmark[] = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5 }));
  // Wrist
  lm[0] = { x: 0.5, y: 0.9 };
  const tipIdx: Record<string, number> = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
  const pipIdx: Record<string, number> = { thumb: 2, index: 6, middle: 10, ring: 14, pinky: 18 };

  for (const finger of ['thumb', 'index', 'middle', 'ring', 'pinky'] as const) {
    if (extended.includes(finger)) {
      lm[tipIdx[finger]] = { x: 0.5, y: 0.2 }; // far above the pip -> extended
    } else {
      lm[tipIdx[finger]] = { x: 0.5, y: 0.5 }; // level with pip -> folded
    }
    lm[pipIdx[finger]] = { x: 0.5, y: 0.45 };
  }

  // Thumb heuristics in getFingerStates use distance to index_mcp (landmark 5).
  lm[5] = { x: 0.55, y: 0.6 };
  lm[2] = { x: 0.45, y: 0.7 };
  return lm;
}

describe('TARGET_POSES', () => {
  it('contains exactly the six tracked poses', () => {
    expect(TARGET_POSES.map((p) => p.id)).toEqual([
      'middle_finger', 'fist', 'open_hand', 'thumbs_up', 'peace', 'ok_sign'
    ]);
  });
});

describe('classifyPose', () => {
  it('detects an open hand', () => {
    expect(classifyPose(makeHand(['index', 'middle', 'ring', 'pinky']))).toBe('open_hand');
  });

  it('detects a fist', () => {
    expect(classifyPose(makeHand([]))).toBe('fist');
  });

  it('detects peace when index + middle are up', () => {
    expect(classifyPose(makeHand(['index', 'middle']))).toBe('peace');
  });

  it('detects thumbs up when only the thumb is up', () => {
    expect(classifyPose(makeHand(['thumb']))).toBe('thumbs_up');
  });

  it('returns null for unmatched configurations', () => {
    expect(classifyPose(makeHand(['index']))).toBeNull();
  });

  it('returns null for empty or short landmark arrays', () => {
    expect(classifyPose([])).toBeNull();
  });
});

describe('generateSteps', () => {
  it('always starts with middle_finger and has the requested length', () => {
    const steps = generateSteps(20, 'both');
    expect(steps).toHaveLength(20);
    expect(steps[0].poseId).toBe('middle_finger');
  });

  it('respects a fixed hand configuration', () => {
    for (const step of generateSteps(15, 'left')) {
      expect(step.handType).toBe('left');
    }
  });

  it('only uses known poses and valid hands', () => {
    const ids = new Set(TARGET_POSES.map((p) => p.id));
    for (const step of generateSteps(30, 'both')) {
      expect(ids.has(step.poseId)).toBe(true);
      expect(['left', 'right', 'both']).toContain(step.handType);
    }
  });
});
