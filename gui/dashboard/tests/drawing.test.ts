import { describe, it, expect, beforeEach } from 'vitest';
import { BONE_CONNECTIONS, drawSkeleton } from '../src/utils/drawing';

/** Minimal canvas 2D context stub that records draw calls. */
function makeCtxStub() {
  const calls = { stroke: 0, fill: 0 };
  return {
    calls,
    lineWidth: 0,
    strokeStyle: '',
    fillStyle: '',
    shadowBlur: 0,
    shadowColor: '',
    beginPath() {},
    moveTo() {},
    lineTo() {},
    arc() {},
    stroke() { calls.stroke += 1; },
    fill() { calls.fill += 1; }
  } as unknown as CanvasRenderingContext2D & { calls: { stroke: number; fill: number } };
}

const fullHand = (n: number) => Array.from({ length: n }, () => [10, 20]);

describe('BONE_CONNECTIONS', () => {
  it('defines 23 connections over the standard 21-joint topology', () => {
    expect(BONE_CONNECTIONS).toHaveLength(23);
    for (const [a, b] of BONE_CONNECTIONS) {
      expect(a).toBeLessThan(21);
      expect(b).toBeLessThan(21);
    }
  });
});

describe('drawSkeleton', () => {
  let ctx: ReturnType<typeof makeCtxStub>;

  beforeEach(() => {
    ctx = makeCtxStub();
  });

  it('draws one stroke per bone and one fill per joint', () => {
    drawSkeleton(ctx, fullHand(21), 320, 320);
    expect(ctx.calls.stroke).toBe(BONE_CONNECTIONS.length);
    expect(ctx.calls.fill).toBe(21);
  });

  it('ignores incomplete hands instead of throwing', () => {
    expect(() => drawSkeleton(ctx, [], 100, 100)).not.toThrow();
    expect(ctx.calls.stroke).toBe(0);
  });

  it('resets the shadow blur after drawing', () => {
    drawSkeleton(ctx, fullHand(21), 200, 200);
    expect(ctx.shadowBlur).toBe(0);
  });
});
