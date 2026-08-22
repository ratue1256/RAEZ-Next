import { describe, it, expect } from 'vitest';
import { mergeMetricsByStep, downsampleData, safeNum, formatBytes } from '../src/utils/metrics';

describe('safeNum', () => {
  it('formats a number with default digits', () => {
    expect(safeNum(0.123456)).toBe('0.1235');
  });

  it('formats numeric strings', () => {
    expect(safeNum('1.25', 2)).toBe('1.25');
  });

  it('returns an em dash for invalid input', () => {
    expect(safeNum(undefined)).toBe('—');
    expect(safeNum('abc')).toBe('—');
    expect(safeNum(null)).toBe('—');
  });
});

describe('formatBytes', () => {
  it('renders kilobytes below 1 MB (binary units)', () => {
    expect(formatBytes(512 * 1024)).toBe('512 KB');
  });

  it('renders megabytes above 1 MB (binary units)', () => {
    expect(formatBytes(2.5 * 1024 * 1024)).toBe('2.5 MB');
  });
});

describe('mergeMetricsByStep', () => {
  it('returns an empty array for invalid input', () => {
    expect(mergeMetricsByStep([])).toEqual([]);
    expect(mergeMetricsByStep(null as any)).toEqual([]);
  });

  it('merges rows sharing the same step without losing values', () => {
    const merged = mergeMetricsByStep([
      { step: 1, epoch: 0, train_total: 0.5 },
      { step: 1, val_total: 0.4 },
      { step: 2, epoch: 1, train_total: 0.3 }
    ]);
    expect(merged).toHaveLength(2);
    expect(merged[0]).toMatchObject({ step: 1, epoch: 0, train_total: 0.5, val_total: 0.4 });
    expect(merged[1]).toMatchObject({ step: 2, epoch: 1 });
  });

  it('sorts by step and forward-fills the last known epoch', () => {
    const merged = mergeMetricsByStep([
      { step: 3 },
      { step: 1, epoch: '2' },
      { step: 2 }
    ]);
    expect(merged.map((r) => r.step)).toEqual([1, 2, 3]);
    expect(merged[1].epoch).toBe(2);
    expect(merged[2].epoch).toBe(2);
  });

  it('ignores null/undefined/empty-string values when merging', () => {
    const merged = mergeMetricsByStep([
      { step: 1, val_mpjpe_3d: 0.05 },
      { step: 1, val_mpjpe_3d: null }
    ]);
    expect(merged[0].val_mpjpe_3d).toBe(0.05);
  });
});

describe('downsampleData', () => {
  it('keeps small arrays untouched (same reference)', () => {
    const data = [{ step: 1 }, { step: 2 }];
    expect(downsampleData(data, 400)).toBe(data);
  });

  it('reduces to roughly maxPoints entries and keeps the last row', () => {
    const data = Array.from({ length: 1000 }, (_, i) => ({ step: i }));
    const sampled = downsampleData(data, 100);
    expect(sampled.length).toBeLessThanOrEqual(110);
    expect(sampled[sampled.length - 1].step).toBe(999);
    expect(sampled[0].step).toBe(0);
  });
});
