import { MetricRow } from '../types';

export function safeNum(val: any, digits = 4): string {
  const n = parseFloat(val);
  return isNaN(n) ? "—" : n.toFixed(digits);
}

export function formatBytes(bytes: number): string {
  if (bytes > 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

export function mergeMetricsByStep(rawMetrics: any[]): MetricRow[] {
  if (!Array.isArray(rawMetrics) || rawMetrics.length === 0) return [];
  const mergedMap = new Map<number, MetricRow>();

  for (const row of rawMetrics) {
    const step = typeof row.step === 'number' ? row.step : parseFloat(row.step);
    if (isNaN(step)) continue;
    if (!mergedMap.has(step)) {
      mergedMap.set(step, { step });
    }
    const existing = mergedMap.get(step)!;
    for (const [key, val] of Object.entries(row)) {
      if (val !== null && val !== undefined && val !== "") {
        existing[key] = val;
      }
    }
  }

  const sortedSteps = Array.from(mergedMap.values()).sort((a, b) => a.step - b.step);
  let lastEpoch = 0;
  for (const row of sortedSteps) {
    if (row.epoch !== undefined && row.epoch !== null && row.epoch !== "") {
      const epochNum = typeof row.epoch === 'number' ? row.epoch : parseFloat(row.epoch);
      if (!isNaN(epochNum)) {
        lastEpoch = epochNum;
      }
    }
    row.epoch = lastEpoch;
  }
  return sortedSteps;
}

export function downsampleData(data: MetricRow[], maxPoints = 400): MetricRow[] {
  if (!Array.isArray(data) || data.length <= maxPoints) return data;
  const factor = Math.ceil(data.length / maxPoints);
  const sampled: MetricRow[] = [];
  for (let i = 0; i < data.length; i += factor) {
    sampled.push(data[i]);
  }
  const lastIndex = data.length - 1;
  if (lastIndex % factor !== 0) {
    sampled.push(data[lastIndex]);
  }
  return sampled;
}
