export type Metric = 'rmse' | 'mae' | 'r2';
export type Model = {
  id: string;
  label: string;
  detail: string;
  color: string;
  hasMaps: boolean;
  metrics: Record<string, number>;
};
export type Catalog = {
  start: string;
  end: string;
  defaultDate: string;
  labelEnd: string;
  days: number;
  labelledDays: number;
  stationDays: number;
  cells: number;
  models: Model[];
  chunks: Record<
    string,
    Record<string, { url: string; days: number; bytes: number; sha256: string }>
  >;
  limitations: string[];
  mapMetricsNote: string;
};
export type Geometry = {
  rows: number;
  cols: number;
  indices: number[];
  transform: number[];
  crs: string;
};
export type GridStats = {
  min: number;
  max: number;
  mean: number;
  median: number;
  p90: number;
};
export type DayRecord = {
  n: number;
  scores: Record<string, { mae: number; rmse: number }>;
  maps: Record<string, GridStats>;
};
export type Daily = Record<string, DayRecord>;

export function validDate(value: string, start: string, end: string) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value) || value < start || value > end)
    return false;
  const date = new Date(`${value}T12:00:00Z`);
  return (
    Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === value
  );
}
export function adjacentDate(value: string, offset: number) {
  const date = new Date(`${value}T12:00:00Z`);
  date.setUTCDate(date.getUTCDate() + offset);
  return date.toISOString().slice(0, 10);
}
export function prettyDate(value: string) {
  return new Intl.DateTimeFormat('en-GB', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(`${value}T12:00:00Z`));
}
export function niceScale(min: number, max: number): [number, number] {
  const step = max - min < 5 ? 0.5 : max - min < 20 ? 2 : 5;
  const lo = Math.floor(min / step) * step;
  return [lo, Math.max(lo + step, Math.ceil(max / step) * step)];
}
export function concentrationColor(value: number, low: number, high: number) {
  const stops = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ];
  const t = Math.max(0, Math.min(1, (value - low) / (high - low || 1))) * 4;
  const i = Math.min(3, Math.floor(t));
  return `rgb(${stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * (t - i))).join(',')})`;
}
export function decodeMonth(
  buffer: ArrayBuffer,
  days: number,
  cells: number,
  day: number,
) {
  if (
    !Number.isInteger(day) ||
    day < 1 ||
    day > days ||
    buffer.byteLength !== days * cells * 4
  )
    throw new Error('The map archive is incomplete or the day is invalid.');
  const view = new DataView(buffer);
  return Float32Array.from({ length: cells }, (_, i) => {
    const value = view.getFloat32(((day - 1) * cells + i) * 4, true);
    if (!Number.isFinite(value) || value < 0 || value > 500)
      throw new Error('The map contains an invalid prediction.');
    return value;
  });
}
export function outlinePath(geometry: Geometry) {
  const cells = new Set(geometry.indices);
  return geometry.indices
    .map((i) => {
      const x = i % geometry.cols,
        y = Math.floor(i / geometry.cols);
      return [
        !cells.has(i - geometry.cols) ? `M${x},${y}h1` : '',
        !cells.has(i + geometry.cols) ? `M${x},${y + 1}h1` : '',
        x === 0 || !cells.has(i - 1) ? `M${x},${y}v1` : '',
        x === geometry.cols - 1 || !cells.has(i + 1) ? `M${x + 1},${y}v1` : '',
      ].join('');
    })
    .join('');
}
