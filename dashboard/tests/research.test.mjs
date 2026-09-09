import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import {
  adjacentDate,
  concentrationColor,
  decodeMonth,
  niceScale,
  outlinePath,
  validDate,
} from '../lib/research.ts';

const read = (path) =>
  readFileSync(new URL(`../public/${path}`, import.meta.url));
const catalog = JSON.parse(read('data/catalog.json'));
const daily = JSON.parse(read('data/daily.json'));
const geometry = JSON.parse(read('data/geometry.json'));

test('calendar accepts leap day and rejects impossible or unsupported dates', () => {
  for (const date of ['2024-01-01', '2024-02-29', '2024-12-31'])
    assert.equal(validDate(date, catalog.start, catalog.end), true);
  for (const date of [
    '',
    '2024-02-30',
    '2024-04-31',
    '2025-01-01',
    '2023-12-31',
    '2024-2-1',
  ])
    assert.equal(validDate(date, catalog.start, catalog.end), false);
  assert.equal(adjacentDate('2024-02-28', 1), '2024-02-29');
  assert.equal(adjacentDate('2024-02-29', 1), '2024-03-01');
  assert.equal(adjacentDate('2024-03-01', -1), '2024-02-29');
});

test('colour scales preserve range, clamp values, and handle constant fields', () => {
  assert.deepEqual(niceScale(4.63, 7.23), [4.5, 7.5]);
  assert.deepEqual(niceScale(6, 6), [6, 6.5]);
  assert.equal(concentrationColor(-1, 0, 50), 'rgb(68,1,84)');
  assert.equal(concentrationColor(51, 0, 50), 'rgb(253,231,37)');
  assert.equal(concentrationColor(25, 0, 50), 'rgb(33,145,140)');
});

test('binary decoder rejects truncation, invalid day, NaN, and negative concentration', () => {
  assert.throws(() => decodeMonth(new ArrayBuffer(3), 1, 1, 1));
  assert.throws(() => decodeMonth(new ArrayBuffer(4), 1, 1, 2));
  assert.throws(() => decodeMonth(new ArrayBuffer(4), 1, 1, 0));
  for (const v of [NaN, -1, Infinity, 501]) {
    const data = new ArrayBuffer(4);
    new DataView(data).setFloat32(0, v, true);
    assert.throws(() => decodeMonth(data, 1, 1, 1));
  }
});

test('London geometry preserves the native projection and cell membership', () => {
  assert.equal(geometry.rows, 48);
  assert.equal(geometry.cols, 64);
  assert.equal(new Set(geometry.indices).size, 1719);
  assert.ok(
    geometry.indices.every((i) => Number.isInteger(i) && i >= 0 && i < 48 * 64),
  );
  assert.deepEqual(geometry.transform, [500000, 1000, 0, 202000, 0, -1000]);
  assert.equal(geometry.crs, 'EPSG:27700');
  assert.equal(
    outlinePath({ rows: 1, cols: 1, indices: [0] }),
    'M0,0h1M0,1h1M0,0v1M1,0v1',
  );
});

test('all 60 monthly files pass hashes and all 1,830 day/model maps match their statistics', () => {
  let maps = 0,
    chunks = 0;
  for (const model of catalog.models.filter((m) => m.hasMaps)) {
    assert.equal(Object.keys(catalog.chunks[model.id]).length, 12);
    for (const [month, chunk] of Object.entries(catalog.chunks[model.id])) {
      const bytes = read(chunk.url);
      assert.equal(bytes.byteLength, chunk.bytes);
      assert.equal(
        createHash('sha256').update(bytes).digest('hex'),
        chunk.sha256,
      );
      const buffer = bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      );
      for (let day = 1; day <= chunk.days; day++) {
        const date = `${month}-${String(day).padStart(2, '0')}`;
        assert.equal(validDate(date, catalog.start, catalog.end), true);
        const values = decodeMonth(buffer, chunk.days, catalog.cells, day);
        const sorted = [...values].sort((a, b) => a - b);
        const stats = daily[date].maps[model.id];
        assert.ok(Math.abs(sorted[0] - stats.min) < 0.00002);
        assert.ok(Math.abs(sorted.at(-1) - stats.max) < 0.00002);
        assert.ok(Math.abs(sorted[859] - stats.median) < 0.00002);
        maps++;
      }
      chunks++;
    }
  }
  assert.equal(chunks, 60);
  assert.equal(maps, 1830);
});

test('all seven scores match the existing published benchmark to four decimals', () => {
  const expected = [
    ['hybrid', 2.3405, 3.7765, 0.5179],
    ['hgb', 2.355, 3.792, 0.5139],
    ['hgb-monotonic', 2.4344, 3.8471, 0.4997],
    ['xgboost', 2.3712, 3.8739, 0.4927],
    ['ann', 2.5768, 4.0871, 0.4353],
    ['unet-ensemble', 2.4891, 4.1227, 0.4254],
    ['unet', 2.5963, 4.2816, 0.3803],
  ];
  assert.equal(catalog.models.length, 7);
  for (const [id, mae, rmse, r2] of expected) {
    const m = catalog.models.find((m) => m.id === id);
    assert.deepEqual(
      [m.metrics.mae, m.metrics.rmse, m.metrics.r2].map((v) => +v.toFixed(4)),
      [mae, rmse, r2],
    );
    assert.equal(m.metrics.n, 9619);
  }
  assert.equal(catalog.models.filter((m) => m.hasMaps).length, 5);
  assert.equal(catalog.chunks.ann, undefined);
  assert.equal(catalog.chunks.xgboost, undefined);
});

test('missing labels are not misrepresented as zero error', () => {
  assert.equal(Object.keys(daily).length, 366);
  assert.equal(
    Object.values(daily).reduce((sum, d) => sum + d.n, 0),
    9619,
  );
  assert.equal(Object.values(daily).filter((d) => d.n > 0).length, 349);
  assert.equal(daily['2024-07-15'].n, 24);
  assert.equal(+daily['2024-07-15'].scores.hybrid.rmse.toFixed(2), 1.9);
  for (let day = 15; day <= 31; day++) {
    const record = daily[`2024-12-${day}`];
    assert.equal(record.n, 0);
    assert.deepEqual(record.scores, {});
    assert.equal(Object.keys(record.maps).length, 5);
  }
});
