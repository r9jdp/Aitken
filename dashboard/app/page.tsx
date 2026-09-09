'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import {
  ArrowDownToLine,
  ArrowUpRight,
  BarChart3,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Layers3,
  Map as MapIcon,
  Waves,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  adjacentDate,
  concentrationColor,
  decodeMonth,
  niceScale,
  outlinePath,
  prettyDate,
  validDate,
} from '@/lib/research';
import type { Catalog, Daily, Geometry, Metric, Model } from '@/lib/research';
import { registerDashboardTools, type ModelContext } from '@/lib/webmcp';

type Archive = { catalog: Catalog; daily: Daily; geometry: Geometry };
type Loaded = { model: string; date: string; values: Float32Array };
const fmt = (v: number | undefined, digits = 2) =>
  v === undefined ? '—' : v.toFixed(digits);
const github = 'https://github.com/r9jdp/Aitken';

async function getJson<T>(url: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok)
    throw new Error(
      `Could not load the research archive (${response.status}).`,
    );
  return response.json();
}

export default function Home() {
  const [archive, setArchive] = useState<Archive>();
  const [initialError, setInitialError] = useState('');
  const [attempt, setAttempt] = useState(0);
  const [tab, setTab] = useState('explorer');
  const [draftModel, setDraftModel] = useState('hybrid');
  const [draftDate, setDraftDate] = useState('2024-07-15');
  const [loaded, setLoaded] = useState<Loaded>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [fixedScale, setFixedScale] = useState(false);
  const [hotspots, setHotspots] = useState(false);
  const [hover, setHover] = useState<number>();
  const [exporting, setExporting] = useState(false);
  const cache = useRef(new Map<string, ArrayBuffer>());
  const request = useRef<AbortController | null>(null);
  const mapRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      getJson<Catalog>('/data/catalog.json', controller.signal),
      getJson<Daily>('/data/daily.json', controller.signal),
      getJson<Geometry>('/data/geometry.json', controller.signal),
    ])
      .then(([catalog, daily, geometry]) =>
        setArchive({ catalog, daily, geometry }),
      )
      .catch((e) => {
        if (e.name !== 'AbortError') setInitialError(e.message);
      });
    return () => controller.abort();
  }, [attempt]);

  const generate = useCallback(
    async (model: string, date: string) => {
      if (!archive) return;
      if (!validDate(date, archive.catalog.start, archive.catalog.end)) {
        setError('Choose a valid date from 1 January to 31 December 2024.');
        return;
      }
      const chunk = archive.catalog.chunks[model]?.[date.slice(0, 7)];
      if (!chunk) {
        setError(
          'This model has no saved full-grid heatmap. Its scores are available in Model benchmark.',
        );
        return;
      }
      request.current?.abort();
      const controller = new AbortController();
      request.current = controller;
      setLoading(true);
      setError('');
      setHover(undefined);
      try {
        let buffer = cache.current.get(chunk.url);
        if (!buffer) {
          const response = await fetch(`/${chunk.url}`, {
            signal: controller.signal,
          });
          if (!response.ok)
            throw new Error(
              `Map download failed (${response.status}). Try again.`,
            );
          buffer = await response.arrayBuffer();
          if (buffer.byteLength !== chunk.bytes)
            throw new Error('Incomplete map download. Please try again.');
          if (crypto.subtle) {
            const digest = await crypto.subtle.digest('SHA-256', buffer);
            const hash = Array.from(new Uint8Array(digest), (b) =>
              b.toString(16).padStart(2, '0'),
            ).join('');
            if (hash !== chunk.sha256)
              throw new Error(
                'The map integrity check failed. Please refresh and try again.',
              );
          }
          if (controller.signal.aborted) return;
          cache.current.set(chunk.url, buffer);
        }
        const values = decodeMonth(
          buffer,
          chunk.days,
          archive.catalog.cells,
          Number(date.slice(-2)),
        );
        if (!controller.signal.aborted) {
          flushSync(() => {
            setLoaded({ model, date, values });
            setLoading(false);
          });
          return {
            status: 'displayed',
            model,
            date,
            cells: values.length,
            ...archive.daily[date].maps[model],
          };
        }
      } catch (e) {
        if (e instanceof Error && e.name !== 'AbortError') {
          setError(e.message);
          return { status: 'error', message: e.message };
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    },
    [archive],
  );

  useEffect(() => {
    // The initial fetch synchronises the view with an external saved archive.
    // eslint-disable-next-line react/react-compiler
    if (archive) void generate('hybrid', archive.catalog.defaultDate);
    return () => request.current?.abort();
  }, [archive, generate]);

  useEffect(() => {
    if (!archive) return;
    const context = (document as Document & { modelContext?: ModelContext })
      .modelContext;
    return registerDashboardTools(
      context,
      archive.catalog,
      async (model, date) => {
        flushSync(() => {
          setTab('explorer');
          setDraftModel(model);
          setDraftDate(date);
        });
        const result = await generate(model, date);
        if (!result || result.status !== 'displayed')
          throw new Error(
            result && 'message' in result
              ? result.message
              : 'Map request was cancelled.',
          );
        return result;
      },
    );
  }, [archive, generate]);

  useEffect(() => {
    const keyboard = () => {
      document.documentElement.dataset.input = 'keyboard';
    };
    const pointer = () => {
      document.documentElement.dataset.input = 'pointer';
    };
    window.addEventListener('keydown', keyboard);
    window.addEventListener('pointerdown', pointer);
    return () => {
      window.removeEventListener('keydown', keyboard);
      window.removeEventListener('pointerdown', pointer);
    };
  }, []);

  const model = archive?.catalog.models.find((m) => m.id === loaded?.model);
  const day = loaded && archive?.daily[loaded.date];
  const stats = loaded && day?.maps[loaded.model];
  const score = loaded && day?.scores[loaded.model];
  const scale: [number, number] = fixedScale
    ? [0, 50]
    : stats
      ? niceScale(stats.min, stats.max)
      : [0, 10];
  const outline = useMemo(
    () => (archive ? outlinePath(archive.geometry) : ''),
    [archive],
  );
  const dirty =
    loaded && (loaded.model !== draftModel || loaded.date !== draftDate);

  const stepDay = (offset: number) => {
    if (!loaded) return;
    const date = adjacentDate(loaded.date, offset);
    setDraftDate(date);
    setDraftModel(loaded.model);
    void generate(loaded.model, date);
  };

  async function downloadPng() {
    if (!mapRef.current || !loaded || !model) return;
    setExporting(true);
    setError('');
    let url = '';
    try {
      const clone = mapRef.current.cloneNode(true) as SVGSVGElement;
      clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      clone.setAttribute('width', '1500');
      clone.setAttribute('height', '1125');
      url = URL.createObjectURL(
        new Blob([new XMLSerializer().serializeToString(clone)], {
          type: 'image/svg+xml',
        }),
      );
      const image = new Image();
      image.src = url;
      await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = 1600;
      canvas.height = 1450;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Image export is unavailable in this browser.');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, 1600, 1450);
      ctx.fillStyle = '#182c28';
      ctx.font = 'bold 34px Arial';
      ctx.fillText('Aitken | Greater London PM2.5', 60, 56);
      ctx.font = '24px Arial';
      ctx.fillText(`${model.label} — ${prettyDate(loaded.date)}`, 60, 100);
      ctx.drawImage(image, 50, 122, 1500, 1125);
      for (let x = 0; x < 480; x++) {
        ctx.fillStyle = concentrationColor(
          scale[0] + (x / 479) * (scale[1] - scale[0]),
          ...scale,
        );
        ctx.fillRect(60 + x, 1270, 2, 18);
      }
      ctx.fillStyle = '#344740';
      ctx.font = '20px Arial';
      ctx.fillText(
        `${fmt(scale[0], 1)}                         ${fmt(scale[1], 1)} µg/m³`,
        60,
        1320,
      );
      ctx.fillText(
        `${fixedScale ? 'Fixed 0–50 scale' : 'Day-specific scale'} · 1 km grid · Predicted daily mean`,
        620,
        1290,
      );
      ctx.font = '18px Arial';
      ctx.fillText(
        'Retrospective 2024 archive; trained on 2021–2023. Not a live forecast or measured surface.',
        60,
        1370,
      );
      if (hotspots)
        ctx.fillText(
          'Outlined cells: highest predicted 10% on this day; not a health threshold.',
          60,
          1400,
        );
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, 'image/png'),
      );
      if (!blob) throw new Error('Could not export the image.');
      const download = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = download;
      link.download = `aitken-${loaded.model}-${loaded.date}.png`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(download), 10000);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not export the image.');
    } finally {
      if (url) URL.revokeObjectURL(url);
      setExporting(false);
    }
  }

  return (
    <div className="app-shell">
      <a href="#main" className="skip-link">
        Skip to dashboard
      </a>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <Waves size={24} />
          </span>
          <span>
            Aitken<span className="brand-sub">London observatory</span>
          </span>
        </div>
        <div className="header-right">
          <span className="archive-badge">
            <span />
            2024 research archive
          </span>
          <a
            href={github}
            target="_blank"
            rel="noreferrer"
            className="source-link"
          >
            View research <ArrowUpRight size={16} />
          </a>
        </div>
      </header>
      <main id="main" className="workspace">
        <div className="page-heading">
          <div>
            <p className="eyebrow">AIR QUALITY / GREATER LONDON</p>
            <h1>A clearer picture of London’s air.</h1>
            <p className="intro">
              Explore daily PM2.5 estimates. See how our models compare.
            </p>
          </div>
          <div className="study-period">
            <span>TRAINED ON</span>
            <strong>2021–2023</strong>
            <span>Evaluated on 2024 observations</span>
          </div>
        </div>
        {!archive ? (
          <section className="empty-state" aria-live="polite">
            <Layers3 size={32} />
            <h2>
              {initialError
                ? 'Archive unavailable'
                : 'Loading the research archive…'}
            </h2>
            <p>
              {initialError || 'Preparing model scores and the London grid.'}
            </p>
            {initialError && (
              <Button
                onClick={() => {
                  setInitialError('');
                  setAttempt((a) => a + 1);
                }}
              >
                Try again
              </Button>
            )}
          </section>
        ) : (
          <>
            <Tabs
              value={tab}
              onValueChange={(v) => setTab(String(v))}
              className="main-tabs"
            >
              <TabsList variant="line" className="navigation-tabs">
                <TabsTrigger value="explorer">
                  <MapIcon size={17} />
                  Heatmap explorer
                </TabsTrigger>
                <TabsTrigger value="benchmark">
                  <BarChart3 size={17} />
                  Model benchmark
                </TabsTrigger>
              </TabsList>
              <TabsContent value="explorer">
                <form
                  className="control-bar"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void generate(draftModel, draftDate);
                  }}
                >
                  <div className="model-field">
                    <label id="model-label" htmlFor="model-picker">
                      Prediction model
                    </label>
                    <Select
                      value={draftModel}
                      onValueChange={(v) => v && setDraftModel(v)}
                    >
                      <SelectTrigger
                        id="model-picker"
                        aria-labelledby="model-label"
                      >
                        <SelectValue>
                          {
                            archive.catalog.models.find(
                              (m) => m.id === draftModel,
                            )?.label
                          }
                        </SelectValue>
                      </SelectTrigger>
                      <SelectContent align="start" alignItemWithTrigger={false}>
                        {archive.catalog.models
                          .filter((m) => m.hasMaps)
                          .map((m) => (
                            <SelectItem key={m.id} value={m.id}>
                              {m.label}
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="date-field">
                    <label htmlFor="map-date">Date</label>
                    <input
                      id="map-date"
                      type="date"
                      required
                      min={archive.catalog.start}
                      max={archive.catalog.end}
                      value={draftDate}
                      onChange={(e) => setDraftDate(e.target.value)}
                    />
                  </div>
                  <Button
                    className="generate-button"
                    type="submit"
                    disabled={loading}
                  >
                    <Layers3 size={16} />
                    {loading ? 'Loading map…' : 'Generate heatmap'}
                  </Button>
                  <span className="control-note">
                    366 days available
                    <br />1 Jan – 31 Dec 2024
                  </span>
                </form>
                {error && (
                  <div className="error-notice" role="alert">
                    {error}
                  </div>
                )}
                {dirty && (
                  <output className="draft-notice">
                    Selection changed. Choose Generate heatmap to update the
                    map.
                  </output>
                )}
                <div className="explorer-grid">
                  <section
                    className="map-card panel"
                    aria-busy={loading}
                    aria-label="Daily prediction map"
                  >
                    <div className="map-heading">
                      <div>
                        <p className="eyebrow">PREDICTED DAILY PM2.5</p>
                        <h2>
                          {loaded
                            ? prettyDate(loaded.date)
                            : 'Preparing your map'}
                        </h2>
                        <p>{model?.label || 'Loading saved predictions…'}</p>
                      </div>
                      <div className="map-actions">
                        <Button
                          variant="outline"
                          size="icon"
                          aria-label="Previous day"
                          disabled={
                            loading ||
                            !loaded ||
                            loaded.date <= archive.catalog.start
                          }
                          onClick={() => stepDay(-1)}
                        >
                          <ChevronLeft />
                        </Button>
                        <Button
                          variant="outline"
                          size="icon"
                          aria-label="Next day"
                          disabled={
                            loading ||
                            !loaded ||
                            loaded.date >= archive.catalog.end
                          }
                          onClick={() => stepDay(1)}
                        >
                          <ChevronRight />
                        </Button>
                        <Button
                          variant="outline"
                          onClick={downloadPng}
                          disabled={!loaded || loading || exporting}
                          aria-label="Download current heatmap as PNG"
                        >
                          <ArrowDownToLine size={16} />
                          <span className="download-label">
                            {exporting ? 'Exporting…' : 'PNG'}
                          </span>
                        </Button>
                      </div>
                    </div>
                    <div className="map-stage">
                      {/* The chart has an image role; mouse hover only exposes a supplementary tooltip. */}
                      {/* eslint-disable jsx-a11y/prefer-tag-over-role, jsx-a11y/no-noninteractive-element-interactions */}
                      {loaded ? (
                        <svg
                          ref={mapRef}
                          className="london-map"
                          viewBox="-4 -2 72 54"
                          role="img"
                          aria-labelledby="map-title map-description"
                          onMouseLeave={() => setHover(undefined)}
                        >
                          <title id="map-title">
                            {model?.label} PM2.5 prediction for{' '}
                            {prettyDate(loaded.date)}
                          </title>
                          <desc id="map-description">
                            Greater London, {archive.catalog.cells}{' '}
                            one-kilometre cells. Range {fmt(stats?.min)} to{' '}
                            {fmt(stats?.max)} micrograms per cubic metre.
                            Predicted, not measured, concentrations.
                          </desc>
                          <rect
                            x="-4"
                            y="-2"
                            width="72"
                            height="54"
                            fill="#f4f7f6"
                          />
                          <g stroke="#dfe7e3" strokeWidth="0.06">
                            {[0, 10, 20, 30, 40, 50, 60].map((x) => (
                              <path key={x} d={`M${x},0v48`} />
                            ))}
                            {[2, 12, 22, 32, 42].map((y) => (
                              <path key={y} d={`M0,${y}h64`} />
                            ))}
                          </g>
                          <g shapeRendering="crispEdges">
                            {archive.geometry.indices.map((cell, i) => (
                              <rect
                                key={cell}
                                x={cell % archive.geometry.cols}
                                y={Math.floor(cell / archive.geometry.cols)}
                                width="1"
                                height="1"
                                fill={concentrationColor(
                                  loaded.values[i],
                                  ...scale,
                                )}
                                stroke={
                                  hotspots &&
                                  stats &&
                                  loaded.values[i] >= stats.p90
                                    ? '#ffffff'
                                    : 'none'
                                }
                                strokeWidth="0.1"
                                onMouseEnter={() => setHover(i)}
                              >
                                <title>
                                  {fmt(loaded.values[i])} µg/m³ · E{' '}
                                  {500 + (cell % 64)} km · N{' '}
                                  {202 - Math.floor(cell / 64)} km (cell corner)
                                </title>
                              </rect>
                            ))}
                          </g>
                          <path
                            d={outline}
                            fill="none"
                            stroke="#344740"
                            strokeWidth="0.12"
                            pointerEvents="none"
                          />
                          <g
                            fill="#62736d"
                            fontFamily="Arial, sans-serif"
                            fontSize="1.05"
                          >
                            {[0, 10, 20, 30, 40, 50, 60].map((x) => (
                              <text key={x} x={x} y="50.5" textAnchor="middle">
                                {500 + x}
                              </text>
                            ))}
                            {[2, 12, 22, 32, 42].map((y) => (
                              <text
                                key={y}
                                x="-1.4"
                                y={y + 0.35}
                                textAnchor="end"
                              >
                                {202 - y}
                              </text>
                            ))}
                            <text x="32" y="52" textAnchor="middle">
                              British National Grid · kilometres
                            </text>
                          </g>
                          <g
                            transform="translate(61 4)"
                            fill="#344740"
                            stroke="#344740"
                            strokeWidth="0.13"
                          >
                            <path d="M0,4V0M-.55,1L0,0 .55,1" fill="none" />
                            <text
                              x="0"
                              y="5.5"
                              stroke="none"
                              textAnchor="middle"
                              fontSize="1.1"
                              fontFamily="Arial"
                            >
                              N
                            </text>
                          </g>
                          <g
                            transform="translate(2 45)"
                            fill="#344740"
                            fontFamily="Arial"
                            fontSize="1.05"
                          >
                            <path
                              d="M0,0v1M0,.5h10M10,0v1"
                              stroke="#344740"
                              strokeWidth="0.15"
                            />
                            <text x="5" y="-.6" textAnchor="middle">
                              10 km
                            </text>
                          </g>
                        </svg>
                      ) : (
                        <div className="map-placeholder">
                          <MapIcon size={36} />
                          <p>Loading London’s prediction grid…</p>
                        </div>
                      )}
                      {/* eslint-enable jsx-a11y/prefer-tag-over-role, jsx-a11y/no-noninteractive-element-interactions */}
                      {loading && loaded && (
                        <output className="loading-pill">
                          Loading next map…
                        </output>
                      )}
                    </div>
                    <div className="map-legend">
                      <div>
                        <strong>
                          PM2.5 concentration <span>µg/m³</span>
                        </strong>
                        <div className="color-ramp" />
                        <div className="legend-ticks">
                          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                            <span key={t}>
                              {fmt(scale[0] + t * (scale[1] - scale[0]), 1)}
                            </span>
                          ))}
                        </div>
                      </div>
                      <p>
                        {hover !== undefined && loaded ? (
                          <>
                            <strong>{fmt(loaded.values[hover])} µg/m³</strong>
                            <br />
                            Selected 1 km grid cell
                          </>
                        ) : (
                          <>
                            {fixedScale
                              ? 'Fixed scale · compare across dates'
                              : 'Day-specific scale · see local variation'}
                            <br />
                            Colours are not health-risk categories.
                          </>
                        )}
                      </p>
                    </div>
                    <div className="map-footnote">
                      <span>
                        <span className="tiny-dot" />1 km grid ·{' '}
                        {archive.catalog.cells.toLocaleString()} London cells
                      </span>
                      <span>Saved model output · no smoothing</span>
                    </div>
                  </section>
                  <aside className="insight-column">
                    <section className="panel day-summary">
                      <p className="eyebrow">ACROSS LONDON</p>
                      <p className="big-number">
                        {fmt(stats?.median)}
                        <span>µg/m³</span>
                      </p>
                      <p className="muted">Median predicted concentration</p>
                      <div className="range-pair">
                        <div>
                          <span>Lowest cell</span>
                          <strong>{fmt(stats?.min)}</strong>
                        </div>
                        <div>
                          <span>Highest cell</span>
                          <strong>{fmt(stats?.max)}</strong>
                        </div>
                      </div>
                      <p className="caption">
                        Spatial variation in the selected model’s estimate, not
                        measurements at every location.
                      </p>
                    </section>
                    <section className="panel accuracy-panel">
                      <div className="section-label">
                        <h3>Accuracy on this day</h3>
                        <CircleHelp size={16} />
                      </div>
                      {day?.n ? (
                        <>
                          <div className="score-pair">
                            <div>
                              <span>RMSE</span>
                              <strong>{fmt(score?.rmse)}</strong>
                            </div>
                            <div>
                              <span>MAE</span>
                              <strong>{fmt(score?.mae)}</strong>
                            </div>
                          </div>
                          <p className="caption">
                            µg/m³ · lower is better
                            <br />
                            Compared with {day.n} LAQN monitoring observations.
                          </p>
                        </>
                      ) : (
                        <p className="caption">
                          No monitoring labels are available for this date. A
                          prediction map is available, but its daily error
                          cannot be calculated.
                        </p>
                      )}
                      <button
                        className="text-button"
                        onClick={() => setTab('benchmark')}
                      >
                        See full-year comparison <ArrowUpRight size={15} />
                      </button>
                    </section>
                    <section className="panel display-panel">
                      <h3>Map display</h3>
                      <div className="switch-row">
                        <label htmlFor="fixed-scale">
                          Fixed concentration scale
                          <span>0–50 µg/m³ for consistent comparison</span>
                        </label>
                        <Switch
                          id="fixed-scale"
                          checked={fixedScale}
                          onCheckedChange={setFixedScale}
                        />
                      </div>
                      <div className="switch-row">
                        <label htmlFor="hotspots">
                          Outline relative hotspots
                          <span>Top 10% of this day’s predictions</span>
                        </label>
                        <Switch
                          id="hotspots"
                          checked={hotspots}
                          onCheckedChange={setHotspots}
                        />
                      </div>
                      {fixedScale && stats && stats.max > 50 && (
                        <p className="caption">
                          Values above 50 µg/m³ share the highest colour; the
                          values themselves are unchanged.
                        </p>
                      )}
                      {hotspots && (
                        <p className="caption">
                          White outlines mark values ≥ {fmt(stats?.p90)} µg/m³.
                          This is a relative ranking, not a health threshold.
                        </p>
                      )}
                    </section>
                    <div className="method-note">
                      <Layers3 size={18} />
                      <div>
                        <strong>
                          {loaded?.model === 'hybrid'
                            ? 'A hybrid, explained simply'
                            : loaded?.model.startsWith('unet')
                              ? 'U-Net, without HGB'
                              : 'A tree-based baseline'}
                        </strong>
                        <p>
                          {loaded?.model === 'hybrid'
                            ? 'HGB makes the starting estimate. Three U-Nets learn a spatial correction, combined in transformed target space to produce this final map.'
                            : loaded?.model.startsWith('unet')
                              ? 'The network directly estimates PM2.5 from the spatial input layers. No HGB prediction or residual correction is used.'
                              : 'Gradient-boosted decision trees estimate PM2.5 from environmental, location and historical features at each cell.'}
                        </p>
                      </div>
                    </div>
                  </aside>
                </div>
                <div className="archive-notice">
                  <CircleHelp size={17} />
                  <p>
                    <strong>A prediction archive, not a live forecast.</strong>{' '}
                    Maps are rendered on demand from saved 2024 model outputs;
                    changing a date does not retrain a model. ANN and XGBoost
                    have benchmark scores, but no saved full-grid maps.
                  </p>
                </div>
              </TabsContent>
              <TabsContent value="benchmark">
                <Benchmark
                  catalog={archive.catalog}
                  onExplore={(id) => {
                    setDraftModel(id);
                    setTab('explorer');
                    void generate(id, draftDate);
                  }}
                />
              </TabsContent>
            </Tabs>
            <details className="research-notes">
              <summary>
                Research notes & limitations <ChevronRight size={17} />
              </summary>
              <div className="notes-grid">
                <div>
                  <h3>What is being compared?</h3>
                  <p>
                    Models were fitted on 2021–2023 and compared on the same{' '}
                    {archive.catalog.stationDays.toLocaleString()} LAQN
                    station-days over {archive.catalog.labelledDays} dates in
                    2024. Labels end on {prettyDate(archive.catalog.labelEnd)}.
                    Some 2024 results were inspected in earlier experiments, so
                    this is a retrospective benchmark, not an untouched final
                    test.
                  </p>
                  <p>
                    RMSE penalises larger errors; MAE is the average absolute
                    error. Both are in µg/m³. R² is a unitless measure of fit,
                    not an accuracy percentage.
                  </p>
                </div>
                <div>
                  <h3>What the map cannot tell us</h3>
                  <ul>
                    {archive.catalog.limitations.slice(1, 4).map((note) => (
                      <li key={note}>{note}</li>
                    ))}
                  </ul>
                  <p>{archive.catalog.mapMetricsNote}</p>
                  <a
                    href={`${github}/blob/main/docs/RESEARCH_PROTOCOL.md`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Read the complete research protocol{' '}
                    <ArrowUpRight size={14} />
                  </a>
                </div>
              </div>
            </details>
          </>
        )}
      </main>
      <footer className="site-footer">
        <span>
          <Waves size={16} /> Aitken · Environmental modelling research
        </span>
        <span>Estimates, not official air-quality guidance.</span>
      </footer>
    </div>
  );
}

function Benchmark({
  catalog,
  onExplore,
}: {
  catalog: Catalog;
  onExplore: (id: string) => void;
}) {
  const [metric, setMetric] = useState<Metric>('rmse');
  const ordered = [...catalog.models].sort((a, b) =>
    metric === 'r2'
      ? b.metrics[metric] - a.metrics[metric]
      : a.metrics[metric] - b.metrics[metric],
  );
  const max =
    metric === 'r2'
      ? 1
      : Math.ceil(Math.max(...ordered.map((m) => m.metrics[metric])));
  const best = catalog.models.find((m) => m.id === 'hybrid')!;
  const hgb = catalog.models.find((m) => m.id === 'hgb')!;
  const direct = catalog.models.find((m) => m.id === 'unet-ensemble')!;
  return (
    <div className="benchmark">
      <div className="benchmark-intro">
        <h2>Same data. Seven model comparisons.</h2>
        <p>
          {catalog.stationDays.toLocaleString()} station-days ·{' '}
          {catalog.labelledDays} observed dates · 2024 evaluation
        </p>
      </div>
      <div className="benchmark-stats">
        <div className="panel">
          <p className="eyebrow">LOWEST OBSERVED RMSE</p>
          <strong>
            {fmt(best.metrics.rmse, 4)}
            <span>µg/m³</span>
          </strong>
          <p>Hybrid HGB + residual U-Net</p>
        </div>
        <div className="panel">
          <p className="eyebrow">HYBRID ADVANTAGE OVER HGB</p>
          <strong>
            {fmt(hgb.metrics.rmse - best.metrics.rmse, 4)}
            <span>µg/m³</span>
          </strong>
          <p>A small improvement, not a decisive win</p>
        </div>
        <div className="panel">
          <p className="eyebrow">STANDALONE U-NET RMSE</p>
          <strong>
            {fmt(direct.metrics.rmse, 4)}
            <span>µg/m³</span>
          </strong>
          <p>Three-seed ensemble · no HGB input</p>
        </div>
      </div>
      <section className="panel comparison-chart">
        <div className="chart-heading">
          <div>
            <h3>Model performance</h3>
            <p>
              {metric === 'r2'
                ? 'Higher is better · unitless · axis starts at zero'
                : 'Lower is better · µg/m³ · axis starts at zero'}
            </p>
          </div>
          <Tabs value={metric} onValueChange={(v) => setMetric(v as Metric)}>
            <TabsList aria-label="Comparison metric">
              <TabsTrigger value="rmse">RMSE</TabsTrigger>
              <TabsTrigger value="mae">MAE</TabsTrigger>
              <TabsTrigger value="r2">R²</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
        {/* A CSS chart is exposed as one labelled image; its accessible data table follows. */}
        {/* eslint-disable jsx-a11y/prefer-tag-over-role */}
        <div
          className="bar-chart"
          role="img"
          aria-label={`All seven models ranked by ${metric}. Exact values are in the table below.`}
        >
          {ordered.map((m, i) => (
            <div className="bar-row" key={m.id}>
              <div className="bar-name">
                <span>{String(i + 1).padStart(2, '0')}</span>
                {m.label}
              </div>
              <div className="bar-track">
                <div
                  className="bar-fill"
                  style={{
                    width: `${Math.max(0, m.metrics[metric] / max) * 100}%`,
                    background: m.color,
                  }}
                />
              </div>
              <strong>{fmt(m.metrics[metric], 4)}</strong>
            </div>
          ))}
          <div className="bar-axis">
            <span />
            <div>
              {[0, 0.25, 0.5, 0.75, 1].map((n) => (
                <span key={n}>{fmt(n * max, metric === 'r2' ? 2 : 1)}</span>
              ))}
            </div>
            <span />
          </div>
        </div>
      </section>
      {/* eslint-enable jsx-a11y/prefer-tag-over-role */}
      <section className="panel results-table">
        <div className="table-heading">
          <h3>Verified results</h3>
          <span>
            <Check size={15} />
            Identical evaluation rows
          </span>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead className="numeric">MAE ↓</TableHead>
              <TableHead className="numeric">RMSE ↓</TableHead>
              <TableHead className="numeric">R² ↑</TableHead>
              <TableHead>Map archive</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {catalog.models.map((m: Model) => (
              <TableRow
                key={m.id}
                className={m.id === 'hybrid' ? 'best-row' : ''}
              >
                <TableCell>
                  <div className="model-name">
                    <span style={{ background: m.color }} />
                    <div>
                      <strong>{m.label}</strong>
                      <small>{m.detail}</small>
                    </div>
                  </div>
                </TableCell>
                <TableCell className="numeric">
                  {fmt(m.metrics.mae, 4)}
                </TableCell>
                <TableCell className="numeric">
                  {fmt(m.metrics.rmse, 4)}
                </TableCell>
                <TableCell className="numeric">
                  {fmt(m.metrics.r2, 4)}
                </TableCell>
                <TableCell>
                  {m.hasMaps ? (
                    <button
                      className="text-button"
                      onClick={() => onExplore(m.id)}
                    >
                      Explore <ArrowUpRight size={14} />
                    </button>
                  ) : (
                    <span className="muted">Scores only</span>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </section>
      <div className="archive-notice">
        <CircleHelp size={17} />
        <p>
          <strong>Read the margin, not just the rank.</strong> The hybrid has
          the lowest errors in this comparison, but its RMSE is only{' '}
          {fmt(hgb.metrics.rmse - best.metrics.rmse, 4)} µg/m³ lower than HGB.
          The descriptive day-bootstrap analysis does not establish a decisive
          advantage. This does not show that U-Net is universally the best
          model.
        </p>
      </div>
    </div>
  );
}
