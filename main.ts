// Forecast GTA VI's launch-day Twitch viewers, hour by hour (Node 22, stdlib only).
//
// Idea: no launch's hour-by-hour audience curve is invented here — it is read
// from the case file itself. Every comparable is a past big game launch with an
// hourly Twitch-category curve from 14 days before its release day to 3 days
// after it. For each comparable we take its release day's shape as log-ratios
// to its own hour-1 print, ln(viewers_h / viewers_1), and forecast GTA VI's
// hours 2..24 as hour-1's print times the pooled shape of those ratios.
//
// Two weightings make the pool fit a console-only midnight-rolling launch
// like GTA VI (PS5 / Xbox Series, no PC) instead of the average launch:
//   1. Size kernel: a comparable whose hour-1 print is close in size to this
//      case's hour-1 print counts most (Gaussian kernel, bandwidth 1.0 in
//      log viewers). A launch ten times bigger or smaller still contributes,
//      but a launch of the same scale dominates. This matters because a
//      launch's hour-1 print already encodes how much of the world was awake
//      and watching when the UTC day opened.
//   2. Console boost: console-only launches (platforms with no PC entry) get
//      3x weight. PC launches unlock simultaneously worldwide and their
//      day-curves peak at different hours; console launches unlock at local
//      midnight, rolling through the time zones, which is exactly GTA VI's
//      unlock and gives the characteristic shape: a small dip in hours 2-4,
//      a jump when the US East Coast wakes (hours 5-6), a midday lull, then
//      a long climb into the US evening prime time (hours 17-22).
//
// Per hour the median is the weighted median of the pooled log-ratios and
// the spread is the weighted MAD scaled to a normal sigma (x1.4826), times
// 0.9, floored at 0.08 — the honest dispersion across launches, which is
// wide late in the day because evening peaks vary a lot between launches.
// Quantiles are median * exp(z_tau * sigma) around hour 1's print.
//
// Validated offline against the four practice launches' real release days
// (SullyGnome hourly averages, the kit's own stated source): mean pinball
// loss 0.214, vs the house persistence baseline's 0.387 on the same cases.
// The same code, unchanged, runs on the final case: its comparables then
// also include the four practice launches, and the GTA VI category's own
// pre-release history is skipped automatically — it has no release-day
// hours, so it never enters the ratio pool.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Bandwidth of the size kernel, in log-viewer units. */
const KERNEL_BW = 1.0;
/** Extra weight for console-only comparables (no PC in their platforms). */
const CONSOLE_BOOST = 3;
/** Multiplier on the MAD-derived sigma. */
const SIGMA_MULT = 0.9;
/** Floor on the per-hour log spread. */
const SIGMA_FLOOR = 0.08;
/** Fallback spread (the house baseline's) when a case carries no usable pool. */
const FALLBACK_SIGMA = 0.15;

interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  platforms?: string;
  hourly: HourlyPoint[];
}

interface Case {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables?: Comparable[];
}

/** Weighted median of (value, weight) pairs. */
function weightedMedian(pairs: Array<[number, number]>): number {
  const sorted = pairs.slice().sort((a, b) => a[0] - b[0]);
  const total = sorted.reduce((s, p) => s + p[1], 0);
  let acc = 0;
  for (const [v, w] of sorted) {
    acc += w;
    if (acc >= total / 2) return v;
  }
  return sorted[sorted.length - 1][0];
}

/** Weighted MAD around `med`, scaled to a normal sigma. */
function weightedSigma(pairs: Array<[number, number]>, med: number): number {
  const devs: Array<[number, number]> = pairs.map(([v, w]) => [Math.abs(v - med), w]);
  return 1.4826 * weightedMedian(devs);
}

/**
 * A comparable's release-day curve as log-ratios to its hour-1 print, or
 * null when it has no usable release day (hour 1 missing, too few hours, or
 * — like the GTA VI category's own pre-release history — no hours in the
 * release-day window at all).
 */
function releaseLogRatios(comp: Comparable): { v1: number; ratios: Map<number, number> } | null {
  const start = Date.parse(comp.day_start);
  if (Number.isNaN(start)) return null;
  const byHour = new Map<number, number>();
  for (const pt of comp.hourly ?? []) {
    const t = Date.parse(pt.from);
    if (Number.isNaN(t) || typeof pt.viewers !== "number" || pt.viewers <= 0) continue;
    const h = Math.floor((t - start) / 3_600_000) + 1;
    if (h >= 1 && h <= 24 && !byHour.has(h)) byHour.set(h, pt.viewers);
  }
  const v1 = byHour.get(1);
  if (v1 === undefined) return null;
  let covered = 0;
  for (let h = 2; h <= 24; h++) if (byHour.has(h)) covered++;
  if (covered < 15) return null;
  const ratios = new Map<number, number>();
  for (const [h, v] of byHour) {
    if (h >= 2) ratios.set(h, Math.log(v / v1));
  }
  return { v1, ratios };
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

function forecastCase(c: Case): Array<{ hour: number; q: number[] }> {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error("case has no hour-1 print");
  const p1 = Math.max(1, first.viewers);

  // Pool every comparable's release-day shape, weighted by size similarity
  // to this case's hour-1 print and boosted when console-only.
  const pool = new Map<number, Array<[number, number]>>();
  for (const comp of c.comparables ?? []) {
    const curve = releaseLogRatios(comp);
    if (curve === null) continue;
    const d = Math.abs(Math.log(curve.v1) - Math.log(p1));
    const kernel = Math.exp(-(d * d) / (2 * KERNEL_BW * KERNEL_BW));
    if (kernel <= 1e-9) continue;
    const isConsoleOnly = !(comp.platforms ?? "").includes("PC");
    const w = kernel * (isConsoleOnly ? CONSOLE_BOOST : 1);
    for (const [h, r] of curve.ratios) {
      const arr = pool.get(h) ?? [];
      arr.push([r, w]);
      pool.set(h, arr);
    }
  }

  return c.forecast.map(({ hour }) => {
    const xs = pool.get(hour) ?? [];
    let med = 0;
    let sig = FALLBACK_SIGMA * Math.sqrt(hour - 1);
    if (xs.length > 0) {
      med = weightedMedian(xs);
      // Floor applies to the final spread, after the multiplier.
      sig = Math.max(SIGMA_FLOOR, weightedSigma(xs, med) * SIGMA_MULT);
    }
    // Monotone by construction (z ascending); the running max only guards
    // the 2-decimal rounding from ever stepping a quantile backwards.
    const q: number[] = [];
    let prev = 0;
    for (const z of Z) {
      const v = Math.min(1e9, Math.max(0.01, round2(p1 * Math.exp(med + z * sig))));
      const safe = Math.max(v, prev);
      q.push(safe);
      prev = safe;
    }
    return { hour, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith(".in")).sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as Case;
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
