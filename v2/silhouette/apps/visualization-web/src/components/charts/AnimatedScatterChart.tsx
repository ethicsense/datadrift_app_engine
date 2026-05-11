import { useEffect, useMemo, useRef, useState } from "react";

import { productImageApiUrl } from "../../lib/api";
import { ScatterSquareChart } from "./ScatterSquareChart";
import { computeEqualScaleScatterExtents } from "../../lib/chartScatterBounds";
import { escapeHtml } from "../../lib/escapeHtml";
import { formatNumber } from "../../lib/formatters";

type ScatterFrame = {
  snapshotId: string;
  label: string;
  newCount?: number;
  retainedCount?: number;
  exitedCount?: number;
  clusterShareTopN?: Array<{
    clusterId: string;
    pointCount: number;
    sharePct: number;
    dominantCategory: string;
    dominantSharePct: number;
  }>;
  points: Array<{
    productId: string;
    brand?: string;
    name?: string;
    rank?: number | null;
    rankVelocity?: number | null;
    movementGroup?: string | null;
    clusterId?: string | null;
    dominantCategory?: string | null;
    x?: number | null;
    y?: number | null;
    mainImage?: string | null;
    lifecycleState?: "new" | "retained" | "exited" | null;
    isGhost?: boolean;
  }>;
};

type AnimatedScatterChartProps = {
  frames: ScatterFrame[];
};

const movementPalette: Record<string, string> = {
  new: "#a855f7",
  retained: "#38bdf8",
  exited: "#f59e0b",
  unknown: "#8b5cf6",
};

const movementLabelMap: Record<string, string> = {
  new: "신규 출현",
  retained: "유지",
  exited: "이탈(잔상)",
  unknown: "비교 기준 없음",
};

const DENSITY_BINS = 52;
const BASE_FRAME_DURATION_MS = 3200;
const DEFAULT_SPEED = 1.35;
const DEFAULT_GLOW = 0.7;
const DEFAULT_TRAIL = 0.72;

function smootherStep(t: number) {
  const clamped = Math.max(0, Math.min(1, t));
  return clamped * clamped * clamped * (clamped * (clamped * 6 - 15) + 10);
}

function temporalWeight(distance: number, sigma: number) {
  return Math.exp(-((distance * distance) / (2 * sigma * sigma)));
}

function blurGrid(grid: number[], size: number, passes = 2) {
  let current = grid.slice();
  const kernel = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
  ];
  const kernelWeight = 16;
  for (let pass = 0; pass < passes; pass += 1) {
    const next = new Array<number>(size * size).fill(0);
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky += 1) {
          for (let kx = -1; kx <= 1; kx += 1) {
            const sy = Math.max(0, Math.min(size - 1, y + ky));
            const sx = Math.max(0, Math.min(size - 1, x + kx));
            sum += current[sy * size + sx] * kernel[ky + 1][kx + 1];
          }
        }
        next[y * size + x] = sum / kernelWeight;
      }
    }
    current = next;
  }
  return current;
}

export function AnimatedScatterChart({ frames }: AnimatedScatterChartProps) {
  const [playing, setPlaying] = useState(true);
  const [progress, setProgress] = useState(0);
  const [speedMultiplier, setSpeedMultiplier] = useState(DEFAULT_SPEED);
  const [glowStrength, setGlowStrength] = useState(DEFAULT_GLOW);
  const [trailStrength, setTrailStrength] = useState(DEFAULT_TRAIL);
  const [interactionMode, setInteractionMode] = useState<"cluster" | "hotspot">("cluster");
  const [hoveredClusterId, setHoveredClusterId] = useState<string | null>(null);
  const [selectedHotspotId, setSelectedHotspotId] = useState<string | null>(null);
  const directionRef = useRef<1 | -1>(1);

  const cross = (
    origin: [number, number],
    pointA: [number, number],
    pointB: [number, number],
  ) => (pointA[0] - origin[0]) * (pointB[1] - origin[1]) - (pointA[1] - origin[1]) * (pointB[0] - origin[0]);

  const buildConvexHull = (points: Array<[number, number]>) => {
    if (points.length <= 3) {
      return points;
    }
    const sorted = [...points].sort((left, right) => {
      if (left[0] !== right[0]) {
        return left[0] - right[0];
      }
      return left[1] - right[1];
    });
    const lower: Array<[number, number]> = [];
    sorted.forEach((point) => {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
        lower.pop();
      }
      lower.push(point);
    });
    const upper: Array<[number, number]> = [];
    [...sorted].reverse().forEach((point) => {
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
        upper.pop();
      }
      upper.push(point);
    });
    lower.pop();
    upper.pop();
    return [...lower, ...upper];
  };

  useEffect(() => {
    if (!playing || frames.length <= 1) {
      return undefined;
    }
    let rafId = 0;
    let prev = performance.now();
    const tick = (now: number) => {
      const delta = Math.max(0, now - prev);
      prev = now;
      setProgress((current) => {
        const maxProgress = Math.max(frames.length - 1, 0);
        if (maxProgress <= 0) {
          return 0;
        }
        let next = current + (delta / (BASE_FRAME_DURATION_MS / speedMultiplier)) * directionRef.current;
        if (next >= maxProgress) {
          next = maxProgress;
          directionRef.current = -1;
        } else if (next <= 0) {
          next = 0;
          directionRef.current = 1;
        }
        return next;
      });
      rafId = window.requestAnimationFrame(tick);
    };
    rafId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(rafId);
  }, [frames.length, playing, speedMultiplier]);

  useEffect(() => {
    if (!frames.length) {
      return;
    }
    setProgress((current) => Math.min(current, Math.max(frames.length - 1, 0)));
  }, [frames.length]);

  const baseIndex = Math.floor(progress);
  const localTween = progress - baseIndex;
  const easedTween = smootherStep(localTween);
  const currentFrame = frames[baseIndex];
  const nextFrame = frames[Math.min(baseIndex + 1, frames.length - 1)] ?? currentFrame;

  useEffect(() => {
    setHoveredClusterId(null);
  }, [interactionMode, playing, baseIndex]);
  const axisExtents = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    frames.forEach((frame) => {
      frame.points.forEach((point) => {
        const xv = point.x;
        const yv = point.y;
        if (typeof xv === "number" && Number.isFinite(xv)) {
          xs.push(xv);
        }
        if (typeof yv === "number" && Number.isFinite(yv)) {
          ys.push(yv);
        }
      });
    });
    return computeEqualScaleScatterExtents(xs, ys);
  }, [frames]);
  const temporalSigma = 1.05 + trailStrength * 0.7;
  const temporalWindow = Math.max(3, Math.round(2 + trailStrength * 3));
  const glowScale = 0.8 + glowStrength * 0.6;

  const buildDensityGrid = (
    points: ScatterFrame["points"],
    extents: { xMin: number; xMax: number; yMin: number; yMax: number },
  ) => {
    const grid = new Array<number>(DENSITY_BINS * DENSITY_BINS).fill(0);
    const xSpan = Math.max(1e-6, extents.xMax - extents.xMin);
    const ySpan = Math.max(1e-6, extents.yMax - extents.yMin);
    points.forEach((point) => {
      const x = Number(point.x);
      const y = Number(point.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return;
      }
      const nx = (x - extents.xMin) / xSpan;
      const ny = (y - extents.yMin) / ySpan;
      if (nx < 0 || nx > 1 || ny < 0 || ny > 1) {
        return;
      }
      const ix = Math.max(0, Math.min(DENSITY_BINS - 1, Math.floor(nx * DENSITY_BINS)));
      const iy = Math.max(0, Math.min(DENSITY_BINS - 1, Math.floor(ny * DENSITY_BINS)));
      grid[iy * DENSITY_BINS + ix] += 1;
    });
    return grid;
  };

  const densityGrids = useMemo(
    () => frames.map((frame) => blurGrid(buildDensityGrid(frame.points, axisExtents), DENSITY_BINS, 4)),
    [axisExtents, frames],
  );

  const densityData = useMemo(() => {
    if (!currentFrame || !densityGrids.length) {
      return [] as Array<[number, number, number]>;
    }
    const blended = new Array<number>(DENSITY_BINS * DENSITY_BINS).fill(0);
    const totalWeights = new Array<number>(DENSITY_BINS * DENSITY_BINS).fill(0);
    const start = Math.max(0, baseIndex - temporalWindow);
    const end = Math.min(frames.length - 1, baseIndex + temporalWindow);
    for (let frameIdx = start; frameIdx <= end; frameIdx += 1) {
      const dist = Math.abs(frameIdx - progress);
      const weight = temporalWeight(dist, temporalSigma);
      const grid = densityGrids[frameIdx];
      for (let idxCell = 0; idxCell < grid.length; idxCell += 1) {
        blended[idxCell] += grid[idxCell] * weight;
        totalWeights[idxCell] += weight;
      }
    }
    const xStep = (axisExtents.xMax - axisExtents.xMin) / DENSITY_BINS;
    const yStep = (axisExtents.yMax - axisExtents.yMin) / DENSITY_BINS;
    const rows: Array<[number, number, number]> = [];
    for (let iy = 0; iy < DENSITY_BINS; iy += 1) {
      for (let ix = 0; ix < DENSITY_BINS; ix += 1) {
        const idxCell = iy * DENSITY_BINS + ix;
        const v = totalWeights[idxCell] > 0 ? blended[idxCell] / totalWeights[idxCell] : 0;
        if (v <= 0.01) {
          continue;
        }
        const x = axisExtents.xMin + (ix + 0.5) * xStep;
        const y = axisExtents.yMin + (iy + 0.5) * yStep;
        rows.push([x, y, Math.pow(v, 0.72 - glowStrength * 0.08)]);
      }
    }
    return rows;
  }, [axisExtents, baseIndex, currentFrame, densityGrids, frames.length, glowStrength, progress, temporalSigma, temporalWindow]);

  const densityMax = useMemo(() => {
    if (!densityData.length) {
      return 1;
    }
    const values = densityData.map((row) => row[2]).sort((a, b) => a - b);
    const percentile = 0.93 - glowStrength * 0.05;
    const idx = Math.min(values.length - 1, Math.floor(values.length * percentile));
    return Math.max(values[idx] ?? values[values.length - 1] ?? 1, 0.35);
  }, [densityData]);

  const visiblePoints = useMemo(() => {
    const aggregates = new Map<
      string,
      {
        point: ScatterFrame["points"][number];
        weight: number;
        xSum: number;
        ySum: number;
        alpha: number;
      }
    >();
    const start = Math.max(0, baseIndex - temporalWindow);
    const end = Math.min(frames.length - 1, baseIndex + temporalWindow);
    for (let frameIdx = start; frameIdx <= end; frameIdx += 1) {
      const frame = frames[frameIdx];
      const weight = temporalWeight(Math.abs(frameIdx - progress), temporalSigma);
      frame.points.forEach((point) => {
        const x = Number(point.x);
        const y = Number(point.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          return;
        }
        const key = String(point.productId);
        const existing = aggregates.get(key) ?? {
          point,
          weight: 0,
          xSum: 0,
          ySum: 0,
          alpha: 0,
        };
        existing.weight += weight;
        existing.xSum += x * weight;
        existing.ySum += y * weight;
        existing.alpha = Math.max(existing.alpha, weight * (point.isGhost ? 0.45 : 1));
        if (weight >= existing.weight * 0.5) {
          existing.point = point;
        }
        aggregates.set(key, existing);
      });
    }
    return Array.from(aggregates.values()).map((entry) => ({
      ...entry.point,
      x: entry.weight > 0 ? entry.xSum / entry.weight : entry.point.x,
      y: entry.weight > 0 ? entry.ySum / entry.weight : entry.point.y,
      _alpha: Math.max(0.025, Math.min(0.16, entry.alpha * 0.12)),
    }));
  }, [baseIndex, frames, progress, temporalSigma, temporalWindow]);

  const groups = useMemo(() => {
    const grouped = new Map<string, Array<ScatterFrame["points"][number] & { _alpha?: number }>>();
    visiblePoints.forEach((point) => {
      const key = point.lifecycleState ?? point.movementGroup ?? "unknown";
      const existing = grouped.get(key) ?? [];
      existing.push(point);
      grouped.set(key, existing);
    });
    return Array.from(grouped.entries());
  }, [visiblePoints]);
  const currentFrameSummary = useMemo(() => {
    if (!currentFrame) {
      return null;
    }
    return {
      label: currentFrame.label,
      snapshotId: currentFrame.snapshotId,
      newCount: currentFrame.newCount ?? 0,
      retainedCount: currentFrame.retainedCount ?? 0,
      exitedCount: currentFrame.exitedCount ?? 0,
      pointCount: currentFrame.points.length,
      topClusters: (currentFrame.clusterShareTopN ?? []).slice(0, 3),
    };
  }, [currentFrame]);

  const currentFramePoints = useMemo(
    () => (currentFrame?.points ?? []).filter((point) => !point.isGhost),
    [currentFrame],
  );

  const clusterRegions = useMemo(() => {
    if (!currentFrameSummary?.topClusters.length) {
      return [] as Array<{
        clusterId: string;
        centroid: [number, number];
        hull: Array<[number, number]>;
        radius: number;
        dominantCategory: string;
        pointCount: number;
        sharePct: number;
        dominantSharePct: number;
      }>;
    }
    return currentFrameSummary.topClusters.flatMap((cluster) => {
      const points = currentFramePoints
        .filter((point) => String(point.clusterId ?? "") === cluster.clusterId)
        .map((point) => [Number(point.x), Number(point.y)] as [number, number])
        .filter((point) => Number.isFinite(point[0]) && Number.isFinite(point[1]));
      if (points.length < 3) {
        return [];
      }
      const hull = buildConvexHull(points);
      const centroid = hull.reduce<[number, number]>((acc, point) => [acc[0] + point[0], acc[1] + point[1]], [0, 0]);
      const center: [number, number] = [centroid[0] / hull.length, centroid[1] / hull.length];
      const radius = Math.max(
        ...points.map((point) => Math.hypot(point[0] - center[0], point[1] - center[1])),
        0.05,
      );
      return [{
        clusterId: cluster.clusterId,
        centroid: center,
        hull,
        radius: radius * 2.2,
        dominantCategory: cluster.dominantCategory,
        pointCount: cluster.pointCount,
        sharePct: cluster.sharePct,
        dominantSharePct: cluster.dominantSharePct,
      }];
    });
  }, [currentFramePoints, currentFrameSummary]);

  const hotspotSummaries = useMemo(() => {
    if (!densityData.length || !currentFramePoints.length) {
      return [] as Array<{
        id: string;
        center: [number, number];
        radius: number;
        intensity: number;
        dominantClusterId: string | null;
        dominantCategory: string;
        retainedSharePct: number;
        avgRank: number | null;
        points: Array<ScatterFrame["points"][number]>;
        samples: Array<ScatterFrame["points"][number]>;
      }>;
    }
    const xSpan = axisExtents.xMax - axisExtents.xMin;
    const ySpan = axisExtents.yMax - axisExtents.yMin;
    const minDistance = Math.max(xSpan, ySpan) * 0.09;
    const sortedCells = [...densityData].sort((left, right) => right[2] - left[2]);
    const peaks: Array<[number, number, number]> = [];
    sortedCells.forEach((cell) => {
      if (peaks.length >= 8) {
        return;
      }
      const tooClose = peaks.some((peak) => Math.hypot(peak[0] - cell[0], peak[1] - cell[1]) < minDistance);
      if (!tooClose) {
        peaks.push(cell);
      }
    });

    return peaks.flatMap((peak, index) => {
      const radius = Math.max(xSpan, ySpan) * (0.06 + peak[2] * 0.03);
      const nearby = currentFramePoints.filter((point) => {
        const x = Number(point.x);
        const y = Number(point.y);
        return Number.isFinite(x) && Number.isFinite(y) && Math.hypot(x - peak[0], y - peak[1]) <= radius;
      });
      if (nearby.length < 3) {
        return [];
      }
      const clusterCounts = new Map<string, number>();
      const categoryCounts = new Map<string, number>();
      let retainedCount = 0;
      let rankSum = 0;
      let rankCount = 0;
      nearby.forEach((point) => {
        const clusterId = String(point.clusterId ?? "");
        if (clusterId) {
          clusterCounts.set(clusterId, (clusterCounts.get(clusterId) ?? 0) + 1);
        }
        const category = String(point.dominantCategory ?? "미분류");
        categoryCounts.set(category, (categoryCounts.get(category) ?? 0) + 1);
        if (point.lifecycleState === "retained") {
          retainedCount += 1;
        }
        if (typeof point.rank === "number" && Number.isFinite(point.rank)) {
          rankSum += point.rank;
          rankCount += 1;
        }
      });
      const dominantClusterId = Array.from(clusterCounts.entries()).sort((left, right) => right[1] - left[1])[0]?.[0] ?? null;
      const dominantCategory = Array.from(categoryCounts.entries()).sort((left, right) => right[1] - left[1])[0]?.[0] ?? "미분류";
      const samples = [...nearby]
        .sort((left, right) => {
          const leftDistance = Math.hypot(Number(left.x) - peak[0], Number(left.y) - peak[1]);
          const rightDistance = Math.hypot(Number(right.x) - peak[0], Number(right.y) - peak[1]);
          if (leftDistance !== rightDistance) {
            return leftDistance - rightDistance;
          }
          return Number(left.rank ?? 999) - Number(right.rank ?? 999);
        })
        .filter((point, sampleIndex, array) => array.findIndex((candidate) => candidate.productId === point.productId) === sampleIndex)
        .slice(0, 3);
      return [{
        id: `hotspot-${index + 1}`,
        center: [peak[0], peak[1]] as [number, number],
        radius,
        intensity: peak[2],
        dominantClusterId,
        dominantCategory,
        retainedSharePct: nearby.length ? (retainedCount / nearby.length) * 100 : 0,
        avgRank: rankCount ? rankSum / rankCount : null,
        points: nearby,
        samples,
      }];
    });
  }, [axisExtents.xMax, axisExtents.xMin, axisExtents.yMax, axisExtents.yMin, currentFramePoints, densityData]);

  useEffect(() => {
    if (interactionMode !== "hotspot" || playing) {
      return;
    }
    if (!hotspotSummaries.length) {
      setSelectedHotspotId(null);
      return;
    }
    const exists = hotspotSummaries.some((hotspot) => hotspot.id === selectedHotspotId);
    if (!exists) {
      setSelectedHotspotId(hotspotSummaries[0].id);
    }
  }, [hotspotSummaries, interactionMode, playing, selectedHotspotId]);

  const activeClusterId = interactionMode === "cluster" ? hoveredClusterId : null;
  const activeHotspot = hotspotSummaries.find((hotspot) => hotspot.id === selectedHotspotId)
    ?? hotspotSummaries[0]
    ?? null;

  const resolveHotspotIdFromPoint = (point: { x?: number | null; y?: number | null }) => {
    const x = Number(point.x);
    const y = Number(point.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return null;
    }
    const containing = hotspotSummaries
      .map((hotspot) => ({
        hotspot,
        distance: Math.hypot(x - hotspot.center[0], y - hotspot.center[1]),
      }))
      .filter((entry) => entry.distance <= entry.hotspot.radius)
      .sort((left, right) => left.distance - right.distance);
    if (containing.length) {
      return containing[0].hotspot.id;
    }
    const nearest = hotspotSummaries
      .map((hotspot) => ({
        hotspot,
        distance: Math.hypot(x - hotspot.center[0], y - hotspot.center[1]),
      }))
      .sort((left, right) => left.distance - right.distance)[0];
    return nearest?.hotspot.id ?? null;
  };

  const extractPointFromEvent = (params: unknown): { x?: number | null; y?: number | null } | null => {
    const data = (params as { data?: unknown } | undefined)?.data;
    if (data && typeof data === "object" && "meta" in data) {
      const meta = (data as { meta?: { x?: number | null; y?: number | null } }).meta;
      if (meta) {
        return meta;
      }
    }
    const value = (params as { value?: unknown } | undefined)?.value;
    if (Array.isArray(value) && value.length >= 2) {
      return {
        x: Number(value[0]),
        y: Number(value[1]),
      };
    }
    return null;
  };

  if (!frames.length || !currentFrame) {
    return <div className="empty-state">임베딩 투영 데이터가 없습니다.</div>;
  }

  const option = {
    backgroundColor: {
      type: "radial",
      x: 0.5,
      y: 0.45,
      r: 0.95,
      colorStops: [
        { offset: 0, color: "rgba(16,24,46,0.99)" },
        { offset: 0.48, color: "rgba(8,12,24,0.985)" },
        { offset: 1, color: "rgba(2,3,8,1)" },
      ],
    },
    animation: false,
    grid: { left: 48, right: 24, top: 56, bottom: 44 },
    tooltip: {
      trigger: "item",
      triggerOn: "click",
      renderMode: "html",
      appendToBody: true,
      confine: false,
      extraCssText: "max-width: 320px; white-space: normal;",
      backgroundColor: "rgba(12,12,14,0.96)",
      borderColor: "rgba(255,255,255,0.08)",
      textStyle: { color: "#d4d4d8" },
      formatter: (params: { seriesType?: string; data?: { meta?: ScatterFrame["points"][number] & Record<string, unknown> } | [number, number, number] }) => {
        const meta = (params.data as { meta?: Record<string, unknown> } | undefined)?.meta;
        if (meta?.__overlayKind === "cluster-region") {
          return [
            `<strong>Cluster ${escapeHtml(String(meta.clusterId ?? "-"))}</strong>`,
            `지배 카테고리: ${escapeHtml(String(meta.dominantCategory ?? "-"))}`,
            `현재 프레임 점유율: ${formatNumber(meta.sharePct ?? "-", "percent")}`,
            `지배 비중: ${formatNumber(meta.dominantSharePct ?? "-", "percent")}`,
          ].join("<br/>");
        }
        if (meta?.__overlayKind === "hotspot") {
          return [
            `<strong>밝은 영역 후보</strong>`,
            `대표 카테고리: ${escapeHtml(String(meta.dominantCategory ?? "-"))}`,
            `대표 클러스터: ${escapeHtml(String(meta.dominantClusterId ?? "-"))}`,
            `유지 비중: ${formatNumber(meta.retainedSharePct ?? "-", "percent")}`,
            `대표 사례: ${formatNumber(meta.sampleCount ?? "-", "integer")}개`,
          ].join("<br/>");
        }
        if (params.seriesType === "scatter" && Array.isArray(params.data) && params.data.length >= 3) {
          const value = Array.isArray(params.data) ? params.data[2] : null;
          return `최근 프레임 응집도: ${formatNumber(value ?? "-", "number")}`;
        }
        const point = (params.data as { meta?: ScatterFrame["points"][number] } | undefined)?.meta;
        if (!point) {
          return "";
        }
        const title = String(point.name ?? point.productId);
        const mainImage =
          typeof point.mainImage === "string" && point.mainImage.trim() ? point.mainImage.trim() : null;
        const imgHtml = mainImage
          ? `<br/><img src="${productImageApiUrl(mainImage)}" alt="" style="max-width:168px;max-height:168px;display:block;margin-top:8px;border-radius:6px;object-fit:contain;" />`
          : "";
        const mainNote = mainImage ? "" : `<span style="color:#a1a1aa;">메인 이미지 없음</span>`;
        return [
          `<strong>${escapeHtml(title)}</strong>`,
          `브랜드: ${escapeHtml(String(point.brand ?? "-"))}`,
          `순위: ${formatNumber(point.rank ?? "-", "integer")}`,
          `순위 변화: ${formatNumber(point.rankVelocity ?? "-", "number")}`,
          `상태: ${escapeHtml(movementLabelMap[point.lifecycleState ?? ""] ?? (point.lifecycleState ?? "-"))}`,
          `클러스터: ${escapeHtml(String(point.clusterId ?? "-"))}`,
          `지배 카테고리: ${escapeHtml(String(point.dominantCategory ?? "-"))}`,
          ...(mainNote ? [mainNote] : []),
          ...(imgHtml ? [imgHtml] : []),
        ].join("<br/>");
      },
    },
    legend: {
      top: 0,
      textStyle: { color: "#d4d4d8" },
      data: groups.map(([name]) => movementLabelMap[name] ?? name),
    },
    visualMap: {
      show: false,
      min: 0,
      max: densityMax,
      calculable: false,
      seriesIndex: [0, 1, 2],
      inRange: {
        color: [
          "rgba(10,18,56,0.03)",
          "rgba(43,74,255,0.18)",
          "rgba(70,210,255,0.48)",
          "rgba(176,88,255,0.72)",
          "rgba(255,214,102,0.88)",
          "rgba(255,245,210,1)",
        ],
      },
    },
    xAxis: {
      type: "value",
      name: "projection-x",
      min: axisExtents.xMin,
      max: axisExtents.xMax,
      nameTextStyle: { color: "#d4d4d8" },
      axisLabel: { color: "#d4d4d8", formatter: (value: number) => formatNumber(value, "number") },
      axisLine: { lineStyle: { color: "rgba(255,255,255,0.10)" } },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.03)" } },
    },
    yAxis: {
      type: "value",
      name: "projection-y",
      min: axisExtents.yMin,
      max: axisExtents.yMax,
      nameTextStyle: { color: "#d4d4d8" },
      axisLabel: { color: "#d4d4d8", formatter: (value: number) => formatNumber(value, "number") },
      axisLine: { lineStyle: { color: "rgba(255,255,255,0.10)" } },
      splitLine: { lineStyle: { color: "rgba(255,255,255,0.03)" } },
    },
    series: [
      {
        name: "__density_outer__",
        type: "scatter",
        data: densityData,
        symbolSize: (value: number[]) => {
          const intensity = Number(value?.[2] ?? 0);
          return Math.max(42, Math.min(108, intensity * 30 * glowScale));
        },
        itemStyle: {
          opacity: 0.08 + glowStrength * 0.06,
        },
        encode: {
          x: 0,
          y: 1,
          value: 2,
        },
        silent: true,
        blendMode: "lighter",
        z: 1,
      },
      {
        name: "__density_mid__",
        type: "scatter",
        data: densityData,
        symbolSize: (value: number[]) => {
          const intensity = Number(value?.[2] ?? 0);
          return Math.max(24, Math.min(76, intensity * 18 * glowScale));
        },
        itemStyle: {
          opacity: 0.18 + glowStrength * 0.14,
        },
        encode: {
          x: 0,
          y: 1,
          value: 2,
        },
        silent: true,
        blendMode: "lighter",
        z: 1.2,
      },
      {
        name: "__density_core__",
        type: "scatter",
        data: densityData,
        symbolSize: (value: number[]) => {
          const intensity = Number(value?.[2] ?? 0);
          return Math.max(10, Math.min(34, intensity * 10 * glowScale));
        },
        itemStyle: {
          opacity: 0.45 + glowStrength * 0.2,
        },
        encode: {
          x: 0,
          y: 1,
          value: 2,
        },
        silent: true,
        blendMode: "lighter",
        z: 1.4,
      },
      ...(!playing && interactionMode === "cluster" ? clusterRegions.map((region) => ({
        name: `cluster-region-${region.clusterId}`,
        type: "line",
        data: [...region.hull, region.hull[0]],
        smooth: 0.42,
        silent: true,
        showSymbol: false,
        lineStyle: {
          color: activeClusterId === region.clusterId ? "rgba(250, 204, 21, 0.96)" : "rgba(226, 232, 240, 0.52)",
          width: activeClusterId === region.clusterId ? 3.2 : 2.1,
        },
        areaStyle: {
          color: activeClusterId === region.clusterId ? "rgba(250, 204, 21, 0.2)" : "rgba(148, 163, 184, 0.085)",
        },
        z: 1.55,
      })) : []),
      ...(!playing && interactionMode === "cluster" ? clusterRegions.map((region) => ({
        name: `cluster-region-hit-${region.clusterId}`,
        type: "scatter",
        symbolSize: Math.max(52, Math.min(168, region.radius * 190)),
        data: [{
          value: region.centroid,
          meta: {
            __overlayKind: "cluster-region",
            clusterId: region.clusterId,
            dominantCategory: region.dominantCategory,
            sharePct: region.sharePct,
            dominantSharePct: region.dominantSharePct,
          },
          itemStyle: {
            color: activeClusterId === region.clusterId ? "rgba(250, 204, 21, 0.16)" : "rgba(255,255,255,0.06)",
            borderColor: activeClusterId === region.clusterId ? "rgba(250, 204, 21, 0.82)" : "rgba(255,255,255,0.24)",
            borderWidth: activeClusterId === region.clusterId ? 2.2 : 1.4,
          },
          label: {
            show: true,
            position: "top",
            distance: 8,
            formatter: `${region.dominantCategory}\n${region.clusterId}`,
            color: "#f4f4f5",
            fontSize: activeClusterId === region.clusterId ? 12 : 11,
            fontWeight: activeClusterId === region.clusterId ? 700 : 600,
            backgroundColor: activeClusterId === region.clusterId ? "rgba(113, 63, 18, 0.92)" : "rgba(24, 24, 27, 0.84)",
            borderColor: activeClusterId === region.clusterId ? "rgba(250, 204, 21, 0.72)" : "rgba(255,255,255,0.14)",
            borderWidth: 1,
            borderRadius: 10,
            padding: [4, 8, 4, 8],
            lineHeight: 16,
          },
        }],
        z: 1.6,
      })) : []),
      ...(!playing && interactionMode === "hotspot" ? hotspotSummaries.map((hotspot) => ({
        name: hotspot.id,
        type: "scatter",
        silent: true,
        tooltip: { show: false },
        symbolSize: Math.max(64, Math.min(190, hotspot.radius * 225)),
        data: [{
          value: hotspot.center,
          itemStyle: {
            color: selectedHotspotId === hotspot.id ? "rgba(0, 0, 0, 0.74)" : "rgba(0, 0, 0, 0.56)",
            borderColor: selectedHotspotId === hotspot.id ? "rgba(255, 245, 210, 0.98)" : "rgba(255, 255, 255, 0.82)",
            borderWidth: selectedHotspotId === hotspot.id ? 3 : 2.2,
          },
        }],
        z: 1.7,
      })) : []),
      ...(!playing && interactionMode === "hotspot" ? hotspotSummaries.flatMap((hotspot) =>
        hotspot.samples.map((sample, index) => ({
          name: `hotspot-sample-${hotspot.id}-${sample.productId}`,
          type: "scatter",
          symbolSize: activeHotspot?.id === hotspot.id ? (index === 0 ? 20 : 16) : 12,
          data: [{
            value: [Number(sample.x), Number(sample.y)],
            name: sample.name ?? sample.productId,
            meta: { ...sample, __overlayKind: "hotspot-sample", hotspotId: hotspot.id },
            itemStyle: activeHotspot?.id === hotspot.id
              ? {
                  color: index === 0 ? "rgba(255, 245, 210, 0.96)" : "rgba(191, 219, 254, 0.92)",
                  borderColor: index === 0 ? "rgba(245, 158, 11, 0.95)" : "rgba(96, 165, 250, 0.9)",
                  borderWidth: index === 0 ? 3 : 2,
                  shadowBlur: index === 0 ? 18 : 10,
                  shadowColor: index === 0 ? "rgba(245, 158, 11, 0.35)" : "rgba(96, 165, 250, 0.28)",
                }
              : {
                  color: "rgba(255, 255, 255, 0.7)",
                  borderColor: "rgba(255, 255, 255, 0.5)",
                  borderWidth: 1.5,
                },
          }],
          z: 2.4,
        }))
      ) : []),
      ...groups.map(([name, points]) => ({
        name: movementLabelMap[name] ?? name,
        type: "scatter",
        silent: !playing,
        symbolSize: (value: unknown, params: { data?: { meta?: ScatterFrame["points"][number] } }) => {
          const point = params.data?.meta;
          if (!point) {
            return 4;
          }
          const rank = Number(point.rank ?? 100);
          return Math.max(3, Math.min(8, 10 - Math.log2(Math.max(rank, 1))));
        },
        itemStyle: {
          color: movementPalette[name] ?? movementPalette.unknown,
          opacity: name === "exited" ? 0.08 : 0.18,
        },
        data: points
          .map((point) => {
            const x = Number(point.x);
            const y = Number(point.y);
            if (!Number.isFinite(x) || !Number.isFinite(y)) {
              return null;
            }
            return {
              value: [x, y],
              name: point.name ?? point.productId,
              meta: point,
              itemStyle: {
                opacity: Math.max(0.03, Math.min(0.16, (point._alpha ?? 0.2) * 0.2)),
              },
            };
          })
          .filter((row): row is { value: [number, number]; name: string; meta: ScatterFrame["points"][number]; itemStyle: { opacity: number } } => Boolean(row)),
        z: 2,
      })),
    ],
  };

  const chartEvents = {
    mouseover: (params: unknown) => {
      const meta = (params as { data?: { meta?: Record<string, unknown> } } | undefined)?.data?.meta;
      if (!meta) {
        return;
      }
      if (interactionMode === "cluster" && meta.__overlayKind === "cluster-region") {
        const nextId = String(meta.clusterId ?? "");
        setHoveredClusterId((prev) => (prev === nextId ? prev : nextId));
      }
    },
    click: (params: unknown) => {
      if (interactionMode === "hotspot") {
        const meta = (params as { data?: { meta?: Record<string, unknown> } } | undefined)?.data?.meta;
        if (meta?.__overlayKind === "hotspot-sample") {
          setSelectedHotspotId(String(meta.hotspotId ?? ""));
          return;
        }
        const point = extractPointFromEvent(params);
        if (point) {
          const hotspotId = resolveHotspotIdFromPoint(point);
          if (hotspotId) {
            setSelectedHotspotId(hotspotId);
          }
        }
      }
    },
    mouseout: () => {
      setHoveredClusterId(null);
    },
    globalout: () => {
      setHoveredClusterId(null);
    },
  };

  return (
    <div>
      <div className="playback-row">
        <button type="button" onClick={() => setPlaying((value) => !value)}>
          {playing ? "일시정지" : "재생"}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(frames.length - 1, 0)}
          value={progress}
          step={0.01}
          onChange={(event) => {
            setProgress(Number(event.target.value));
          }}
        />
        <span>{currentFrame.label}</span>
        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#a1a1aa" }}>
          속도
          <input
            type="range"
            min={0.8}
            max={1.8}
            step={0.05}
            value={speedMultiplier}
            onChange={(event) => setSpeedMultiplier(Number(event.target.value))}
            style={{ width: 92 }}
          />
          <span>{speedMultiplier.toFixed(2)}x</span>
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#a1a1aa" }}>
          광량
          <input
            type="range"
            min={0.2}
            max={1}
            step={0.05}
            value={glowStrength}
            onChange={(event) => setGlowStrength(Number(event.target.value))}
            style={{ width: 92 }}
          />
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#a1a1aa" }}>
          잔광
          <input
            type="range"
            min={0.2}
            max={1}
            step={0.05}
            value={trailStrength}
            onChange={(event) => setTrailStrength(Number(event.target.value))}
            style={{ width: 92 }}
          />
        </label>
        <span style={{ color: "#71717a", fontSize: 12 }}>
          nebula mode
        </span>
      </div>
      {!playing ? (
        <div className="animated-scatter-mode-toggle">
          <button
            type="button"
            className={interactionMode === "cluster" ? "is-active" : ""}
            onClick={() => setInteractionMode("cluster")}
          >
            클러스터 관측
          </button>
          <button
            type="button"
            className={interactionMode === "hotspot" ? "is-active" : ""}
            onClick={() => setInteractionMode("hotspot")}
          >
            핫스팟 관측
          </button>
        </div>
      ) : null}
      {currentFrameSummary ? (
        <div className="animated-scatter-summary">
          <div className="animated-scatter-summary__headline">
            <div>
              <small>현재 프레임</small>
              <strong>{currentFrameSummary.label}</strong>
            </div>
            <span>{currentFrameSummary.snapshotId}</span>
          </div>
          <div className="animated-scatter-summary__stats">
            <div><span>신규</span><strong>{formatNumber(currentFrameSummary.newCount, "integer")}</strong></div>
            <div><span>유지</span><strong>{formatNumber(currentFrameSummary.retainedCount, "integer")}</strong></div>
            <div><span>이탈</span><strong>{formatNumber(currentFrameSummary.exitedCount, "integer")}</strong></div>
            <div><span>표시 포인트</span><strong>{formatNumber(currentFrameSummary.pointCount, "integer")}</strong></div>
          </div>
          {currentFrameSummary.topClusters.length ? (
            <div className="animated-scatter-summary__clusters">
              {currentFrameSummary.topClusters.map((cluster) => (
                <div
                  key={`${currentFrameSummary.snapshotId}-${cluster.clusterId}`}
                  className={`animated-scatter-summary__cluster${interactionMode === "cluster" && activeClusterId === cluster.clusterId ? " is-highlighted" : ""}`}
                  onMouseEnter={() => {
                    if (!playing && interactionMode === "cluster") {
                      setHoveredClusterId(cluster.clusterId);
                    }
                  }}
                  onMouseLeave={() => {
                    if (interactionMode === "cluster") {
                      setHoveredClusterId(null);
                    }
                  }}
                >
                  <strong>Cluster {cluster.clusterId}</strong>
                  <span>{cluster.dominantCategory}</span>
                  <small>
                    점유율 {formatNumber(cluster.sharePct, "percent")} · 지배 비중 {formatNumber(cluster.dominantSharePct, "percent")}
                  </small>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
      {!playing && interactionMode === "hotspot" ? (
        <div className="animated-scatter-hotspot-panel">
          <div className="animated-scatter-hotspot-panel__header">
            <div>
              <small>밝은 영역 해석</small>
              <strong>{activeHotspot ? "순위권에서 반복 응집한 시각 스타일" : "정지 후 밝은 영역을 클릭해 대표 사례를 확인하세요"}</strong>
            </div>
            {activeHotspot ? (
              <span>
                핫스팟 {hotspotSummaries.length}개 관측
                {" · "}
                유지 비중 {formatNumber(activeHotspot.retainedSharePct, "percent")}
                {activeHotspot.avgRank != null ? ` · 평균 순위 ${formatNumber(activeHotspot.avgRank, "integer")}위` : ""}
              </span>
            ) : null}
          </div>
          {activeHotspot ? (
            <>
              <div className="animated-scatter-hotspot-panel__list">
                {hotspotSummaries.map((hotspot, index) => (
                  <button
                    key={hotspot.id}
                    type="button"
                    className={`animated-scatter-hotspot-panel__spot${activeHotspot.id === hotspot.id ? " is-active" : ""}`}
                    onClick={() => setSelectedHotspotId(hotspot.id)}
                  >
                    <strong>핫스팟 {index + 1}</strong>
                    <span>
                      x {formatNumber(hotspot.center[0], "number")} · y {formatNumber(hotspot.center[1], "number")}
                    </span>
                    <small>
                      {hotspot.dominantCategory} · {hotspot.dominantClusterId ?? "-"} · 사례 {formatNumber(hotspot.samples.length, "integer")}개
                    </small>
                  </button>
                ))}
              </div>
              <div className="animated-scatter-hotspot-panel__facts">
                <div><span>대표 클러스터</span><strong>{activeHotspot.dominantClusterId ?? "-"}</strong></div>
                <div><span>대표 카테고리</span><strong>{activeHotspot.dominantCategory}</strong></div>
                <div><span>관측 점 수</span><strong>{formatNumber(activeHotspot.points.length, "integer")}</strong></div>
                <div><span>대표 사례</span><strong>{formatNumber(activeHotspot.samples.length, "integer")}개</strong></div>
              </div>
              <div className="animated-scatter-hotspot-panel__samples">
                {activeHotspot.samples.map((sample, index) => (
                  <div
                    key={`${activeHotspot.id}-${sample.productId}`}
                    className={`animated-scatter-hotspot-panel__sample${index === 0 ? " is-primary" : ""}`}
                  >
                    <div className="animated-scatter-hotspot-panel__sample-image">
                      {sample.mainImage ? <img src={productImageApiUrl(sample.mainImage)} alt="" /> : <span>이미지 없음</span>}
                    </div>
                    <div className="animated-scatter-hotspot-panel__sample-body">
                      <strong>{index === 0 ? `대표 사례 · ${sample.name ?? sample.productId}` : sample.name ?? sample.productId}</strong>
                      <span>{sample.brand ?? "-"}</span>
                      <small>
                        순위 {formatNumber(sample.rank ?? "-", "integer")} · 상태 {movementLabelMap[sample.lifecycleState ?? ""] ?? (sample.lifecycleState ?? "-")}
                      </small>
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="animated-scatter-hotspot-panel__empty">
              밝은 halo는 최근 프레임 전후에서 비슷한 상품이 반복적으로 응집한 구간입니다. 정지한 뒤 밝은 영역을 클릭하면 대표 사례를 볼 수 있습니다.
            </p>
          )}
        </div>
      ) : null}
      <ScatterSquareChart option={option} onEvents={chartEvents} />
    </div>
  );
}
