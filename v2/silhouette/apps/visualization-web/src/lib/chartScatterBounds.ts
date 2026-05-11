/** x·y 축 구간 길이를 동일하게 유지하면서, 데이터 직사각형을 덮는 최소 정사각형 도메인(축별 중심)을 만든다. */
export type EqualScaleScatterExtents = { xMin: number; xMax: number; yMin: number; yMax: number };

export function computeEqualScaleScatterExtents(
  xs: number[],
  ys: number[],
  paddingRatio = 0.03,
): EqualScaleScatterExtents {
  const xnums = xs.filter((n) => typeof n === "number" && Number.isFinite(n));
  const ynums = ys.filter((n) => typeof n === "number" && Number.isFinite(n));
  if (!xnums.length || !ynums.length) {
    return { xMin: -1, xMax: 1, yMin: -1, yMax: 1 };
  }
  const xmin = Math.min(...xnums);
  const xmax = Math.max(...xnums);
  const ymin = Math.min(...ynums);
  const ymax = Math.max(...ynums);
  const w = Math.max(xmax - xmin, 1e-12);
  const h = Math.max(ymax - ymin, 1e-12);
  const side = Math.max(w, h) * (1 + 2 * paddingRatio);
  const cx = (xmin + xmax) / 2;
  const cy = (ymin + ymax) / 2;
  return {
    xMin: cx - side / 2,
    xMax: cx + side / 2,
    yMin: cy - side / 2,
    yMax: cy + side / 2,
  };
}
