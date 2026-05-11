import { useLayoutEffect, useMemo, useRef, useState } from "react";

import ReactECharts, { type EChartsOption } from "echarts-for-react";

type ScatterSquareChartProps = {
  option: EChartsOption;
  onEvents?: Record<string, (params: unknown) => void>;
  /** 플롯 한 변 최대값으로 쓸 뷰포트 높이 비율 (정사각형이 세로로 과도하게 커지지 않게) */
  maxHeightRatio?: number;
};

const MIN_SIDE = 240;

export function ScatterSquareChart({ option, onEvents, maxHeightRatio = 0.88 }: ScatterSquareChartProps) {
  const outerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<InstanceType<typeof ReactECharts>>(null);
  const [innerSize, setInnerSize] = useState(() =>
    typeof window !== "undefined" ? Math.min(Math.floor(window.innerWidth * 0.5), Math.floor(window.innerHeight * maxHeightRatio)) : 400,
  );
  const dpr = useMemo(() => {
    if (typeof window === "undefined") {
      return 1;
    }
    return Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  }, []);

  useLayoutEffect(() => {
    const el = outerRef.current;
    if (!el) {
      return undefined;
    }
    const measure = () => {
      const w = el.getBoundingClientRect().width;
      const cap = window.innerHeight * maxHeightRatio;
      const side = Math.max(MIN_SIDE, Math.floor(Math.min(w, cap)));
      setInnerSize((prev) => (prev === side ? prev : side));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [maxHeightRatio]);

  useLayoutEffect(() => {
    const inst = chartRef.current?.getEchartsInstance();
    if (!inst) {
      return undefined;
    }
    const rafId = requestAnimationFrame(() => {
      if (!inst.isDisposed()) {
        inst.resize();
      }
    });
    return () => cancelAnimationFrame(rafId);
  }, [innerSize]);

  return (
    <div ref={outerRef} className="chart-scatter-square-fill">
      <div className="chart-scatter-square-inner" style={{ width: innerSize, height: innerSize }}>
        <ReactECharts
          ref={chartRef}
          option={option}
          onEvents={onEvents}
          notMerge
          lazyUpdate
          opts={{ renderer: "canvas", devicePixelRatio: dpr }}
          style={{ width: "100%", height: "100%" }}
        />
      </div>
    </div>
  );
}
