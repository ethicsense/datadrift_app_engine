import React, { useEffect, useState } from "react";
import CardRenderer from "./visualization/CardRenderer";
import { downloadReportFile } from "../utils/reportDownload";

function sanitizeReportBasename(name, fallbackId) {
  const raw = name || fallbackId || "dataset";
  return (
    String(raw)
      .replace(/[/\\?*:|"<>]/g, "_")
      .replace(/\.[^/.]+$/, "")
      .replace(/\s+/g, "_")
      .slice(0, 120) || "dataset"
  );
}

export default function DriftStudio({ backend, baseDataset, targetDataset, onBack }) {
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [status, setStatus] = useState(null);
  const [reportBusy, setReportBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    let pollTimer = null;
    if (!baseDataset?.id || !targetDataset?.id) return;

    setLoading(true);
    setErr(null);
    setAnalysis(null);
    setStatus(null);

    const fetchStatus = async () => {
      try {
        const r = await fetch(
          `${backend}/drift/${baseDataset.id}/${targetDataset.id}/status`,
          { credentials: "same-origin" }
        );
        if (!r.ok) return null;
        return await r.json();
      } catch {
        return null;
      }
    };

    const fetchArtifactIndex = async (runIdValue) => {
      const r = await fetch(`${backend}/runs/${runIdValue}/artifact_index`, {
        credentials: "same-origin",
      });
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        throw new Error(data?.detail || "artifact_index fetch failed");
      }
      return await r.json();
    };

    const fetchDrift = async () => {
      const r = await fetch(`${backend}/drift/v2`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ base_id: baseDataset.id, target_id: targetDataset.id }),
      });
      const data = await r.json().catch(() => ({}));
      if (r.ok && data?.cached) return { kind: "result", data };
      if (r.ok && (data?.status === "started" || data?.status === "running")) {
        return { kind: "running", data };
      }
      throw new Error(data?.detail || "Drift failed");
    };

    const run = async () => {
      try {
        const res = await fetchDrift();
        if (!alive) return;
        if (res.kind === "result") {
          const currentRunId = res.data?.run_id || `drift_${baseDataset.id}_${targetDataset.id}`;
          const index = await fetchArtifactIndex(currentRunId);
          setAnalysis({
            artifact_index: index,
            run_id: currentRunId,
            backend,
            report: res.data?.report,
            plan_name: res.data?.plan_name,
            used_cached_eda: res.data?.used_cached_eda,
          });
          setLoading(false);
          return;
        }

        setLoading(false);
        const st0 = await fetchStatus();
        if (alive && st0) setStatus(st0);

        pollTimer = setInterval(async () => {
          const st = await fetchStatus();
          if (!alive || !st) return;
          setStatus(st);
          if (st?.cache_status?.drift) {
            clearInterval(pollTimer);
            pollTimer = null;
            try {
              setLoading(true);
              const done = await fetchDrift();
              if (!alive) return;
              if (done.kind === "result") {
                const currentRunId = done.data?.run_id || `drift_${baseDataset.id}_${targetDataset.id}`;
                const index = await fetchArtifactIndex(currentRunId);
                setAnalysis({
                  artifact_index: index,
                  run_id: currentRunId,
                  backend,
                  report: done.data?.report,
                  plan_name: done.data?.plan_name,
                  used_cached_eda: done.data?.used_cached_eda,
                });
              }
            } catch (e) {
              if (!alive) return;
              setErr(e?.message || String(e));
            } finally {
              if (alive) setLoading(false);
            }
          }
        }, 2000);
      } catch (e) {
        if (!alive) return;
        setErr(e?.message || String(e));
        setLoading(false);
      }
    };

    run();

    return () => {
      alive = false;
      if (pollTimer) clearInterval(pollTimer);
    };
  }, [backend, baseDataset?.id, targetDataset?.id]);

  const fetchReport = async () => {
    if (!baseDataset?.id || !targetDataset?.id) return;
    setReportBusy(true);
    setErr(null);
    try {
      const r = await fetch(
        `${backend}/report/drift/${baseDataset.id}/${targetDataset.id}`,
        { credentials: "same-origin" }
      );
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data?.detail || "Report generation failed");
      if (!data?.pdf) {
        throw new Error("PDF 리포트 생성에 실패했습니다. 서버 로그를 확인해주세요.");
      }
      await downloadReportFile({
        backend,
        path: data.pdf,
        filename: "drift_report.pdf",
      });
      setAnalysis((prev) =>
        prev ? { ...prev, report: { ...prev.report, pdf: data.pdf } } : prev
      );
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setReportBusy(false);
    }
  };

  return (
    <div className="w-full">
      <div className="sticky top-0 z-30 -mx-4 px-4 sm:px-6 py-3 mb-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-gray-200 bg-white/95 backdrop-blur-sm shadow-sm">
        <div className="flex items-center gap-3 min-w-0">
          {onBack && (
            <button
              type="button"
              onClick={onBack}
              className="flex-shrink-0 px-3 py-2 bg-gray-200 rounded text-xs hover:bg-gray-300"
            >
              ← 뒤로
            </button>
          )}
          <div className="min-w-0">
            <h1 className="text-xl font-semibold leading-tight">Dataset Drift</h1>
            <div
              className="text-xs text-gray-500 truncate max-w-[min(100vw-12rem,42rem)]"
              title={`${baseDataset?.name} vs ${targetDataset?.name}`}
            >
              Base: {baseDataset?.name} / Target: {targetDataset?.name}
            </div>
          </div>
        </div>
        <div className="flex flex-col items-end gap-0.5 shrink-0 ml-auto">
          <button
            type="button"
            onClick={fetchReport}
            disabled={reportBusy || !baseDataset?.id || !targetDataset?.id}
            className="px-4 py-2 bg-gray-900 text-white rounded text-xs font-medium hover:bg-black disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
          >
            {reportBusy ? "PDF 생성 중…" : "PDF 다운로드"}
          </button>
          <span className="text-[10px] text-gray-400 text-right max-w-[220px] leading-tight">
            서버에서 리포트 생성 후 저장
          </span>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-4">
        {loading && <div className="p-4 border rounded bg-gray-50">로딩 중...</div>}
        {err && <div className="p-4 border rounded bg-red-50 text-red-700">오류: {err}</div>}

        {!loading && !err && !analysis && (
          <div className="p-4 border rounded bg-blue-50 text-blue-800">
            <div className="font-medium mb-1">드리프트 분석 진행 중…</div>
            <div className="text-xs">
              화면을 이동해도 서버에서 계속 진행됩니다.
            </div>
            {status?.has_running_tasks && (
              <div className="mt-2 text-xs text-blue-700">
                상태: {status?.status?.state || "running"}
              </div>
            )}
            {status?.status?.used_cached_eda === true && (
              <div className="mt-2 text-xs text-blue-700">
                EDA 재사용: 예 (drift만 수행)
              </div>
            )}
            {status?.status?.used_cached_eda === false && (
              <div className="mt-2 text-xs text-blue-700">
                EDA 포함: 예 (EDA → drift 순서로 수행)
              </div>
            )}
            {status?.status?.error && (
              <div className="mt-2 text-xs text-red-700">
                오류: {status.status.error}
              </div>
            )}
          </div>
        )}

        {!loading && !err && analysis && (
          <div className="p-4 border rounded bg-white">
            {analysis?.plan_name && (
              <div className="text-[11px] text-gray-500 mb-2">
                plan: {analysis.plan_name}
                {analysis?.used_cached_eda ? " · EDA 재사용" : ""}
              </div>
            )}
            <CardRenderer analysisResult={analysis} />
            <details className="mt-4">
              <summary className="text-xs text-gray-500 cursor-pointer">
                artifact_index 보기
              </summary>
              <pre className="text-xs overflow-auto bg-gray-50 p-3 rounded border mt-2">
                {JSON.stringify(analysis?.artifact_index, null, 2)}
              </pre>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}
