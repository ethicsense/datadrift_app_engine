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

export default function EDAStudio({ backend, dataset, onBack }) {
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [status, setStatus] = useState(null); // /eda/{id}/status payload
  const [reportBusy, setReportBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    let pollTimer = null;
    setLoading(true);
    setErr(null);
    setAnalysis(null);
    setStatus(null);

    const fetchStatus = async () => {
      try {
        const r = await fetch(`${backend}/eda/${dataset.id}/status`);
        if (!r.ok) return null;
        return await r.json();
      } catch {
        return null;
      }
    };

    const fetchArtifactIndex = async (runIdValue) => {
      const r = await fetch(`${backend}/runs/${runIdValue}/artifact_index`);
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        throw new Error(data?.detail || "artifact_index fetch failed");
      }
      return await r.json();
    };

    const fetchEda = async () => {
      const r = await fetch(`${backend}/eda/${dataset.id}`);
      const data = await r.json().catch(() => ({}));
      // 200: cached 결과 반환
      if (r.ok && data?.cached) return { kind: "result", data };
      // 202-ish 대체: 현재 API는 200으로 {status:"started"/"running"}을 반환
      if (r.ok && (data?.status === "started" || data?.status === "running")) {
        return { kind: "running", data };
      }
      // error
      throw new Error(data?.detail || "EDA failed");
    };

    const run = async () => {
      try {
        const res = await fetchEda();
        if (!alive) return;
        if (res.kind === "result") {
          const currentRunId = res.data?.run_id || dataset.id;
          const index = await fetchArtifactIndex(currentRunId);
          setAnalysis({ artifact_index: index, run_id: currentRunId, backend });
          setLoading(false);
          return;
        }

        // 실행 시작됨 → status 폴링해서 cache 생기면 다시 결과 요청
        setLoading(false);
        const st0 = await fetchStatus();
        if (alive && st0) setStatus(st0);

        pollTimer = setInterval(async () => {
          const st = await fetchStatus();
          if (!alive || !st) return;
          setStatus(st);
          if (st?.cache_status?.eda) {
            clearInterval(pollTimer);
            pollTimer = null;
            // cached 결과 재요청
            try {
              setLoading(true);
              const done = await fetchEda();
              if (!alive) return;
              if (done.kind === "result") {
                const currentRunId = done.data?.run_id || dataset.id;
                const index = await fetchArtifactIndex(currentRunId);
                setAnalysis({ artifact_index: index, run_id: currentRunId, backend });
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
  }, [backend, dataset.id]);

  const fetchReport = async () => {
    setReportBusy(true);
    setErr(null);
    try {
      const baseName = sanitizeReportBasename(dataset?.name, dataset?.id);
      const r = await fetch(`${backend}/report/eda/${dataset.id}`, {
        credentials: "same-origin",
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data?.detail || "Report generation failed");
      if (!data?.pdf) {
        throw new Error("PDF 리포트 생성에 실패했습니다. 서버 로그를 확인해주세요.");
      }
      await downloadReportFile({
        backend,
        path: data.pdf,
        filename: `${baseName}_report.pdf`,
      });
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setReportBusy(false);
    }
  };

  return (
    <div className="w-full">
      {/* main(p-4) 안에서 전폭으로 펼쳐 상단에 고정 → 스크롤해도 PDF가 우측 상단에 보임 */}
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
            <h1 className="text-xl font-semibold leading-tight">EDA</h1>
            <div className="text-xs text-gray-500 truncate max-w-[min(100vw-12rem,42rem)]" title={dataset?.name}>
              Dataset: {dataset?.name}
            </div>
          </div>
        </div>
        <div className="flex flex-col items-end gap-0.5 shrink-0 ml-auto">
          <button
            type="button"
            onClick={fetchReport}
            disabled={reportBusy}
            className="px-4 py-2 bg-gray-900 text-white rounded text-xs font-medium hover:bg-black disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
          >
            {reportBusy ? "PDF 생성 중…" : "PDF 다운로드"}
          </button>
          <span className="text-[10px] text-gray-400 text-right max-w-[220px] leading-tight">
            서버에서 리포트 생성 후 저장
          </span>
        </div>
      </div>

      <div className="max-w-6xl mx-auto">
      {loading && <div className="p-4 border rounded bg-gray-50">로딩 중...</div>}
      {err && <div className="p-4 border rounded bg-red-50 text-red-700">오류: {err}</div>}

      {!loading && !err && !analysis && (
        <div className="p-4 border rounded bg-blue-50 text-blue-800">
          <div className="font-medium mb-1">분석 진행 중…</div>
          <div className="text-xs">
            화면을 이동해도 서버에서 계속 진행됩니다. (목록에서 “분석 중” 배지로 확인 가능)
          </div>
          {status?.has_running_tasks && (
            <div className="mt-2 text-xs text-blue-700">
              상태: {status?.status?.state || "running"}
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


