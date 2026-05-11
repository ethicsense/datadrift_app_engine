import React, { useEffect, useState } from "react";
import { downloadReportFile } from "../utils/reportDownload";

function TreeNode({ node, depth = 0 }) {
  const [expanded, setExpanded] = useState(depth < 2);
  const hasChildren = node.children && node.children.length > 0;

  return (
    <div className="ml-2">
      <div
        className={`text-xs flex items-center gap-1 py-0.5 ${hasChildren ? "cursor-pointer hover:text-blue-600" : ""}`}
        onClick={() => hasChildren && setExpanded(!expanded)}
      >
        {hasChildren && <span className="w-3 text-gray-400">{expanded ? "▼" : "▶"}</span>}
        {hasChildren ? "📁" : "📄"} {node.name}
      </div>
      {hasChildren && expanded && (
        <div className="ml-3 border-l border-gray-200 pl-2">
          {node.children.map((c, i) => (
            <TreeNode key={`${c.name}-${i}`} node={c} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function ZipDetail({ backend, dataset, onBack, onOpenEDA }) {
  const [ds, setDs] = useState(dataset);
  const [loading, setLoading] = useState(true);
  const [reportPaths, setReportPaths] = useState(null);
  const [ddoc, setDdoc] = useState(null);
  const [ddocError, setDdocError] = useState(null);

  useEffect(() => {
    setLoading(true);
    // ZIP 구조/트리/간단 통계는 EDA 플러그인과 무관하게 Dataset preview로 제공
    Promise.all([
      fetch(`${backend}/datasets/${dataset.id}`).then(async (r) => {
        if (!r.ok) throw new Error(`Dataset 로딩 실패 (HTTP ${r.status})`);
        return r.json();
      }),
      fetch(`${backend}/datasets/${dataset.id}/ddoc`).then(async (r) => {
        if (!r.ok) throw new Error(`ddoc.yaml 로딩 실패 (HTTP ${r.status})`);
        return r.json();
      }),
    ])
      .then(([dsData, ddocData]) => {
        setDs(dsData);
        setDdoc(ddocData);
        setDdocError(null);
      })
      .catch((err) => {
        console.error("ZIP 메타 로딩 실패:", err);
        setDdocError(err?.message || String(err));
      })
      .finally(() => setLoading(false));
  }, [backend, dataset.id]);

  const downloadReport = () => {
    fetch(`${backend}/report/eda/${dataset.id}`)
      .then(async (r) => {
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          throw new Error(data?.detail || `리포트 생성 실패 (HTTP ${r.status})`);
        }
        return data;
      })
      .then((info) => {
        setReportPaths(info);
        if (info.pdf) {
          return downloadReportFile({
            backend,
            path: info.pdf,
            filename: `${dataset?.name || dataset?.id || "eda-report"}.pdf`,
          });
        }
        throw new Error("PDF 리포트 생성에 실패했습니다.");
      })
      .catch((e) => {
        alert(e?.message || String(e));
      });
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto p-4">
        <button
          onClick={onBack}
          className="mb-4 px-3 py-2 bg-gray-200 rounded text-xs hover:bg-gray-300 transition"
        >
          ← 뒤로
        </button>
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
          ZIP 구조 분석 중...
        </div>
      </div>
    );
  }

  if (!ds) {
    return (
      <div className="max-w-6xl mx-auto p-4">
        <button
          onClick={onBack}
          className="mb-4 px-3 py-2 bg-gray-200 rounded text-xs hover:bg-gray-300 transition"
        >
          ← 뒤로
        </button>
        <div className="text-sm text-red-500">ZIP 메타를 불러올 수 없습니다.</div>
        {ddocError && <div className="mt-2 text-xs text-gray-500">{ddocError}</div>}
      </div>
    );
  }

  const preview = ds.preview || {};
  const stats = preview.stats || {};

  return (
    <div className="max-w-6xl mx-auto p-4">
      {/* 헤더 */}
      <div className="flex items-center justify-between mb-4">
        <button
          onClick={onBack}
          className="px-3 py-2 bg-gray-200 rounded text-xs hover:bg-gray-300 transition"
        >
          ← 뒤로
        </button>
        
        {/* 심화 분석 버튼 */}
        <button
          onClick={onOpenEDA}
          className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg text-sm font-medium hover:from-green-600 hover:to-emerald-700 transition shadow-md hover:shadow-lg flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          심화 분석 (EDA)
        </button>
      </div>

      {/* 타이틀 */}
      <div className="mb-6">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          📦 {dataset.name}
        </h2>
        <div className="text-xs text-gray-500 mt-1">
          Modality: <span className="font-medium text-gray-700">{preview?.meta?.modality || preview.zip_type || "unknown"}</span> · 
          Total files: <span className="font-medium text-gray-700">{stats.total_files ?? 0}</span>
        </div>
      </div>

      {/* 파일 통계 카드 */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
        <StatCard label="이미지" value={stats.image_files ?? 0} icon="🖼️" color="blue" />
        <StatCard label="텍스트" value={stats.text_files ?? 0} icon="📝" color="green" />
        <StatCard label="CSV" value={stats.csv_files ?? 0} icon="📊" color="yellow" />
        <StatCard label="JSON" value={stats.json_files ?? 0} icon="🔧" color="purple" />
        <StatCard label="XML" value={stats.xml_files ?? 0} icon="📄" color="gray" />
      </div>

      {/* ddoc.yaml 확인 */}
      <div className="bg-white rounded-lg shadow p-4 mb-4">
        <div className="flex items-center justify-between mb-3">
          <div className="font-semibold text-sm flex items-center gap-2">
            🧾 ddoc.yaml
          </div>
          {ddoc?.truncated && (
            <div className="text-[11px] text-amber-600">
              (너무 길어서 일부만 표시됨)
            </div>
          )}
        </div>
        {ddoc?.raw ? (
          <pre className="max-h-80 overflow-auto border rounded p-2 bg-gray-50 text-[11px] whitespace-pre-wrap">
            {ddoc.raw}
          </pre>
        ) : (
          <div className="text-xs text-gray-500">
            ddoc.yaml을 불러올 수 없습니다{ddocError ? `: ${ddocError}` : ""}.
          </div>
        )}
      </div>

      {/* 디렉토리 구조 */}
      <div className="bg-white rounded-lg shadow p-4 mb-4">
        <div className="flex items-center justify-between mb-3">
          <div className="font-semibold text-sm flex items-center gap-2">
            📁 폴더 구조
          </div>
          <div className="text-xs text-gray-500">
            Top-level: {stats.subdirs?.join(", ") || "-"}
          </div>
        </div>
        {preview.tree ? (
          <div className="max-h-80 overflow-auto border rounded p-2 bg-gray-50">
            <TreeNode node={preview.tree} />
          </div>
        ) : (
          <div className="text-xs text-gray-500">트리 정보 없음</div>
        )}
      </div>

      {/* 리포트 다운로드 */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="font-semibold text-sm mb-3">📑 리포트</div>
        <button
          onClick={downloadReport}
          className="px-4 py-2 bg-blue-600 text-white rounded text-xs hover:bg-blue-700 transition"
        >
          ZIP EDA 리포트 다운로드 (PDF)
        </button>
        {reportPaths && (
          <div className="mt-2 text-[10px] text-gray-500">
            PDF: {reportPaths.pdf || "생성 실패 또는 WeasyPrint 런타임 의존성 누락"}
          </div>
        )}
      </div>
    </div>
  );
}


/* ====================================== */
/* 통계 카드 컴포넌트 */
/* ====================================== */
function StatCard({ label, value, icon, color = "gray" }) {
  const colorClasses = {
    blue: "bg-blue-50 border-blue-200 text-blue-700",
    green: "bg-green-50 border-green-200 text-green-700",
    yellow: "bg-yellow-50 border-yellow-200 text-yellow-700",
    purple: "bg-purple-50 border-purple-200 text-purple-700",
    gray: "bg-gray-50 border-gray-200 text-gray-700",
  };

  return (
    <div className={`p-3 rounded-lg border ${colorClasses[color]}`}>
      <div className="flex items-center gap-2">
        <span className="text-lg">{icon}</span>
        <div>
          <div className="text-xs opacity-75">{label}</div>
          <div className="text-lg font-bold">{value}</div>
        </div>
      </div>
    </div>
  );
}


/* ====================================== */
/* Roboflow 섹션 */
/* ====================================== */
function RoboflowSection({ roboflow }) {
  return (
    <div className="bg-white rounded-lg shadow p-4 mb-4">
      <div className="font-semibold text-sm mb-3 flex items-center gap-2">
        🏷️ Roboflow 클래스
        <span className="text-xs font-normal text-gray-500">
          ({roboflow.classes.length}개)
        </span>
      </div>
      
      <div className="flex flex-wrap gap-2 mb-4">
        {roboflow.classes.map((c) => (
          <span
            key={c}
            className="px-3 py-1 bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-700 rounded-full text-xs font-medium border border-blue-200"
          >
            {c}
          </span>
        ))}
      </div>

      {/* 스플릿별 통계 */}
      <div className="grid grid-cols-3 gap-3">
        {Object.entries(roboflow.splits).map(([splitName, s]) => (
          <div key={splitName} className="p-3 bg-gray-50 rounded-lg">
            <div className="font-semibold text-xs mb-2 capitalize text-gray-700">
              {splitName}
            </div>
            <div className="text-xs text-gray-600 space-y-1">
              <div>🖼️ Images: <span className="font-medium">{s.num_images}</span></div>
              <div>📝 Labels: <span className="font-medium">{s.num_labels}</span></div>
            </div>
            {s.class_counts && (
              <div className="mt-2 pt-2 border-t border-gray-200">
                <div className="text-[10px] text-gray-500 mb-1">클래스별 개수:</div>
                <div className="text-[10px] space-y-0.5">
                  {Object.entries(s.class_counts).slice(0, 5).map(([cls, cnt]) => (
                    <div key={cls} className="flex justify-between">
                      <span className="truncate">{cls}</span>
                      <span className="font-medium">{cnt}</span>
                    </div>
                  ))}
                  {Object.keys(s.class_counts).length > 5 && (
                    <div className="text-gray-400">+{Object.keys(s.class_counts).length - 5}개 더...</div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}


/* ====================================== */
/* YOLO 섹션 */
/* ====================================== */
function YoloSection({ yolo }) {
  return (
    <div className="bg-white rounded-lg shadow p-4 mb-4">
      <div className="font-semibold text-sm mb-3 flex items-center gap-2">
        🎯 YOLO 클래스
        <span className="text-xs font-normal text-gray-500">
          ({yolo.classes?.length || 0}개)
        </span>
      </div>
      
      {yolo.classes && (
        <div className="flex flex-wrap gap-2 mb-4">
          {yolo.classes.map((c, i) => (
            <span
              key={i}
              className="px-3 py-1 bg-gradient-to-r from-orange-100 to-amber-100 text-orange-700 rounded-full text-xs font-medium border border-orange-200"
            >
              {c}
            </span>
          ))}
        </div>
      )}

      {/* 스플릿별 통계 */}
      {yolo.splits && (
        <div className="grid grid-cols-3 gap-3">
          {Object.entries(yolo.splits).map(([splitName, s]) => (
            <div key={splitName} className="p-3 bg-gray-50 rounded-lg">
              <div className="font-semibold text-xs mb-2 capitalize text-gray-700">
                {splitName}
              </div>
              <div className="text-xs text-gray-600 space-y-1">
                <div>🖼️ Images: <span className="font-medium">{s.num_images}</span></div>
                <div>📝 Labels: <span className="font-medium">{s.num_labels}</span></div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


/* ====================================== */
/* VOC 섹션 */
/* ====================================== */
function VocSection({ voc }) {
  return (
    <div className="bg-white rounded-lg shadow p-4 mb-4">
      <div className="font-semibold text-sm mb-3 flex items-center gap-2">
        📋 VOC 클래스
        <span className="text-xs font-normal text-gray-500">
          ({voc.classes?.length || 0}개)
        </span>
      </div>
      
      {voc.classes && (
        <div className="flex flex-wrap gap-2">
          {voc.classes.map((c, i) => (
            <span
              key={i}
              className="px-3 py-1 bg-gradient-to-r from-green-100 to-teal-100 text-green-700 rounded-full text-xs font-medium border border-green-200"
            >
              {c}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}


/* ====================================== */
/* COCO 섹션 */
/* ====================================== */
function CocoSection({ coco }) {
  return (
    <div className="bg-white rounded-lg shadow p-4 mb-4">
      <div className="font-semibold text-sm mb-3 flex items-center gap-2">
        🎨 COCO 클래스
        <span className="text-xs font-normal text-gray-500">
          ({coco.classes?.length || 0}개)
        </span>
      </div>
      
      {coco.classes && (
        <div className="flex flex-wrap gap-2 mb-4">
          {coco.classes.map((c, i) => (
            <span
              key={i}
              className="px-3 py-1 bg-gradient-to-r from-pink-100 to-rose-100 text-pink-700 rounded-full text-xs font-medium border border-pink-200"
            >
              {c}
            </span>
          ))}
        </div>
      )}

      {coco.stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="p-2 bg-gray-50 rounded text-center">
            <div className="text-xs text-gray-500">이미지</div>
            <div className="text-lg font-bold text-gray-700">{coco.stats.num_images}</div>
          </div>
          <div className="p-2 bg-gray-50 rounded text-center">
            <div className="text-xs text-gray-500">어노테이션</div>
            <div className="text-lg font-bold text-gray-700">{coco.stats.num_annotations}</div>
          </div>
          <div className="p-2 bg-gray-50 rounded text-center">
            <div className="text-xs text-gray-500">카테고리</div>
            <div className="text-lg font-bold text-gray-700">{coco.stats.num_categories}</div>
          </div>
        </div>
      )}
    </div>
  );
}
