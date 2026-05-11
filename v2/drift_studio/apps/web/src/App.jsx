import React, { useEffect, useState } from "react";
import Layout from "./components/Layout";
import HomeLanding from "./components/HomeLanding";
import DatasetGrid from "./components/DatasetGrid";
import EDAStudio from "./components/EDAStudio";
import DriftStudio from "./components/DriftStudio";
import ZipDetail from "./components/ZipDetail";
import { Routes, Route } from "react-router-dom";

// 리버스 프록시 환경: 상대 경로 사용
const BACKEND = "/api";

export default function App() {
  const [datasets, setDatasets] = useState([]);
  const [datasetsError, setDatasetsError] = useState(null);

  // Drift
  const [compareBase, setCompareBase] = useState(null);
  const [compareTarget, setCompareTarget] = useState(null);

  // View state (used in Datasets route)
  const [view, setView] = useState("workspace");
  const [selectedDataset, setSelectedDataset] = useState(null);

  const fetchDatasets = async () => {
    try {
      setDatasetsError(null);
      const r = await fetch(`${BACKEND}/datasets/`);
      if (!r.ok) {
        throw new Error(`datasets 목록 조회 실패 (HTTP ${r.status})`);
      }

      const ct = r.headers.get("content-type") || "";
      if (!ct.includes("application/json")) {
        const body = await r.text();
        throw new Error(
          `datasets 응답이 JSON이 아닙니다 (content-type=${ct || "unknown"}). ` +
            `프록시/리다이렉트 문제일 수 있어요. body=${body.slice(0, 200)}`
        );
      }

      const data = await r.json();
      setDatasets(Array.isArray(data) ? data : []);
    } catch (e) {
      console.error("Failed to fetch datasets:", e);
      setDatasets([]);
      setDatasetsError(e?.message || String(e));
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  return (
    <Layout>
      <Routes>
        <Route
          path="/"
          element={
            <HomeLanding />
          }
        />
        <Route
          path="/datasets"
          element={
            <>
              {datasetsError && (
                <div className="max-w-6xl mx-auto mb-4 p-3 rounded-lg border border-red-200 bg-red-50 text-sm text-red-700">
                  {datasetsError}
                </div>
              )}
              {view === "workspace" && (
                <DatasetGrid
                  datasets={datasets}
                  backend={BACKEND}
                  refresh={fetchDatasets}
                  onEDA={(ds) => {
                    setSelectedDataset(ds);
                    setView("eda");
                  }}
                  onDrift={(ds) => {
                    setCompareBase(ds);
                    setView("selectTarget");
                  }}
                  onSelect={(ds) => {
                    if (ds.type === "zip") {
                      setSelectedDataset(ds);
                      setView("zipDetail");
                    } else {
                      setSelectedDataset(ds);
                      setView("eda");
                    }
                  }}
                  driftMode={false}
                />
              )}

              {view === "selectTarget" && (
                <DatasetGrid
                  datasets={datasets}
                  backend={BACKEND}
                  title="비교 대상 데이터셋 선택"
                  driftMode={true}
                  compareBase={compareBase}
                  onSelectTarget={(ds) => {
                    setCompareTarget(ds);
                    setView("drift");
                  }}
                  onBack={() => setView("workspace")}
                />
              )}

              {view === "eda" && selectedDataset && (
                <EDAStudio
                  backend={BACKEND}
                  dataset={selectedDataset}
                  onBack={() => setView("workspace")}
                />
              )}

              {view === "drift" && compareBase && compareTarget && (
                <DriftStudio
                  backend={BACKEND}
                  baseDataset={compareBase}
                  targetDataset={compareTarget}
                  onBack={() => setView("workspace")}
                />
              )}

              {view === "zipDetail" && selectedDataset && (
                <ZipDetail
                  backend={BACKEND}
                  dataset={selectedDataset}
                  onBack={() => setView("workspace")}
                  onOpenEDA={() => setView("eda")}
                />
              )}
            </>
          }
        />
      </Routes>
    </Layout>
  );
}