import React from "react";
import { useNavigate } from "react-router-dom";
import heroImage from "../assets/example.png";

export default function HomeLanding() {
  const navigate = useNavigate();

  return (
    <div className="min-h-[calc(100vh-3.5rem)] flex items-center justify-center bg-gradient-to-b from-slate-50 to-white">
      <div className="max-w-6xl w-full px-6 py-10 grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
        {/* Left: Text & Actions */}
        <div>
          <p className="text-xs font-semibold tracking-[0.2em] text-blue-500 mb-3 uppercase">
            ddoc studio
          </p>
          <h1 className="text-3xl md:text-4xl font-bold text-slate-900 leading-tight mb-4">
            Drift-Centric ML Workspace Hub
          </h1>
          <p className="text-sm md:text-base text-slate-600 mb-6">
            데이터 드리프트를 중심으로 ML 실험과 워크스페이스를 연결하는
            통합 허브입니다. 데이터셋을 업로드하고, 분포를 탐색하고, 드리프트
            분석 결과를 워크스페이스로 이어가 보세요.
          </p>

          <div className="flex flex-wrap items-center gap-3 mb-4">
            <button
              onClick={() => navigate("/datasets")}
              className="inline-flex items-center px-4 py-2.5 rounded-md bg-blue-600 text-white text-sm font-medium shadow-sm hover:bg-blue-700 transition"
            >
              데이터 업로드 & 탐색
            </button>
            <button
              onClick={() => navigate("/workspace")}
              className="inline-flex items-center px-4 py-2.5 rounded-md border border-slate-200 text-slate-700 text-sm font-medium bg-white hover:bg-slate-50 transition"
            >
              워크스페이스 열기
            </button>
          </div>

          <p className="text-xs text-slate-400">
            Drift-centric ML experiments · SQLite + DVC hybrid storage · ddoc
            workspace integration
          </p>
        </div>

        {/* Right: Custom Image */}
        <div className="relative flex items-center justify-center">
          <img
            src={heroImage}
            alt="ddoc studio hero"
            className="w-full h-auto max-h-[420px] rounded-2xl shadow-sm object-cover"
          />
        </div>
      </div>
    </div>
  );
}

