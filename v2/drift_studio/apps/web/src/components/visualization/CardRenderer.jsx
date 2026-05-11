import React, { useEffect, useMemo, useState } from "react";
import { getMatchingCards } from "./cards";

export default function CardRenderer({ analysisResult, className = "" }) {
  const [payloads, setPayloads] = useState({});
  const [payloadLoading, setPayloadLoading] = useState(false);

  const artifactIndex = analysisResult?.artifact_index || analysisResult;
  const runId = analysisResult?.run_id;
  const backend = analysisResult?.backend;

  useEffect(() => {
    let alive = true;
    const load = async () => {
      if (!artifactIndex || !runId || !backend) return;
      const refArtifacts = (artifactIndex.artifacts || []).filter(
        (artifact) => artifact?.payload?.mode === "ref"
      );
      if (!refArtifacts.length) {
        if (alive) setPayloads({});
        return;
      }
      setPayloadLoading(true);
      try {
        const entries = await Promise.all(
          refArtifacts.map(async (artifact) => {
            const r = await fetch(`${backend}/runs/${runId}/artifacts/${artifact.id}`);
            if (!r.ok) {
              throw new Error(`Failed to load artifact: ${artifact.id}`);
            }
            const data = await r.json();
            return [artifact.id, data?.data ?? data];
          })
        );
        if (!alive) return;
        setPayloads(Object.fromEntries(entries));
      } catch {
        if (!alive) return;
        setPayloads({});
      } finally {
        if (alive) setPayloadLoading(false);
      }
    };
    load();
    return () => {
      alive = false;
    };
  }, [artifactIndex, runId, backend]);

  const cards = useMemo(() => {
    if (!artifactIndex) return [];
    return getMatchingCards(artifactIndex, payloads);
  }, [artifactIndex, payloads]);

  if (!artifactIndex) {
    return (
      <div className="p-4 border rounded bg-gray-50 text-gray-500">
        분석 결과가 없습니다.
      </div>
    );
  }

  if (!cards.length) {
    return (
      <div className="p-4 border rounded bg-yellow-50 text-yellow-800">
        이 분석 결과에 대한 카드가 없습니다.
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {payloadLoading && (
        <div className="p-3 border rounded bg-gray-50 text-xs text-gray-600">
          시각화 데이터 로딩 중...
        </div>
      )}
      {cards.map((card, idx) => {
        const CardComponent = card.component;
        return (
          <CardComponent
            key={card.id || idx}
            data={card.data}
            analysisResult={analysisResult}
            cardConfig={card}
          />
        );
      })}
    </div>
  );
}
