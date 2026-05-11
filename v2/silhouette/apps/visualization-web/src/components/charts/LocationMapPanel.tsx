import { useEffect, useMemo, useRef } from "react";

import L from "leaflet";

import { formatNumber } from "../../lib/formatters";

type LocationMapPanelProps = {
  rows: Record<string, unknown>[];
};

function toNumber(value: unknown) {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function LocationMapPanel({ rows }: LocationMapPanelProps) {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const layerGroupRef = useRef<L.LayerGroup | null>(null);
  const points = useMemo(
    () =>
      rows
        .map((row) => ({
          ...row,
          lat: toNumber(row.lat),
          lng: toNumber(row.lng),
          recordCount: toNumber(row.recordCount) ?? 0,
        }))
        .filter((row) => row.lat !== null && row.lng !== null) as Array<
        Record<string, unknown> & { lat: number; lng: number; recordCount: number }
      >,
    [rows],
  );

  useEffect(() => {
    if (!points.length) {
      const existing = mapInstanceRef.current;
      if (existing) {
        existing.remove();
        mapInstanceRef.current = null;
        layerGroupRef.current = null;
      }
      return;
    }
    const container = mapRef.current;
    if (!container || mapInstanceRef.current) {
      return;
    }
    const map = L.map(container, {
      zoomControl: true,
      scrollWheelZoom: false,
    });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
    mapInstanceRef.current = map;
    layerGroupRef.current = L.layerGroup().addTo(map);
    return () => {
      map.remove();
      mapInstanceRef.current = null;
      layerGroupRef.current = null;
    };
  }, [points.length]);

  useEffect(() => {
    const map = mapInstanceRef.current;
    const layerGroup = layerGroupRef.current;
    if (!map || !layerGroup || !points.length) {
      return;
    }
    layerGroup.clearLayers();
    const latLngs: L.LatLngExpression[] = [];
    points.forEach((point) => {
      latLngs.push([point.lat, point.lng]);
      const radius = Math.max(6, Math.min(20, 6 + Math.sqrt(point.recordCount)));
      const marker = L.circleMarker([point.lat, point.lng], {
        radius,
        color: "#60a5fa",
        weight: 1.5,
        fillColor: "#2563eb",
        fillOpacity: 0.65,
      });
      marker.bindPopup(
        [
          `<strong>${String(point.locationLabel ?? "-")}</strong>`,
          `상품 수: ${formatNumber(point.recordCount, "integer")}`,
          `브랜드 수: ${formatNumber(point.brandCount, "integer")}`,
          `평균 순위: ${formatNumber(point.avgRank, "number")}`,
          `평균 가격: ${formatNumber(point.avgPrice, "price")}`,
          `평균 할인율: ${formatNumber(point.avgDiscountPct, "percent")}`,
        ].join("<br />"),
      );
      marker.addTo(layerGroup);
    });
    if (latLngs.length === 1) {
      map.setView(latLngs[0], 12);
      return;
    }
    map.fitBounds(L.latLngBounds(latLngs), { padding: [24, 24] });
  }, [points]);

  if (!points.length) {
    return <div className="empty-state">표시할 위치 좌표가 없습니다. 주소 파싱 또는 지오코딩 결과를 기다려 주세요.</div>;
  }

  return (
    <div className="location-map-panel">
      <div ref={mapRef} className="location-map-panel__map" />
      <div className="location-map-panel__legend">
        <span>점의 크기: 해당 지역의 상품 수</span>
        <span>점의 위치: `영업소재지`에서 파싱한 `구/동` 집계 기준 좌표</span>
      </div>
    </div>
  );
}
