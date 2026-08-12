import { useMemo, useState } from "react";
import { DEMO_INSTALLED, DEMO_KITS, type DomainKit } from "./kits";

const PAGE_LABELS: Record<string, string> = {
  monitor: "모니터",
  collect: "수집",
  drift: "드리프트",
  ontology: "온톨로지",
  review: "검토",
  models: "모델",
};

const SLOT_LABELS: Record<string, string> = {
  image_encoder: "이미지 인코더",
  text_encoder: "텍스트 인코더",
  entity_extractor: "엔티티 추출",
  domain_head: "도메인 헤드",
  car_detector: "차량 검출",
  plate_detector: "번호판 검출",
  ocr: "문자 인식",
};

function shortModelName(modelId: string): string {
  const parts = modelId.split("/");
  return parts[parts.length - 1] || modelId;
}

export function KitHomePage() {
  const [previewKitId, setPreviewKitId] = useState(DEMO_KITS[0]?.kitId ?? null);
  const [activeKitId, setActiveKitId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const previewKit =
    DEMO_KITS.find((kit) => kit.kitId === previewKitId) || DEMO_KITS[0] || null;

  const requiredModels = useMemo(() => {
    if (!previewKit) return [];
    return previewKit.modelRecipe.map((spec) => ({
      slot: spec.slot,
      slotLabel: SLOT_LABELS[spec.slot] || spec.slot,
      modelId: spec.modelId,
      present: DEMO_INSTALLED[spec.modelId] === true,
    }));
  }, [previewKit]);

  const missingCount = requiredModels.filter((row) => !row.present).length;

  function previewKitChoice(kit: DomainKit) {
    setNotice(null);
    setPreviewKitId(kit.kitId);
  }

  function activateKit(kit: DomainKit) {
    setPreviewKitId(kit.kitId);
    setActiveKitId(kit.kitId);
    setNotice(
      `「${kit.labelKo || kit.label}」 킷이 선택되었습니다. 이 페이지는 컨셉 미리보기이며, 실제 워크스페이스는 포함되지 않습니다.`,
    );
  }

  return (
    <main className="home-landing">
      <div className="home-landing__atmosphere" aria-hidden="true" />

      <header className="home-landing__hero">
        <p className="home-landing__eyebrow">DATA DRIFT LAB · CONCEPT</p>
        <h1 className="home-landing__brand">Silhouette</h1>
        <p className="home-landing__lede">
          관심 있는 분야(도메인 킷)를 고르면, 그 분석에 필요한 모델이 자동으로 연결됩니다.
          데이터가 어떻게 변하고 있는지 Silhouette이 함께 살펴봅니다.
        </p>
      </header>

      {notice ? <p className="home-notice">{notice}</p> : null}

      <div className="home-landing__stage">
        <section className="home-panel home-panel--kits" aria-labelledby="home-kits-title">
          <div className="home-panel__head">
            <h2 id="home-kits-title">도메인 킷</h2>
            <p>
              킷을 고른 뒤 <strong>활성화</strong>를 누르면 모델 플레인과 연결되는 흐름을
              미리봅니다.
            </p>
          </div>
          <ul className="home-kit-list">
            {DEMO_KITS.map((kit) => {
              const isPreview = previewKitId === kit.kitId;
              const isActive = activeKitId === kit.kitId;
              return (
                <li key={kit.kitId}>
                  <div
                    className={[
                      "home-kit",
                      isPreview ? "is-selected is-focused" : "",
                      !isPreview && previewKit?.kitId === kit.kitId ? "is-focused" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    <button
                      type="button"
                      className="home-kit__select"
                      onClick={() => previewKitChoice(kit)}
                    >
                      <div className="home-kit__top">
                        <strong>{kit.labelKo || kit.label}</strong>
                        {isPreview ? (
                          <span className="home-pill home-pill--link">
                            {isActive ? "활성" : "선택됨"}
                          </span>
                        ) : null}
                      </div>
                      <span className="home-kit__id">
                        {kit.kitId} · v{kit.kitVersion}
                      </span>
                      <span className="home-kit__meta">
                        {kit.workspacePages.map((p) => PAGE_LABELS[p] || p).join(" · ") ||
                          "워크스페이스"}
                      </span>
                      <div className="home-kit__slots" aria-label="필요 모델 슬롯">
                        {kit.modelRecipe.map((spec) => (
                          <span key={spec.slot} className="home-slot">
                            <em>{SLOT_LABELS[spec.slot] || spec.slot}</em>
                            <span>{shortModelName(spec.modelId)}</span>
                          </span>
                        ))}
                      </div>
                    </button>
                    {isPreview ? (
                      <button
                        type="button"
                        className="home-kit__activate"
                        onClick={() => activateKit(kit)}
                      >
                        이 킷 활성화
                      </button>
                    ) : null}
                  </div>
                </li>
              );
            })}
          </ul>
        </section>

        <div className="home-bridge" aria-hidden="true">
          <span />
          <p>연결</p>
          <span />
        </div>

        <section className="home-panel home-panel--models" aria-labelledby="home-models-title">
          <div className="home-panel__head">
            <h2 id="home-models-title">필요 모델</h2>
            <p>
              {previewKit
                ? `${previewKit.labelKo || previewKit.label} 킷이 분석을 위해 필요로 하는 모델입니다.`
                : "왼쪽에서 도메인 킷을 선택하면 필요 모델이 여기에 표시됩니다."}
              {missingCount > 0 ? ` · ${missingCount}개 없음` : ""}
            </p>
          </div>
          <ul className="home-model-list">
            {requiredModels.map((row) => (
              <li
                key={`${row.slot}-${row.modelId}`}
                className={["home-model", row.present ? "is-linked" : "is-missing"]
                  .filter(Boolean)
                  .join(" ")}
              >
                <div className="home-model__main">
                  <strong title={row.modelId}>{row.modelId || "—"}</strong>
                  <span className="home-model__id">{row.slotLabel}</span>
                </div>
                <div className="home-model__flags">
                  {row.present ? (
                    <span className="home-pill home-pill--ok">있음</span>
                  ) : (
                    <span className="home-pill home-pill--warn">없음</span>
                  )}
                </div>
              </li>
            ))}
          </ul>
          {previewKit ? (
            <p className="home-panel__foot muted">
              활성화하면 Silhouette이{" "}
              <strong>{previewKit.labelKo || previewKit.label}</strong>에 맞는 모델을 연결해
              분석을 시작합니다. 없는 모델은 나중에 모델 화면에서 준비할 수 있습니다.
            </p>
          ) : null}
        </section>
      </div>
    </main>
  );
}
