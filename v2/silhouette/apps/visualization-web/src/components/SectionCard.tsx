import { useState, type PropsWithChildren } from "react";

import type { ExplainabilityNote, NarrativeSection, WidgetExplainability } from "../types";

type SectionCardProps = PropsWithChildren<{
  title: string;
  description?: string;
  section?: NarrativeSection;
  takeaway?: string;
  explainability?: WidgetExplainability;
  bodyCollapsible?: boolean;
  defaultBodyExpanded?: boolean;
  bodyToggleLabel?: string;
}>;

const sectionLabelMap: Record<NarrativeSection, string> = {
  summary: "요약",
  input: "입력",
  formula: "계산",
  result: "결과",
  interpretation: "해석",
  examples: "사례",
};

const explainabilityLabelMap: Record<keyof WidgetExplainability, string> = {
  context: "현재 보기",
  readingGuide: "읽는 단위",
  interpretationRules: "해석 기준",
  caveats: "주의사항",
  drilldown: "다음 탐색",
};

export function SectionCard({
  title,
  description,
  section,
  takeaway,
  explainability,
  bodyCollapsible = false,
  defaultBodyExpanded = true,
  bodyToggleLabel = "내용",
  children,
}: SectionCardProps) {
  const [explainabilityOpen, setExplainabilityOpen] = useState(false);
  const [bodyExpanded, setBodyExpanded] = useState(defaultBodyExpanded);
  const explainabilitySections: Array<[keyof WidgetExplainability, WidgetExplainability[keyof WidgetExplainability]]> = [
    ["context", explainability?.context],
    ["readingGuide", explainability?.readingGuide],
    ["interpretationRules", explainability?.interpretationRules],
    ["caveats", explainability?.caveats],
    ["drilldown", explainability?.drilldown],
  ];
  const hasExplainability = explainabilitySections.some(([, value]) => Array.isArray(value) && value.length > 0);

  return (
    <section className={`section-card${section ? ` section-card--${section}` : ""}`}>
      <div className="section-card__header">
        {section ? <span className="section-card__eyebrow">{sectionLabelMap[section]}</span> : null}
        <h2>{title}</h2>
        {description ? <p>{description}</p> : null}
        {takeaway ? <p className="section-card__takeaway">{takeaway}</p> : null}
        {hasExplainability ? (
          <>
            <button
              type="button"
              className="section-card__toggle"
              onClick={() => setExplainabilityOpen((value) => !value)}
              aria-expanded={explainabilityOpen}
            >
              {explainabilityOpen ? "해석 가이드 접기" : "해석 가이드 펼치기"}
            </button>
            <div className={`section-card__explainability${explainabilityOpen ? " is-expanded" : ""}`}>
            {explainability?.context?.length ? (
              <div className="section-card__explainability-block">
                <span className="section-card__explainability-title">{explainabilityLabelMap.context}</span>
                <div className="section-card__fact-grid">
                  {explainability.context.map((fact) => (
                    <div
                      key={`${fact.label}-${fact.value}`}
                      className={`section-card__fact${fact.tone ? ` is-${fact.tone}` : ""}`}
                    >
                      <span>{fact.label}</span>
                      <strong>{fact.value}</strong>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
            {explainabilitySections
              .filter(([key, value]) => key !== "context" && Array.isArray(value) && value.length > 0)
              .map(([key, value]) => (
                <div key={key} className="section-card__explainability-block">
                  <span className="section-card__explainability-title">{explainabilityLabelMap[key]}</span>
                  <ul className="section-card__note-list">
                    {(value as ExplainabilityNote[]).map((note, index) => (
                      <li key={`${key}-${index}`} className={note.tone ? `is-${note.tone}` : undefined}>
                        {note.label ? <strong>{note.label}</strong> : null}
                        <span>{note.text}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </>
        ) : null}
        {bodyCollapsible ? (
          <button
            type="button"
            className="section-card__toggle section-card__toggle--body"
            onClick={() => setBodyExpanded((value) => !value)}
            aria-expanded={bodyExpanded}
          >
            {bodyExpanded ? `${bodyToggleLabel} 접기` : `${bodyToggleLabel} 펼치기`}
          </button>
        ) : null}
      </div>
      <div className={`section-card__body${bodyCollapsible && !bodyExpanded ? " is-collapsed" : ""}`}>
        {children}
      </div>
    </section>
  );
}
