import { useState, type ReactNode } from "react";

import type { ExplainabilityFact, ExplainabilityNote } from "../types";

type PageContextBarProps = {
  title: string;
  summary: string;
  badges?: ExplainabilityFact[];
  notes?: ExplainabilityNote[];
  controls?: ReactNode;
};

export function PageContextBar({ title, summary, badges = [], notes = [], controls }: PageContextBarProps) {
  const [expanded, setExpanded] = useState(false);
  const hasDetails = badges.length > 0 || notes.length > 0 || Boolean(controls);

  return (
    <section className="page-context-bar">
      <div className="page-context-bar__header">
        <div>
          <small>현재 읽는 기준</small>
          <h2>{title}</h2>
        </div>
        <p>{summary}</p>
        {hasDetails ? (
          <button
            type="button"
            className="page-context-bar__toggle"
            onClick={() => setExpanded((value) => !value)}
            aria-expanded={expanded}
          >
            {expanded ? "상세 맥락 접기" : "상세 맥락 펼치기"}
          </button>
        ) : null}
      </div>
      {hasDetails ? (
        <div className={`page-context-bar__details${expanded ? " is-expanded" : ""}`}>
          {badges.length ? (
            <div className="page-context-bar__badges">
              {badges.map((badge) => (
                <div
                  key={`${badge.label}-${badge.value}`}
                  className={`page-context-bar__badge${badge.tone ? ` is-${badge.tone}` : ""}`}
                >
                  <span>{badge.label}</span>
                  <strong>{badge.value}</strong>
                </div>
              ))}
            </div>
          ) : null}
          {notes.length ? (
            <div className="page-context-bar__notes">
              {notes.map((note, index) => (
                <div key={`${note.label ?? "note"}-${index}`} className={`page-context-bar__note${note.tone ? ` is-${note.tone}` : ""}`}>
                  {note.label ? <strong>{note.label}</strong> : null}
                  <p>{note.text}</p>
                </div>
              ))}
            </div>
          ) : null}
          {controls ? <div className="page-context-bar__controls">{controls}</div> : null}
        </div>
      ) : null}
    </section>
  );
}
