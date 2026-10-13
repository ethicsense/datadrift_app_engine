"""Report: pad×blur full grid — heatmaps + per-attribute Top 5 vs A."""

from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

ATTRS = ("pattern", "sleeve", "neck", "color", "fit")
ATTR_KO = {"pattern": "무늬", "sleeve": "소매", "neck": "넥라인", "color": "색상", "fit": "핏"}

PADS = (0.0, 0.06, 0.12, 0.20, 0.30)
BLURS = (4.0, 8.0, 12.0, 21.0, 32.0, 48.0)

CSS = """
:root{--fg:#1d2a26;--mute:#5f6d68;--line:#e2e8e4;--bg:#f6f8f6;--pos:#1f7a56;--neg:#b5532a;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:40px 28px 80px}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:20px;margin:44px 0 6px}
p.lead,p.hint,.meta{color:var(--mute)}
.meta{font-size:13px}
.tldr{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--pos);border-radius:10px;padding:16px 20px;margin:18px 0 0}
table{border-collapse:separate;border-spacing:0;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-top:10px}
th,td{padding:9px 10px;text-align:center;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;font-size:13px}
th{background:#eef2ef}
td.l,th.l{text-align:left}
.cell b{display:block;font-size:14px}
.cell small{display:block;font-size:10px;color:var(--mute)}
.top5{display:grid;gap:12px;margin-top:12px}
.top5 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.top5 h3{margin:0 0 8px;font-size:15px}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-top:12px}
summary{cursor:pointer;font-weight:600}
code{background:#eef2ef;padding:1px 5px;border-radius:4px;font-size:12px}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/soft_pad_blur_grid-20261007",
    )
    p.add_argument("--bank", default="pad_blur_grid")
    return p.parse_args()


def _sig(d: dict[str, float]) -> int:
    if d["ci_low"] > 0:
        return 1
    if d["ci_high"] < 0:
        return -1
    return 0


def _tint(value: float, scale: float) -> str:
    if scale <= 0:
        return "transparent"
    alpha = min(abs(value) / scale, 1.0) * 0.55 + 0.08
    rgb = "31,122,86" if value > 0 else "181,83,42"
    return f"rgba({rgb},{alpha:.2f})"


def _cell_name(pad: float, blur: float) -> str:
    return f"p{int(round(pad * 1000)):03d}_b{int(round(blur)):03d}"


def _label(pad: float, blur: float) -> str:
    return f"pad {pad * 100:.0f}% · σ={int(blur)}"


def _variant_arms(pairwise: dict[str, Any]) -> tuple[str, ...]:
    arms = []
    for key in pairwise:
        m = re.match(r"A_vs_(.+)", key)
        if m:
            arms.append(m.group(1))
    return tuple(sorted(set(arms)))


def _pool(pairwise: dict[str, Any], arms_data: dict[str, Any], arms: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for arm in arms:
        key = f"A_vs_{arm}"
        if key not in pairwise or arm not in arms_data:
            continue
        m = re.match(r"p(\d{3})_b(\d+)", arm)
        pad = int(m.group(1)) / 1000.0 if m else 0.0
        blur = float(m.group(2)) if m else 0.0
        out.append(
            {
                "arm": arm,
                "label": _label(pad, blur) if m else arm,
                "pairwise": pairwise[key],
                "arms_row": arms_data[arm],
                "a_row": arms_data["A"],
            }
        )
    return out


def _heatmap_attr(pairwise: dict[str, Any], attr: str, scale: float) -> str:
    head = (
        "<tr><th class='l'>pad \\ σ</th>"
        + "".join(f"<th>{int(b)}</th>" for b in BLURS)
        + "</tr>"
    )
    body = []
    for pad in PADS:
        cells = []
        for blur in BLURS:
            arm = _cell_name(pad, blur)
            key = f"A_vs_{arm}"
            if key not in pairwise:
                cells.append("<td>—</td>")
                continue
            d = pairwise[key][attr]
            sig = _sig(d)
            pp = d["delta"] * 100
            bg = _tint(d["delta"], scale) if sig else "transparent"
            cls = "cell" if sig else "cell"
            cells.append(
                f"<td class='{cls}' style='background:{bg}'><b>{pp:+.1f}</b>"
                f"<small>{'유' if sig else '?'}</small></td>"
            )
        body.append(
            f"<tr><td class='l'><b>{pad * 100:.0f}%</b></td>{''.join(cells)}</tr>"
        )
    return f"<table>{head}{''.join(body)}</table>"


def _top5(pool: list[dict[str, Any]]) -> str:
    cards = []
    for attr in ATTRS:
        ranked = sorted(pool, key=lambda r: float(r["pairwise"][attr]["delta"]), reverse=True)[:5]
        rows = [
            "<tr><th class='l'>#</th><th class='l'>설정</th><th>A 대비 Δ</th><th>95% CI</th>"
            "<th>purity@10</th><th></th></tr>"
        ]
        for i, row in enumerate(ranked, 1):
            d = row["pairwise"][attr]
            sig = _sig(d)
            ap = float(row["arms_row"][attr]["purity_at_k"])
            aa = float(row["a_row"][attr]["purity_at_k"])
            rows.append(
                "<tr>"
                f"<td class='l'>{i}</td>"
                f"<td class='l'>{escape(row['label'])}</td>"
                f"<td><b>{d['delta']*100:+.1f}%p</b></td>"
                f"<td><small>[{d['ci_low']*100:+.1f}, {d['ci_high']*100:+.1f}]</small></td>"
                f"<td>{ap*100:.1f}% <small>(A {aa*100:.1f}%)</small></td>"
                f"<td><small>{'유의' if sig else '?'}</small></td>"
                "</tr>"
            )
        cards.append(
            f"<div class='card'><h3>{ATTR_KO[attr]} — Top 5 (전체 {len(pool)}조합 · A 대비)</h3>"
            f"<table>{''.join(rows)}</table></div>"
        )
    hint = (
        "<p class='hint'>Top 5는 본 프로젝트 pad×blur 그리드 <b>모든 조합</b>에서 purity@10 Δ 기준. "
        "히트맵은 속성별 A 대비 Δ(%p).</p>"
    )
    return hint + f"<div class='top5'>{''.join(cards)}</div>"


def build(project: Path, bank: str) -> Path:
    metrics = json.loads((project / "metrics" / "cluster_quality.json").read_text(encoding="utf-8"))
    summary = json.loads((project / "embeddings" / bank / "summary.json").read_text(encoding="utf-8"))
    pairwise = metrics["banks"][bank]["pairwise"]
    arms_data = metrics["banks"][bank]["arms"]
    arms = _variant_arms(pairwise)
    pool = _pool(pairwise, arms_data, arms)
    n = summary.get("coverage", {}).get("A") or summary.get("n")
    scale = max(abs(pairwise[f"A_vs_{a}"][attr]["delta"]) for a in arms for attr in ATTRS) * 1.05

    heatmaps = "".join(
        f"<h3>{ATTR_KO[attr]}</h3>{_heatmap_attr(pairwise, attr, scale)}" for attr in ATTRS
    )

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pad×blur 전체 그리드</title><style>{CSS}</style></head><body><main>
<h1>패딩 × 블러 전 조합 그리드</h1>
<p class="lead">선행 1축 스윕(패딩 사다리·pad6% 블러 세분화)을 바탕으로, pad 5단 × blur 6단 = 30조합을 한 프로젝트에서 평가합니다. dilate=7.</p>
<p class="meta">상품 {n:,}장 · {len(pool)} variants · {escape(project.name)}</p>

<div class="tldr"><b>구간 선정</b><ul>
<li><b>pad</b>: 0 · 6% · 12% · 20% · 30% (기존 soft_consistency와 동일)</li>
<li><b>blur σ</b>: 4 · 8 · 12 · 21 · 32 · 48 (저·중·고; 12–32는 pad6% 세분화에서 이득 구간)</li>
<li>기존 run에서 재사용 가능한 셀은 <code>reuse_npy</code>로 복사, 나머지만 인코딩</li>
</ul></div>

<h2>1) 속성별 pad×blur 히트맵 (A 대비 Δ %p)</h2>
{heatmaps}

<h2>2) 속성별 Top 5</h2>
{_top5(pool)}

<details open><summary>설계</summary>
<ul>
<li>기준: A · 지표: purity@10 Δ, bootstrap 2000</li>
<li>한 셀 = (pad_ratio, background_blur_sigma, dilate 7)</li>
</ul>
</details>
</main></body></html>"""
    out = project / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    print(build(args.project_dir, args.bank))


if __name__ == "__main__":
    main()
