"""Report: pad_ratio=6% fixed, fine-grained blur σ ladder + per-attr top3 vs A."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

ATTRS = ("pattern", "sleeve", "neck", "color", "fit")
ATTR_KO = {"pattern": "무늬", "sleeve": "소매", "neck": "넥라인", "color": "색상", "fit": "핏"}

BLUR_ARMS = (
    "blur_s02",
    "blur_s04",
    "blur_s06",
    "blur_s08",
    "blur_s12",
    "blur_s16",
    "blur_s21",
    "blur_s32",
    "blur_s48",
)
ARM_INFO = {
    "A": ("원본", "기준"),
    "blur_s02": ("σ=2", "매우 약함"),
    "blur_s04": ("σ=4", "약함"),
    "blur_s06": ("σ=6", ""),
    "blur_s08": ("σ=8", ""),
    "blur_s12": ("σ=12", ""),
    "blur_s16": ("σ=16", ""),
    "blur_s21": ("σ=21", "기존 soft 블러"),
    "blur_s32": ("σ=32", "강함"),
    "blur_s48": ("σ=48", "매우 강함"),
}
EXAMPLES = (
    ("3cd64dfc19420475", "그래픽/로고"),
    ("7aaccf9badfc256e", "스트라이프"),
    ("cb0571ddec533d9f", "후드"),
)

CSS = """
:root{--fg:#1d2a26;--mute:#5f6d68;--line:#e2e8e4;--bg:#f6f8f6;--pos:#1f7a56;--neg:#b5532a;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:40px 28px 80px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:20px;margin:44px 0 6px}
p.lead,p.hint,.meta{color:var(--mute)}
.meta{font-size:13px}
.tldr{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--pos);border-radius:10px;padding:16px 20px;margin:18px 0 0}
.tldr ul{margin:6px 0 0;padding-left:20px}
table{border-collapse:separate;border-spacing:0;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-top:10px}
th,td{padding:9px 10px;text-align:center;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{background:#eef2ef;font-size:12px}
td.l,th.l{text-align:left}
tr:last-child td{border-bottom:none}
.cell b{display:block;font-size:15px}
.cell small{display:block;font-size:10px;color:var(--mute)}
.ns{opacity:.55}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mute);margin-top:8px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px}
.trend{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-top:12px}
.bar-row{display:grid;grid-template-columns:120px 1fr 70px;gap:8px;align-items:center;font-size:13px;margin:6px 0}
.track{position:relative;height:16px;background:#f1f4f2;border-radius:4px}
.track .zero{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:#9aa6a1}
.track .bar{position:absolute;top:2px;height:12px;border-radius:3px}
.ex{display:grid;gap:8px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto;margin-top:10px}
.ex .h{font-size:11px;color:var(--mute);text-align:center;font-weight:600}
.ex .rl{font-size:13px;font-weight:600}
.ex img{width:100%;aspect-ratio:1;object-fit:contain;background:#eef1ef;border-radius:6px}
.top3{display:grid;gap:12px;margin-top:12px}
.top3 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.top3 h3{margin:0 0 8px;font-size:15px}
.rank{display:inline-block;min-width:1.4em;font-weight:700;color:var(--mute)}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-top:12px}
summary{cursor:pointer;font-weight:600}
code{background:#eef2ef;padding:1px 5px;border-radius:4px;font-size:12px}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/soft_pad06_blur_fine-20261007",
    )
    p.add_argument("--bank", default="soft_consistency")
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


def _available_arms(pairwise: dict[str, Any]) -> tuple[str, ...]:
    return tuple(arm for arm in BLUR_ARMS if f"A_vs_{arm}" in pairwise)


def _heatmap(pairwise: dict[str, Any], arms: tuple[str, ...]) -> str:
    scale = max(
        abs(pairwise[f"A_vs_{arm}"][a]["delta"]) for arm in arms for a in ATTRS
    )
    head = (
        "<tr><th class='l'>속성</th>"
        + "".join(f"<th>{escape(ARM_INFO[arm][0])}</th>" for arm in arms)
        + "</tr>"
    )
    body = []
    for a in ATTRS:
        cells = []
        for arm in arms:
            d = pairwise[f"A_vs_{arm}"][a]
            sig = _sig(d)
            pp = d["delta"] * 100
            note = "유의" if sig else "불확실"
            cls = "cell" if sig else "cell ns"
            bg = _tint(d["delta"], scale) if sig else "transparent"
            cells.append(
                f"<td class='{cls}' style='background:{bg}'><b>{pp:+.1f}%p</b>"
                f"<small>{note}</small></td>"
            )
        body.append(f"<tr><td class='l'><b>{ATTR_KO[a]}</b></td>{''.join(cells)}</tr>")
    return f"<table>{head}{''.join(body)}</table>"


def _trend_lines(pairwise: dict[str, Any], arms: tuple[str, ...], attr: str, scale: float) -> str:
    rows = []
    for arm in arms:
        d = pairwise[f"A_vs_{arm}"][attr]
        sig = _sig(d)
        color = "var(--pos)" if d["delta"] >= 0 else "var(--neg)"
        width = abs(d["delta"]) / scale * 50
        left = 50 if d["delta"] >= 0 else 50 - width
        rows.append(
            f"<div class='bar-row{'' if sig else ' ns'}'><span>{escape(ARM_INFO[arm][0])}</span>"
            f"<div class='track'><span class='zero'></span>"
            f"<span class='bar' style='left:{left:.1f}%;width:{max(width,0.8):.1f}%;background:{color}'></span>"
            f"</div><span>{d['delta']*100:+.1f}%p</span></div>"
        )
    return "".join(rows)


# Sibling soft_consistency arms (pad ladder + blur@pad12) included in global Top3.
SOFT_CONSISTENCY_DIR = (
    REPO_ROOT / "artifacts/garment_segmentation/soft_consistency_pad_blur-20261007"
)
SOFT_CONSISTENCY_LABELS = {
    "pad_r0": "pad 0% · blur 21",
    "pad_r06": "pad 6% · blur 21",
    "soft_ref": "pad 12% · blur 21 (soft_ref)",
    "pad_r20": "pad 20% · blur 21",
    "pad_r30": "pad 30% · blur 21",
    "blur_s04": "pad 12% · blur σ=4",
    "blur_s08": "pad 12% · blur σ=8",
    "blur_s40": "pad 12% · blur σ=40",
    "blur_s60": "pad 12% · blur σ=60",
}
PAD06_LABELS = {
    "blur_s02": "pad 6% · blur σ=2",
    "blur_s04": "pad 6% · blur σ=4",
    "blur_s06": "pad 6% · blur σ=6",
    "blur_s08": "pad 6% · blur σ=8",
    "blur_s12": "pad 6% · blur σ=12",
    "blur_s16": "pad 6% · blur σ=16",
    "blur_s21": "pad 6% · blur σ=21",
    "blur_s32": "pad 6% · blur σ=32",
    "blur_s48": "pad 6% · blur σ=48",
}


def _collect_all_variant_pool(
    local_pairwise: dict[str, Any],
    local_arms: dict[str, Any],
    local_arm_names: tuple[str, ...],
    *,
    bank: str,
) -> list[dict[str, Any]]:
    """All soft-mask settings across soft_consistency + this pad06 fine-blur run."""
    pool: list[dict[str, Any]] = []
    for arm in local_arm_names:
        key = f"A_vs_{arm}"
        if key not in local_pairwise or arm not in local_arms:
            continue
        pool.append(
            {
                "id": f"pad06:{arm}",
                "label": PAD06_LABELS.get(arm, arm),
                "pairwise": local_pairwise[key],
                "arms_row": local_arms[arm],
                "a_row": local_arms["A"],
            }
        )
    sibling = SOFT_CONSISTENCY_DIR / "metrics" / "cluster_quality.json"
    if sibling.is_file():
        sm = json.loads(sibling.read_text(encoding="utf-8"))
        sp = sm["banks"][bank]["pairwise"]
        sa = sm["banks"][bank]["arms"]
        for arm, label in SOFT_CONSISTENCY_LABELS.items():
            key = f"A_vs_{arm}"
            if key not in sp or arm not in sa:
                continue
            pool.append(
                {
                    "id": f"soft:{arm}",
                    "label": label,
                    "pairwise": sp[key],
                    "arms_row": sa[arm],
                    "a_row": sa["A"],
                }
            )
    return pool


def _top3_per_attr(pool: list[dict[str, Any]]) -> str:
    """Per attribute: top-3 among all experimental settings by purity@10 Δ vs A."""
    cards = []
    for attr in ATTRS:
        ranked = sorted(
            pool,
            key=lambda row: float(row["pairwise"][attr]["delta"]),
            reverse=True,
        )[:3]
        rows = [
            "<tr><th class='l'>순위</th><th>설정</th><th>A 대비 Δ</th><th>95% CI</th>"
            "<th>purity@10</th><th></th></tr>"
        ]
        for i, row in enumerate(ranked, 1):
            d = row["pairwise"][attr]
            sig = _sig(d)
            arm_p = float(row["arms_row"][attr]["purity_at_k"])
            a_purity = float(row["a_row"][attr]["purity_at_k"])
            note = "유의" if sig else "불확실"
            rows.append(
                "<tr>"
                f"<td class='l'><span class='rank'>{i}</span></td>"
                f"<td class='l'>{escape(row['label'])}</td>"
                f"<td><b>{d['delta']*100:+.1f}%p</b></td>"
                f"<td><small>[{d['ci_low']*100:+.1f}, {d['ci_high']*100:+.1f}]</small></td>"
                f"<td>{arm_p*100:.1f}% <small>(A {a_purity*100:.1f}%)</small></td>"
                f"<td><small>{note}</small></td>"
                "</tr>"
            )
        cards.append(
            f"<div class='card'><h3>{ATTR_KO[attr]} — Top 3 (전체 설정 · A 대비 purity@10 Δ)</h3>"
            f"<table>{''.join(rows)}</table></div>"
        )
    return (
        "<p class='hint'>속성별 Top 3는 pad/blur 사다리(<code>soft_consistency_pad_blur-20261007</code>)와 "
        "본 실험(pad 6% · blur 세분화)의 <b>모든 설정</b>을 한 풀에서 산정합니다. "
        "Δ·CI는 A 기준, purity@10은 암 절대값(괄호 A).</p>"
        f"<div class='top3'>{''.join(cards)}</div>"
    )


def _examples(root: Path, arms: tuple[str, ...]) -> str:
    # show subset of arms to keep grid readable
    show = tuple(a for a in ("blur_s04", "blur_s08", "blur_s21", "blur_s48") if a in arms)
    cols = ("A",) + show
    rows = ["<div class='h'></div>"] + [f"<div class='h'>{escape(ARM_INFO.get(a, (a,))[0])}</div>" for a in cols]
    n = len(cols)
    grid = f"grid-template-columns:90px repeat({n},1fr)"
    for item_id, label in EXAMPLES:
        paths = []
        ok = True
        for a in cols:
            p = root / "previews" / a / f"{item_id}.jpg"
            if not p.is_file():
                ok = False
                break
            paths.append(p)
        if not ok:
            continue
        rows.append(f"<div class='rl'>{escape(label)}</div>")
        for a, p in zip(cols, paths):
            rows.append(f"<img src='{p.relative_to(root).as_posix()}' alt='{a}'>")
    return f"<div class='ex' style='{grid}'>{''.join(rows)}</div>"


def _auto_tldr(pairwise: dict[str, Any], arms: tuple[str, ...]) -> str:
    scores = {
        arm: sum(
            pairwise[f"A_vs_{arm}"][a]["delta"]
            for a in ATTRS
            if _sig(pairwise[f"A_vs_{arm}"][a]) >= 0
        )
        for arm in arms
    }
    best = max(scores, key=scores.get)
    return (
        f"<li>pad <b>6%</b> 고정 · 블러 σ 세분화 ({len(arms)}단)</li>"
        f"<li>유의·비악화 Δ 합 최고: <b>{escape(ARM_INFO[best][0])}</b> ({best})</li>"
        "<li>비교 기준: 원본 <b>A</b> 대비 purity@10 Δ (bootstrap 2000)</li>"
        "<li>속성별 Top 3 표는 아래 2절</li>"
    )


def build(project: Path, bank: str) -> Path:
    metrics = json.loads((project / "metrics" / "cluster_quality.json").read_text(encoding="utf-8"))
    summary = json.loads((project / "embeddings" / bank / "summary.json").read_text(encoding="utf-8"))
    pairwise = metrics["banks"][bank]["pairwise"]
    arms_data = metrics["banks"][bank]["arms"]
    arms = _available_arms(pairwise)
    if not arms:
        raise SystemExit(f"no blur arms found in metrics for bank={bank}")
    n = summary.get("coverage", {}).get("A") or summary.get("n")
    scale = max(abs(pairwise[f"A_vs_{arm}"][a]["delta"]) for arm in arms for a in ATTRS) * 1.05

    src_note = project.name
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pad 6% · 블러 세분화</title><style>{CSS}</style></head><body><main>
<h1>pad 6%에서 블러 강도를 세분화하면 군집 이득이 어떻게 변하나</h1>
<p class="lead">크롭 패딩을 6%로 고정하고 배경 블러 σ만 촘촘히 바꿔, soft-mask FashionSigLIP 군집도가 A 대비 어떻게 움직이는지 봅니다. dilate=7.</p>
<p class="meta">상품 {n:,}장 · bank {escape(bank)} · {escape(str(metrics.get('created_at', ''))[:10])} · 프로젝트 {escape(src_note)}</p>

<div class="tldr"><b>결론 요약</b><ul>{_auto_tldr(pairwise, arms)}</ul></div>

<h2>1) 블러 세분화 히트맵 (pad 6% 고정)</h2>
<p class="hint">σ 2 → 4 → 6 → 8 → 12 → 16 → 21 → 32 → 48. 값 = A 대비 이웃 순도 변화(%p).</p>
{_heatmap(pairwise, arms)}
<div class="legend"><span><i class='sw' style='background:rgba(31,122,86,.5)'></i>A보다 좋음</span>
<span><i class='sw' style='background:rgba(181,83,42,.5)'></i>A보다 나쁨</span>
<span>흐림 = 차이 불확실</span></div>
<div class="trend"><b>색상 Δ 막대</b>{_trend_lines(pairwise, arms, "color", scale)}</div>
<p class="hint">예시 프리뷰 (일부 σ)</p>
{_examples(project, arms)}

<h2>2) 속성별 지표 Top 3 (전체 실험 설정 · A 기준)</h2>
{_top3_per_attr(_collect_all_variant_pool(pairwise, arms_data, arms, bank=bank))}

<h2>참고</h2>
<details open><summary>설계</summary>
<ul>
<li>기준선: 항상 <b>A(원본)</b></li>
<li>고정: pad_ratio=0.06, dilate_px=7</li>
<li>스윕: background_blur_sigma ∈ {{2,4,6,8,12,16,21,32,48}}</li>
<li>주 지표: purity@10 A 대비 Δ (bootstrap 2000)</li>
<li>Top 3: soft_consistency 패딩·블러 사다리 + 본 실험 세분화 블러를 <b>한 풀</b>에서 산정</li>
</ul>
</details>
<details><summary>산출물</summary>
<ul>
<li>임베딩: <code>embeddings/soft_consistency/</code></li>
<li>지표: <code>metrics/cluster_quality.json</code></li>
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
