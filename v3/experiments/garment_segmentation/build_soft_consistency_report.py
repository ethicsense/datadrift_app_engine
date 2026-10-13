"""Report for soft_consistency_pad_blur ladders (A baseline)."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

ATTRS = ("pattern", "sleeve", "neck", "color", "fit")
ATTR_KO = {"pattern": "무늬", "sleeve": "소매", "neck": "넥라인", "color": "색상", "fit": "핏"}

PAD_ARMS = ("pad_r0", "pad_r06", "soft_ref", "pad_r20", "pad_r30")
BLUR_ARMS = ("blur_s04", "blur_s08", "soft_ref", "blur_s40", "blur_s60")
ARM_INFO = {
    "A": ("원본", "기준"),
    "pad_r0": ("pad 0%", "크롭 패딩 없음"),
    "pad_r06": ("pad 6%", "타이트"),
    "soft_ref": ("pad 12% / blur 21", "기존 soft 기준점"),
    "pad_r20": ("pad 20%", "넉넉"),
    "pad_r30": ("pad 30%", "매우 넉넉"),
    "blur_s04": ("blur σ=4", "아주 약함"),
    "blur_s08": ("blur σ=8", "약함"),
    "blur_s40": ("blur σ=40", "강함"),
    "blur_s60": ("blur σ=60", "매우 강함"),
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
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-top:12px}
summary{cursor:pointer;font-weight:600}
code{background:#eef2ef;padding:1px 5px;border-radius:4px;font-size:12px}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/soft_consistency_pad_blur-20261007",
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


def _heatmap(pairwise: dict[str, Any], arms: tuple[str, ...], title_prefix: str) -> str:
    scale = max(
        abs(pairwise[f"A_vs_{arm}"][a]["delta"])
        for arm in arms
        for a in ATTRS
        if f"A_vs_{arm}" in pairwise and a in pairwise[f"A_vs_{arm}"]
    )
    head = (
        f"<tr><th class='l'>{escape(title_prefix)}</th>"
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


def _consistency_note(pairwise: dict[str, Any], arms: tuple[str, ...], axis: str) -> str:
    """Simple monotonicity check on mean Δ across attrs with significant positive count."""
    lines = []
    for a in ATTRS:
        deltas = [pairwise[f"A_vs_{arm}"][a]["delta"] for arm in arms]
        # Spearmann-like: count adjacent increases
        ups = sum(1 for i in range(len(deltas) - 1) if deltas[i + 1] > deltas[i] + 1e-6)
        downs = sum(1 for i in range(len(deltas) - 1) if deltas[i + 1] < deltas[i] - 1e-6)
        if ups >= 3 and downs <= 1:
            trend = "대체로 증가"
        elif downs >= 3 and ups <= 1:
            trend = "대체로 감소"
        else:
            trend = "비단조/혼재"
        best_i = max(range(len(arms)), key=lambda i: deltas[i])
        lines.append(
            f"<li><b>{ATTR_KO[a]}</b>: {trend} · 최고 {escape(ARM_INFO[arms[best_i]][0])} "
            f"({deltas[best_i]*100:+.1f}%p)</li>"
        )
    return f"<ul>{''.join(lines)}</ul><p class='hint'>{escape(axis)} 구간에서 속성별로 Δ가 단조성(일관성)을 보이는지 요약합니다.</p>"


def _examples(root: Path, arms: tuple[str, ...]) -> str:
    cols = ("A",) + arms
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


def _top3_per_attr(
    pairwise: dict[str, Any],
    arms_data: dict[str, Any],
    arms: tuple[str, ...],
    title: str,
) -> str:
    """Per attribute: top-3 arms by purity@10 Δ vs A."""
    blocks = [
        f"<h3>{escape(title)}</h3>",
        "<p class='hint'>속성별 Top 3는 패딩·블러 축을 나누지 않고 <b>실험한 모든 설정</b>에서 산정합니다. "
        "Δ·CI는 A 기준(%p), purity는 암 절대값 (괄호 A).</p>",
    ]
    for attr in ATTRS:
        ranked = sorted(
            arms,
            key=lambda arm: pairwise[f"A_vs_{arm}"][attr]["delta"],
            reverse=True,
        )[:3]
        a_purity = float(arms_data["A"][attr]["purity_at_k"])
        rows = [
            "<tr><th class='l'>#</th><th>설정</th><th>A 대비 Δ</th><th>95% CI</th><th>purity@10</th></tr>"
        ]
        for i, arm in enumerate(ranked, 1):
            d = pairwise[f"A_vs_{arm}"][attr]
            arm_p = float(arms_data[arm][attr]["purity_at_k"])
            rows.append(
                "<tr>"
                f"<td class='l'>{i}</td>"
                f"<td>{escape(ARM_INFO[arm][0])}</td>"
                f"<td><b>{d['delta']*100:+.1f}%p</b></td>"
                f"<td><small>[{d['ci_low']*100:+.1f}, {d['ci_high']*100:+.1f}]</small></td>"
                f"<td>{arm_p*100:.1f}% <small>(A {a_purity*100:.1f}%)</small></td>"
                "</tr>"
            )
        blocks.append(
            f"<div class='trend'><b>{ATTR_KO[attr]}</b><table>{''.join(rows)}</table></div>"
        )
    return "".join(blocks)


def _auto_tldr(pairwise: dict[str, Any]) -> str:
    def best(arms: tuple[str, ...]) -> str:
        scores = {}
        for arm in arms:
            scores[arm] = sum(
                pairwise[f"A_vs_{arm}"][a]["delta"]
                for a in ATTRS
                if _sig(pairwise[f"A_vs_{arm}"][a]) >= 0
            )
        arm = max(scores, key=scores.get)
        return f"{ARM_INFO[arm][0]} ({arm})"

    return (
        f"<li>패딩 구간 최고(유의·비악화 Δ 합): <b>{escape(best(PAD_ARMS))}</b></li>"
        f"<li>블러 구간 최고(유의·비악화 Δ 합): <b>{escape(best(BLUR_ARMS))}</b></li>"
        "<li>모든 비교는 원본 <b>A</b> 대비 purity@10 Δ (bootstrap 2000)입니다.</li>"
        "<li>가운데 <b>soft_ref</b>(pad 12% · blur 21)는 기존 soft mask 기준점입니다.</li>"
    )


def build(project: Path, bank: str) -> Path:
    metrics = json.loads((project / "metrics" / "cluster_quality.json").read_text(encoding="utf-8"))
    summary = json.loads((project / "embeddings" / bank / "summary.json").read_text(encoding="utf-8"))
    pairwise = metrics["banks"][bank]["pairwise"]
    arms_data = metrics["banks"][bank]["arms"]
    n = summary.get("coverage", {}).get("A") or summary.get("n")

    scale = max(
        abs(pairwise[f"A_vs_{arm}"][a]["delta"])
        for arm in set(PAD_ARMS + BLUR_ARMS)
        for a in ATTRS
    ) * 1.05

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Soft-mask 패딩·블러 일관성 실험</title><style>{CSS}</style></head><body><main>
<h1>패딩·블러 강도가 바뀌어도 군집 이득이 일관적인가</h1>
<p class="lead">크롭 박스 패딩(pad_ratio)과 배경 블러(σ)를 각각 5단계로 바꿔, FashionSigLIP soft-mask 군집도가 A 대비 어떻게 움직이는지 봅니다. dilate는 7로 고정했습니다.</p>
<p class="meta">상품 {n:,}장 · bank {escape(bank)} · {escape(str(metrics.get('created_at',''))[:10])} · 소스 코호트 attribute_cluster_eval-20261002 · 경로 artifacts/garment_segmentation/{escape(project.name)}</p>

<div class="tldr"><b>결론 요약</b><ul>{_auto_tldr(pairwise)}</ul></div>

<h2>1) 패딩 구간 (blur σ=21 고정)</h2>
<p class="hint">pad_ratio 0 → 6% → 12%(ref) → 20% → 30%. 값 = A 대비 이웃 순도 변화(%p).</p>
{_heatmap(pairwise, PAD_ARMS, "속성")}
<div class="legend"><span><i class='sw' style='background:rgba(31,122,86,.5)'></i>A보다 좋음</span>
<span><i class='sw' style='background:rgba(181,83,42,.5)'></i>A보다 나쁨</span>
<span>흐림 = 차이 불확실</span></div>
<div class="trend"><b>속성별 추세</b>{_consistency_note(pairwise, PAD_ARMS, "패딩")}</div>
<div class="trend"><b>색상 Δ 막대</b>{_trend_lines(pairwise, PAD_ARMS, "color", scale)}</div>
<p class="hint">예시 (패딩 축)</p>
{_examples(project, PAD_ARMS)}

<h2>2) 블러 구간 (pad 12% 고정)</h2>
<p class="hint">σ 4 → 8 → 21(ref) → 40 → 60.</p>
{_heatmap(pairwise, BLUR_ARMS, "속성")}
<div class="legend"><span><i class='sw' style='background:rgba(31,122,86,.5)'></i>A보다 좋음</span>
<span><i class='sw' style='background:rgba(181,83,42,.5)'></i>A보다 나쁨</span></div>
<div class="trend"><b>속성별 추세</b>{_consistency_note(pairwise, BLUR_ARMS, "블러")}</div>
<div class="trend"><b>색상 Δ 막대</b>{_trend_lines(pairwise, BLUR_ARMS, "color", scale)}</div>
<p class="hint">예시 (블러 축)</p>
{_examples(project, BLUR_ARMS)}

<h2>3) 속성별 Top 3 (A 기준 purity@10 Δ)</h2>
{_top3_per_attr(pairwise, arms_data, tuple(dict.fromkeys(PAD_ARMS + BLUR_ARMS)), "패딩·블러 전체 설정")}

<h2>참고</h2>
<details open><summary>설계</summary>
<ul>
<li>기준선: 항상 <b>A(원본)</b></li>
<li>패딩 축: <code>pad_ratio</code>만 변경, dilate=7, blur=21</li>
<li>블러 축: <code>background_blur_sigma</code>만 변경, pad=0.12, dilate=7</li>
<li><code>soft_ref</code>는 기존 soft(C)와 동일 설정이며 두 축의 공통 기준점</li>
<li>지표: purity@10 A 대비 Δ, bootstrap 2000</li>
</ul>
</details>
<details><summary>산출물</summary>
<ul>
<li>임베딩: <code>embeddings/soft_consistency/</code></li>
<li>지표: <code>metrics/cluster_quality.json</code></li>
<li>부모 라벨/코호트는 symlink</li>
</ul>
</details>
</main></body></html>"""
    out = project / "report.html"
    out.write_text(html, encoding="utf-8")
    # silence unused
    _ = arms_data
    return out


def main() -> None:
    args = parse_args()
    print(build(args.project_dir, args.bank))


if __name__ == "__main__":
    main()
