"""Build report for background masking methods in the selected optimal range."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTRS = ("pattern", "sleeve", "neck", "color", "fit")
# 핏은 클래스 정의·라벨 기준이 모호해 종합 개선 순위에서 제외한다.
CORE_ATTRS = ("pattern", "sleeve", "neck", "color")
ATTR_KO = {
    "pattern": "무늬",
    "sleeve": "소매",
    "neck": "넥라인",
    "color": "색상",
    "fit": "핏",
}
METHODS = ("gaussian", "mean_fill", "desaturate", "dim", "pixelate")
METHOD_KO = {
    "gaussian": "Gaussian blur σ=8",
    "mean_fill": "적응형 평균색 채움",
    "desaturate": "배경 저채도(15%)",
    "dim": "배경 감광(65%)",
    "pixelate": "배경 픽셀화(24px)",
}
PADS = (0, 60, 120)
EXAMPLES = (
    ("3cd64dfc19420475", "그래픽/로고"),
    ("7aaccf9badfc256e", "스트라이프"),
    ("cb0571ddec533d9f", "후드"),
)
# 대표 비교: 최적 pad 6%에서 A + 5방식
EXAMPLE_ARMS = ("A",) + tuple(f"{m}_p060" for m in METHODS)
EXAMPLE_LABELS = {
    "A": "A (원본)",
    **{f"{m}_p060": METHOD_KO[m] for m in METHODS},
}

CSS = """
:root{--fg:#1d2a26;--mute:#5f6d68;--line:#e2e8e4;--bg:#f6f8f6;--pos:#1f7a56;--neg:#b5532a;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",system-ui,sans-serif}
main{max-width:1120px;margin:0 auto;padding:40px 28px 80px}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:20px;margin:42px 0 6px}
h3{font-size:15px;margin:24px 0 4px}
p.lead,p.hint,.meta{color:var(--mute)}
.meta{font-size:13px}
.tldr{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--pos);border-radius:10px;padding:16px 20px;margin-top:18px}
table{border-collapse:separate;border-spacing:0;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-top:8px}
th,td{padding:9px 10px;text-align:center;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;font-size:13px}
th{background:#eef2ef}
td.l,th.l{text-align:left}
.cell b{display:block;font-size:14px}
.cell small{display:block;color:var(--mute);font-size:10px}
.cards{display:grid;gap:12px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h3{margin:0 0 8px}
code{background:#eef2ef;padding:1px 5px;border-radius:4px;font-size:12px}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-top:12px}
.ex{display:grid;gap:8px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto;margin-top:10px}
.ex .h{font-size:11px;color:var(--mute);text-align:center;font-weight:600;line-height:1.3}
.ex .rl{font-size:13px;font-weight:600}
.ex img{width:100%;aspect-ratio:1;object-fit:contain;background:#eef1ef;border-radius:6px}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=REPO_ROOT
        / "artifacts/garment_segmentation/soft_background_methods-20261008-n2k",
    )
    parser.add_argument("--bank", default="background_methods")
    return parser.parse_args()


def _variant_name(method: str, pad: int) -> str:
    return f"{method}_p{pad:03d}"


def _sig(delta: dict[str, float]) -> int:
    if delta["ci_low"] > 0:
        return 1
    if delta["ci_high"] < 0:
        return -1
    return 0


def _tint(value: float, scale: float) -> str:
    if scale <= 0:
        return "transparent"
    alpha = min(abs(value) / scale, 1.0) * 0.52 + 0.08
    rgb = "31,122,86" if value > 0 else "181,83,42"
    return f"rgba({rgb},{alpha:.2f})"


def _heatmap(pairwise: dict[str, Any], attr: str, scale: float) -> str:
    head = (
        "<tr><th class='l'>방식</th>"
        + "".join(f"<th>pad {p / 10:.0f}%</th>" for p in PADS)
        + "</tr>"
    )
    rows = []
    for method in METHODS:
        cells = []
        for pad in PADS:
            arm = _variant_name(method, pad)
            delta = pairwise[f"A_vs_{arm}"][attr]
            sig = _sig(delta)
            value = delta["delta"] * 100
            cells.append(
                f"<td class='cell' style='background:{_tint(delta['delta'], scale) if sig else 'transparent'}'>"
                f"<b>{value:+.1f}%p</b><small>{'유의' if sig else '불확실'}</small></td>"
            )
        rows.append(
            f"<tr><td class='l'><b>{escape(METHOD_KO[method])}</b></td>{''.join(cells)}</tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def _top5(
    pairwise: dict[str, Any], arms_data: dict[str, Any], variants: list[str]
) -> str:
    cards = []
    for attr in ATTRS:
        ranked = sorted(
            variants,
            key=lambda arm: pairwise[f"A_vs_{arm}"][attr]["delta"],
            reverse=True,
        )[:5]
        base = float(arms_data["A"][attr]["purity_at_k"])
        rows = [
            "<tr><th class='l'>#</th><th class='l'>조건</th><th>A 대비 Δ</th>"
            "<th>95% CI</th><th>purity@10</th></tr>"
        ]
        for rank, arm in enumerate(ranked, 1):
            method, pad_text = arm.rsplit("_p", 1)
            delta = pairwise[f"A_vs_{arm}"][attr]
            purity = float(arms_data[arm][attr]["purity_at_k"])
            label = f"{METHOD_KO[method]} · pad {int(pad_text) / 10:.0f}%"
            rows.append(
                "<tr>"
                f"<td class='l'>{rank}</td><td class='l'>{escape(label)}</td>"
                f"<td><b>{delta['delta']*100:+.1f}%p</b></td>"
                f"<td>[{delta['ci_low']*100:+.1f}, {delta['ci_high']*100:+.1f}]</td>"
                f"<td>{purity*100:.1f}% <small>(A {base*100:.1f}%)</small></td>"
                "</tr>"
            )
        cards.append(
            f"<div class='card'><h3>{ATTR_KO[attr]} Top 5</h3>"
            f"<table>{''.join(rows)}</table></div>"
        )
    return f"<div class='cards'>{''.join(cards)}</div>"


def _overall(pairwise: dict[str, Any], variants: list[str]) -> list[tuple[float, str]]:
    scores = []
    for arm in variants:
        values = [pairwise[f"A_vs_{arm}"][attr]["delta"] for attr in ATTRS]
        scores.append((sum(values) / len(values), arm))
    return sorted(scores, reverse=True)


def _examples(project: Path, arms: tuple[str, ...], labels: dict[str, str]) -> str:
    cols = arms
    n = len(cols)
    grid = f"grid-template-columns:88px repeat({n},minmax(96px,1fr))"
    rows = ["<div class='h'></div>"] + [
        f"<div class='h'>{escape(labels.get(a, a))}</div>" for a in cols
    ]
    shown = 0
    for item_id, label in EXAMPLES:
        paths = []
        ok = True
        for arm in cols:
            path = project / "previews" / arm / f"{item_id}.jpg"
            if not path.is_file():
                ok = False
                break
            paths.append(path)
        if not ok:
            continue
        rows.append(f"<div class='rl'>{escape(label)}</div>")
        for arm, path in zip(cols, paths):
            rel = path.relative_to(project).as_posix()
            rows.append(f"<img src='{escape(rel)}' alt='{escape(arm)}'>")
        shown += 1
    if shown == 0:
        return "<p class='hint'>프리뷰 이미지가 없습니다. <code>previews/</code>를 생성한 뒤 리포트를 다시 빌드하세요.</p>"
    return f"<div class='ex' style='{grid}'>{''.join(rows)}</div>"


def _pad_examples(project: Path, method: str) -> str:
    """한 방식의 pad 0/6/12% 비교."""
    arms = ("A",) + tuple(_variant_name(method, pad) for pad in PADS)
    labels = {
        "A": "A (원본)",
        **{
            _variant_name(method, pad): f"pad {pad / 10:.0f}%"
            for pad in PADS
        },
    }
    return _examples(project, arms, labels)


def _overall_top5(
    pairwise: dict[str, Any], variants: list[str], scale: float
) -> str:
    """종합 Top5 히트맵: 행=속성, 열={순위, 기법}."""
    ranked: list[tuple[float, float, str]] = []
    for arm in variants:
        deltas = [pairwise[f"A_vs_{arm}"][attr]["delta"] for attr in ATTRS]
        mean_d = sum(deltas) / len(deltas)
        worst_d = min(deltas)
        ranked.append((mean_d, worst_d, arm))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = ranked[:5]

    col_labels: list[str] = []
    top_arms: list[str] = []
    for rank, (mean_d, _worst, arm) in enumerate(top, 1):
        method, pad_text = arm.rsplit("_p", 1)
        technique = f"{METHOD_KO[method]} · pad {int(pad_text) / 10:.0f}%"
        col_labels.append(f"{rank}. {technique}")
        top_arms.append(arm)

    head = (
        "<tr><th class='l'>속성</th>"
        + "".join(f"<th>{escape(label)}</th>" for label in col_labels)
        + "</tr>"
    )
    body = []
    for attr in ATTRS:
        cells = []
        for arm in top_arms:
            delta = pairwise[f"A_vs_{arm}"][attr]
            sig = _sig(delta)
            value = delta["delta"] * 100
            bg = _tint(delta["delta"], scale) if sig else "transparent"
            cells.append(
                f"<td class='cell' style='background:{bg}'>"
                f"<b>{value:+.1f}%p</b>"
                f"<small>{'유의' if sig else '불확실'}</small></td>"
            )
        body.append(
            f"<tr><td class='l'><b>{ATTR_KO[attr]}</b></td>{''.join(cells)}</tr>"
        )

    criterion = (
        "<p class='hint'>선정 기준: <b>5속성(무늬·소매·넥·색·핏) purity@10의 A 대비 Δ 산술평균</b> "
        "내림차순. 동점이면 최악 속성 Δ가 더 높은 쪽을 우선합니다. "
        "열 헤더는 <code>{순위}. {기법}</code> 형식입니다. "
        "색상 Δ와 핏이 평균을 끌어올리므로, 의사결정용 종합 수치는 아래 핏 제외 표를 봅니다.</p>"
    )
    return criterion + f"<table>{head}{''.join(body)}</table>"


def _arm_label(arm: str) -> str:
    method, pad_text = arm.rsplit("_p", 1)
    return f"{METHOD_KO[method]} · pad {int(pad_text) / 10:.0f}%"


def _rank_mean(
    pairwise: dict[str, Any], variants: list[str], attrs: tuple[str, ...]
) -> list[tuple[float, float, str]]:
    ranked: list[tuple[float, float, str]] = []
    for arm in variants:
        deltas = [pairwise[f"A_vs_{arm}"][attr]["delta"] for attr in attrs]
        ranked.append((sum(deltas) / len(deltas), min(deltas), arm))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return ranked


def _improvement(
    pairwise: dict[str, Any], arms_data: dict[str, Any], variants: list[str]
) -> str:
    """절대 purity와 핏 제외 4속성 종합 개선."""
    base_rows = [
        "<tr><th class='l'>속성</th><th>A purity@10</th><th class='l'>해석</th></tr>"
    ]
    notes = {
        "pattern": "중상",
        "sleeve": "이미 높음",
        "neck": "이미 높음",
        "color": "가장 약함 · 잔여 갭의 중심",
        "fit": "라벨 기준이 모호해 종합 순위에서 제외",
    }
    bases = {attr: float(arms_data["A"][attr]["purity_at_k"]) for attr in ATTRS}
    for attr in ATTRS:
        base_rows.append(
            "<tr>"
            f"<td class='l'><b>{ATTR_KO[attr]}</b></td>"
            f"<td>{bases[attr]*100:.1f}%</td>"
            f"<td class='l'>{escape(notes[attr])}</td>"
            "</tr>"
        )
    core_base = sum(bases[attr] for attr in CORE_ATTRS) / len(CORE_ATTRS)
    base_rows.append(
        "<tr>"
        "<td class='l'><b>4속성 평균</b> <small>(핏 제외)</small></td>"
        f"<td><b>{core_base*100:.1f}%</b></td>"
        "<td class='l'>무늬·소매·넥·색</td>"
        "</tr>"
    )

    ranked5 = _rank_mean(pairwise, variants, ATTRS)
    rank5 = {arm: index for index, (_, _, arm) in enumerate(ranked5, 1)}
    ranked4 = _rank_mean(pairwise, variants, CORE_ATTRS)

    rows = [
        "<tr><th class='l'>#</th><th class='l'>조건</th><th>4속성 평균 Δ</th>"
        "<th>절대 평균</th><th>무늬 / 소매 / 넥 / 색</th><th>5속성 순위</th></tr>"
    ]
    for rank, (mean_d, _worst, arm) in enumerate(ranked4[:5], 1):
        deltas = [pairwise[f"A_vs_{arm}"][attr]["delta"] for attr in CORE_ATTRS]
        abs_mean = sum(
            float(arms_data[arm][attr]["purity_at_k"]) for attr in CORE_ATTRS
        ) / len(CORE_ATTRS)
        detail = " / ".join(f"{delta*100:+.1f}" for delta in deltas)
        rows.append(
            "<tr>"
            f"<td class='l'>{rank}</td>"
            f"<td class='l'>{escape(_arm_label(arm))}</td>"
            f"<td><b>{mean_d*100:+.2f}%p</b></td>"
            f"<td>{abs_mean*100:.1f}% <small>(A {core_base*100:.1f}%)</small></td>"
            f"<td>{escape(detail)}</td>"
            f"<td>#{rank5[arm]}</td>"
            "</tr>"
        )

    top_mean, _, top_arm = ranked4[0]
    second_mean, _, second_arm = ranked4[1]
    dim_arm = "dim_p060"
    dim_rank4 = next(i for i, (_, _, arm) in enumerate(ranked4, 1) if arm == dim_arm)
    note = (
        "<p class='hint'>핏은 클래스 개념과 라벨 기준이 모호해 개선 대상으로 두고, "
        "종합 개선은 <b>무늬·소매·넥라인·색상</b> purity@10 Δ의 산술평균으로 봅니다. "
        f"1·2위는 <b>{escape(_arm_label(top_arm))}</b> ({top_mean*100:+.2f}%p)와 "
        f"<b>{escape(_arm_label(second_arm))}</b> ({second_mean*100:+.2f}%p)로 사실상 동률입니다. "
        f"5속성 #{rank5[dim_arm]}위였던 {escape(_arm_label(dim_arm))}는 핏 Δ 덕분에 올라간 순위라 "
        f"핏 제외 시 #{dim_rank4}위입니다. "
        "절대 수준은 소매·넥이 이미 높고, 남는 갭은 색상(약 40% → 43%대)에 있습니다.</p>"
    )
    return (
        "<h3>A 절대 수준</h3>"
        f"<table>{''.join(base_rows)}</table>"
        "<h3>핏 제외 종합 Top 5</h3>"
        + note
        + f"<table>{''.join(rows)}</table>"
    )


def build(project: Path, bank: str) -> Path:
    metrics = json.loads(
        (project / "metrics/cluster_quality.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (project / "embeddings" / bank / "summary.json").read_text(encoding="utf-8")
    )
    payload = metrics["banks"][bank]
    pairwise = payload["pairwise"]
    arms_data = payload["arms"]
    variants = [
        _variant_name(method, pad) for method in METHODS for pad in PADS
    ]
    scale = (
        max(
            abs(pairwise[f"A_vs_{arm}"][attr]["delta"])
            for arm in variants
            for attr in ATTRS
        )
        * 1.05
    )
    ranked4 = _rank_mean(pairwise, variants, CORE_ATTRS)
    top4_mean, _, top4_arm = ranked4[0]
    second4_mean, _, second4_arm = ranked4[1]
    n = summary["coverage"]["A"]

    heatmaps = "".join(
        f"<h3>{ATTR_KO[attr]}</h3>{_heatmap(pairwise, attr, scale)}"
        for attr in ATTRS
    )
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>배경 마스킹 방식 비교</title><style>{CSS}</style></head><body><main>
<h1>최적 pad·mask 범위에서 배경 처리 방식 비교</h1>
<p class="lead">n2k pad×blur 결과로 선정한 pad 0–12%, dilate 7, feather σ=5에서 비교 가치가 있는 배경 억제 방식만 A와 비교합니다.</p>
<p class="meta">상품 {n:,}장 · 15조건 + A · purity@10 bootstrap 2000 · {escape(project.name)}</p>

<div class="tldr"><b>종합 개선 (핏 제외)</b><ul>
<li>의사결정 지표: 무늬·소매·넥·색 purity@10 Δ 평균. 핏은 라벨 기준이 모호해 순위에서 제외</li>
<li>동률 1위: <b>{escape(_arm_label(top4_arm))}</b> ({top4_mean*100:+.2f}%p), <b>{escape(_arm_label(second4_arm))}</b> ({second4_mean*100:+.2f}%p)</li>
<li>유효 범위: pad <b>0–12%</b>, blur σ <b>4–12</b>; dilate 0은 sleeve/neck/fit 저하로 제외</li>
</ul></div>

<h2>1) 처리 방식 예시</h2>
<p class="hint">동일 상품에 대해 <b>pad 6%</b>·dilate 7·feather σ=5를 고정하고 배경 처리만 바꿉니다. 왼쪽이 원본 A입니다.</p>
{_examples(project, EXAMPLE_ARMS, EXAMPLE_LABELS)}
<details><summary>방식별 pad 0% / 6% / 12% 비교</summary>
{"".join(
    f"<h3>{escape(METHOD_KO[method])}</h3>{_pad_examples(project, method)}"
    for method in METHODS
)}
</details>

<h2>2) 속성별 방식 × pad 히트맵</h2>
<p class="hint">값은 모두 원본 A 대비 purity@10 Δ(%p). 물음표/불확실은 95% CI가 0을 포함합니다.</p>
{heatmaps}

<h2>3) 속성별 Top 5</h2>
{_top5(pairwise, arms_data, variants)}

<h2>4) 종합 Top 5 (5속성)</h2>
{_overall_top5(pairwise, variants, scale)}

<h2>5) 종합 개선 (핏 제외)</h2>
{_improvement(pairwise, arms_data, variants)}

<h2>6) 비교 방식 선정</h2>
<details open><summary>포함</summary><ul>
<li>Gaussian σ=8: 현재 최적 기준</li>
<li>적응형 평균색 채움: 검정 hard mask의 분포 충격 없이 배경 정보를 제거</li>
<li>저채도 15%: 배경 색상 간섭만 억제하고 형태·밝기를 보존</li>
<li>감광 65%: 문맥과 색을 유지하면서 의류 주목도를 높임</li>
<li>24px 픽셀화: 저주파 구조는 남기고 배경 디테일을 제거</li>
</ul></details>
<details><summary>제외</summary><ul>
<li>랜덤 노이즈: 임베딩 분산 증가 가능성이 큼</li>
<li>검정 hard fill: 기존 D에서 sleeve/neck/fit 저하 확인</li>
<li>bilateral/median blur: 배경 경계를 보존해 억제 목적과 맞지 않거나 Gaussian과 중복</li>
</ul></details>
</main></body></html>"""
    out = project / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    print(build(args.project_dir, args.bank))


if __name__ == "__main__":
    main()
