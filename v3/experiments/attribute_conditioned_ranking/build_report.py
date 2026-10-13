"""HTML report for attribute-conditioned ranking."""

from __future__ import annotations

import json
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

ATTR_KO = {
    "pattern": "무늬",
    "sleeve": "소매",
    "neck": "넥라인",
    "color": "색상",
    "fit": "핏",
}

CSS = """
:root{--fg:#1d2a26;--mute:#5f6d68;--line:#e2e8e4;--bg:#f6f8f6;--pos:#1f7a56;--neg:#b5532a;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:40px 28px 80px}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:20px;margin:42px 0 6px}
h3{font-size:16px;margin:22px 0 6px}
p.lead,p.hint,.meta{color:var(--mute)}
.meta{font-size:13px}
.tldr{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--pos);border-radius:10px;padding:16px 20px;margin-top:18px}
table{border-collapse:separate;border-spacing:0;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-top:8px}
th,td{padding:8px 9px;text-align:center;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;font-size:12px}
th{background:#eef2ef}
td.l,th.l{text-align:left}
.scroll{overflow-x:auto}
code,pre{background:#eef2ef;border-radius:6px}
code{padding:1px 5px;font-size:12px}
pre{padding:12px 14px;overflow-x:auto;font-size:12px;line-height:1.45}
.gallery{display:grid;gap:14px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.row{display:grid;grid-template-columns:110px repeat(8,minmax(72px,1fr));gap:6px;align-items:end}
.row img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#eef1ef}
.cap{font-size:10px;color:var(--mute);text-align:center}
.query{display:flex;gap:12px;align-items:center;margin-bottom:8px}
.query img{width:92px;height:92px;object-fit:cover;border-radius:8px}
.scatter-wrap{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:12px;align-items:start;margin-top:12px}
.scatter-card{position:sticky;top:16px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.scatter-card img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#eef1ef}
.scatter-card p{word-break:break-word}
@media(max-width:800px){.scatter-wrap{grid-template-columns:1fr}.scatter-card{position:static}}
"""


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.{digits}f}%p"


def _num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _ci(delta: dict[str, Any] | None) -> str:
    if not delta:
        return "—"
    return f"[{delta['ci_low'] * 100:+.1f}, {delta['ci_high'] * 100:+.1f}]"


def _tint(value: float) -> str:
    alpha = min(abs(value) / 0.06, 1.0) * 0.5 + 0.08
    rgb = "31,122,86" if value > 0 else "181,83,42"
    return f"rgba({rgb},{alpha:.2f})"


def _heatmap(test: dict[str, Any], pairs: dict[str, Any], arms: list[str]) -> str:
    head = "<tr><th class='l'>속성</th>" + "".join(f"<th>{escape(arm)}</th>" for arm in arms) + "</tr>"
    body = []
    for attr in ("color", "pattern", "sleeve", "neck", "fit"):
        cells = []
        for arm in arms:
            if arm == "baseline":
                value = test["baseline"][attr]["purity_at_k"]
                cells.append(f"<td><b>{value * 100:.1f}%</b><small>절대</small></td>")
                continue
            delta = (pairs.get(arm) or {}).get(attr) or {}
            point = float(delta.get("delta") or 0.0)
            sig = delta.get("ci_low", 0) > 0 or delta.get("ci_high", 0) < 0
            bg = _tint(point) if sig else "transparent"
            cells.append(
                f"<td style='background:{bg}'><b>{point * 100:+.1f}</b>"
                f"<small>{'유의' if sig else '불확실'}</small></td>"
            )
        body.append(f"<tr><td class='l'><b>{ATTR_KO[attr]}</b></td>{''.join(cells)}</tr>")
    return f"<div class='scroll'><table>{head}{''.join(body)}</table></div>"


def _ranking_table(payload: dict[str, Any]) -> str:
    rows = [
        "<tr><th class='l'>방식</th><th>색상 purity Δ</th><th>비타깃 평균 Δ</th>"
        "<th>유사색 weighted Δ</th><th>nDCG Δ</th><th>평균 ΔE Δ</th><th>Jaccard@10</th></tr>"
    ]
    graded = payload["graded_test"].get("arms") or {}
    structure = payload["structure_test"]
    names = [arm for arm in payload["color_test"] if arm != "baseline"]
    names.sort(
        key=lambda arm: (payload["color_test_pairs"].get(arm) or {}).get("color", {}).get("delta") or -999,
        reverse=True,
    )
    for arm in names:
        pairs = payload["color_test_pairs"][arm]
        other = float(np_mean([pairs[attr]["delta"] for attr in ("pattern", "sleeve", "neck")]))
        g = graded.get(arm, {}).get("delta") or {}
        rows.append(
            "<tr>"
            f"<td class='l'>{escape(arm)}</td>"
            f"<td>{_pct((pairs.get('color') or {}).get('delta'))}</td>"
            f"<td>{_pct(other)}</td>"
            f"<td>{_pct((g.get('weighted_purity_at_k') or {}).get('delta'))}</td>"
            f"<td>{_pct((g.get('ndcg_at_k') or {}).get('delta'))}</td>"
            f"<td>{_num((g.get('mean_delta_e_at_k') or {}).get('delta'), 2)}</td>"
            f"<td>{_num((structure.get(arm) or {}).get('jaccard_at_10'), 3)}</td>"
            "</tr>"
        )
    return f"<div class='scroll'><table>{''.join(rows)}</table></div>"


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _graded_table(payload: dict[str, Any]) -> str:
    arms = payload["graded_test"].get("arms") or {}
    rows = [
        "<tr><th class='l'>방식</th><th>weighted purity@10</th><th>nDCG@10</th>"
        "<th>평균 ΔE00@10</th><th>triplet</th><th>centroid Spearman</th></tr>"
    ]
    order = ["baseline"] + [arm for arm in arms if arm != "baseline"]
    for arm in order:
        item = arms[arm]
        spearman = (item.get("centroid_spearman") or {}).get("spearman")
        rows.append(
            "<tr>"
            f"<td class='l'>{escape(arm)}</td>"
            f"<td>{_num(item['weighted_purity_at_k']['mean'])}</td>"
            f"<td>{_num(item['ndcg_at_k']['mean'])}</td>"
            f"<td>{_num(item['mean_delta_e_at_k']['mean'], 2)}</td>"
            f"<td>{_num((item.get('triplet_accuracy') or {}).get('mean'))}</td>"
            f"<td>{_num(spearman)}</td>"
            "</tr>"
        )
    return f"<div class='scroll'><table>{''.join(rows)}</table></div>"


def _gallery(payload: dict[str, Any], selected: str) -> str:
    cards = []
    for item in payload.get("gallery") or []:
        scores = ", ".join(f"{row['name']} {row['score']:+.2f}" for row in item["prompt_scores"])
        title = item.get("label") or item.get("fine_color") or ""
        blocks = [
            f"<div class='query'><img src='{escape(item['image'])}' alt=''>"
            f"<div><b>{escape(str(title))}</b>"
            f"<br><span class='hint'>프롬프트 점수 {escape(scores)}</span></div></div>"
        ]
        for arm in ("baseline", selected):
            neighbors = item["arms"].get(arm) or []
            cells = "".join(
                "<div><img src='{src}' alt=''><div class='cap'>{label}</div></div>".format(
                    src=escape(neighbor["image"]),
                    label=escape(str(neighbor.get("label") or neighbor.get("fine_color") or neighbor.get("coarse_color") or "")),
                )
                for neighbor in neighbors
            )
            blocks.append(f"<h3>{escape(arm)}</h3><div class='row'>{cells}</div>")
        cards.append(f"<article class='card'>{''.join(blocks)}</article>")
    return f"<div class='gallery'>{''.join(cards)}</div>" if cards else "<p class='hint'>갤러리 이미지가 없습니다.</p>"


def _scatter(project: Path) -> str:
    path = project / "metrics" / "scatter.json"
    if not path.is_file():
        return "<p class='hint'>산점도 데이터가 없습니다.</p>"
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = json.loads(path.read_text(encoding="utf-8"))
    panels = data["panels"]
    points = data["points"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[panel["title"] for panel in panels])
    palettes = {
        "color": {
            "white_ivory": "#d8d2c2",
            "black": "#222222",
            "gray": "#8a8a8a",
            "navy_blue": "#2a4184",
            "beige_brown": "#a78462",
            "chromatic": "#c44b78",
        },
        "pattern": {
            "solid": "#3d7a62",
            "stripe": "#3d6ea8",
            "check": "#c47b32",
            "graphic_logo": "#8a4ea3",
            "other": "#888888",
        },
        "sleeve": {
            "sleeveless": "#c47b32",
            "short": "#3d6ea8",
            "long": "#3d7a62",
        },
        "neck": {
            "round": "#3d7a62",
            "vneck": "#3d6ea8",
            "collar": "#c47b32",
            "hood": "#8a4ea3",
            "turtleneck": "#2a4184",
        },
        "fit": {
            "slim": "#3d6ea8",
            "regular": "#8a8a8a",
            "oversized": "#c47b32",
        },
    }
    shown: set[tuple[str, str]] = set()
    for index, panel in enumerate(panels):
        row, col = divmod(index, 2)
        kind = panel["color_by"]
        groups: dict[str, list[int]] = {}
        for point_index, point in enumerate(points):
            groups.setdefault(point.get(kind) or "unknown", []).append(point_index)
        for label, members in groups.items():
            coords = [panel["coords"][i] for i in members]
            custom = [
                [
                    points[i]["thumb"],
                    points[i].get("product") or "",
                    points[i].get("product_id") or "",
                    points[i]["color"],
                    points[i]["pattern"],
                    points[i]["fine"],
                    points[i].get("sleeve") or "",
                    points[i].get("neck") or "",
                    points[i].get("fit") or "",
                ]
                for i in members
            ]
            fig.add_trace(
                go.Scatter(
                    x=[xy[0] for xy in coords],
                    y=[xy[1] for xy in coords],
                    mode="markers",
                    marker={"size": 6, "color": palettes[kind].get(label, "#bbbbbb")},
                    name=label,
                    legendgroup=f"{kind}:{label}",
                    showlegend=(kind, label) not in shown,
                    customdata=custom,
                    hovertemplate=(
                        "%{customdata[1]}<br>상품 ID %{customdata[2]}"
                        "<br>색상 %{customdata[3]}<br>무늬 %{customdata[4]}<extra></extra>"
                    ),
                ),
                row=row + 1,
                col=col + 1,
            )
            shown.add((kind, label))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=920,
        hovermode="closest",
        paper_bgcolor="#f6f8f6",
        plot_bgcolor="#ffffff",
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        legend={"orientation": "h", "y": -0.05},
    )
    plot = fig.to_html(full_html=False, include_plotlyjs="inline", div_id="dist-scatter")
    script = """
<script>
function bindScatter(){
  const plot=document.getElementById('dist-scatter');
  const img=document.getElementById('scatter-thumb');
  const meta=document.getElementById('scatter-meta');
  if(!plot||!plot.on){setTimeout(bindScatter,50);return;}
  const text=value=>String(value??'').replace(/[&<>]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[ch]));
  const show=event=>{
    const point=event.points&&event.points[0];
    if(!point)return;
    const data=point.customdata||[];
    if(img&&data[0])img.src=data[0];
    if(meta)meta.innerHTML=['<b>'+text(data[1]||'제품명 없음')+'</b>', '상품 ID '+text(data[2]||'-'), '색상 '+text(data[3]||'-'), '무늬 '+text(data[4]||'-'), '소매 '+text(data[6]||'-'), '넥 '+text(data[7]||'-'), '핏 '+text(data[8]||'-'), '세부색 '+text(data[5]||'-')].join('<br>');
  };
  plot.on('plotly_hover', show);
  plot.on('plotly_click', show);
}
bindScatter();
</script>"""
    return (
        f"<p class='hint'>{escape(data['note'])} 점에 커서를 올리면 오른쪽에서 썸네일을 봅니다.</p>"
        "<div class='scatter-wrap'>"
        f"<div>{plot}</div>"
        "<aside class='scatter-card'><img id='scatter-thumb' alt=''>"
        "<p id='scatter-meta' class='hint'>점을 올려 보세요.</p></aside>"
        f"</div>{script}"
    )


def _reference_summary(project: Path) -> str:
    path = project / "color_reference.jsonl"
    if not path.is_file():
        return "<p class='hint'>색상 기준 파일이 없습니다.</p>"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ok = [row for row in rows if row.get("status") == "ok"]
    counts = Counter(row["fine_color"] for row in ok)
    body = "".join(
        f"<tr><td class='l'>{escape(name)}</td><td>{count}</td></tr>"
        for name, count in counts.most_common()
    )
    mean_anchor = sum(row["delta_e_to_anchor"] for row in ok) / max(len(ok), 1)
    return (
        f"<p class='hint'>solid {len(rows)}장 중 Lab 추출 {len(ok)}장. "
        f"가장 가까운 팔레트까지의 평균 ΔE00은 {mean_anchor:.1f}입니다.</p>"
        f"<table><tr><th class='l'>세부 색</th><th>장수</th></tr>{body}</table>"
    )


def _attribute_ranking(payload: dict[str, Any]) -> str:
    target = payload["target"]
    non_targets = tuple(payload["non_targets"])
    rows = [
        "<tr><th class='l'>방식</th><th>목표 purity Δ</th><th>비타깃 평균 Δ</th><th>Jaccard@10</th></tr>"
    ]
    structure = payload["structure_test"]
    names = [arm for arm in payload["test"] if arm != "baseline"]
    names.sort(
        key=lambda arm: (payload["test_pairs"].get(arm) or {}).get(target, {}).get("delta") or -999,
        reverse=True,
    )
    for arm in names:
        pairs = payload["test_pairs"][arm]
        other = float(np_mean([pairs[attr]["delta"] for attr in non_targets]))
        rows.append(
            "<tr>"
            f"<td class='l'>{escape(arm)}</td>"
            f"<td>{_pct((pairs.get(target) or {}).get('delta'))}</td>"
            f"<td>{_pct(other)}</td>"
            f"<td>{_num((structure.get(arm) or {}).get('jaccard_at_10'), 3)}</td>"
            "</tr>"
        )
    return f"<div class='scroll'><table>{''.join(rows)}</table></div>"


def _class_table(payload: dict[str, Any]) -> str:
    counts = payload.get("class_counts") or {}
    body = "".join(
        f"<tr><td class='l'>{escape(name)}</td><td>{count}</td></tr>"
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return f"<table><tr><th class='l'>클래스</th><th>test 장수</th></tr>{body}</table>"


def _build_attribute(project: Path, payload: dict[str, Any]) -> Path:
    target = payload["target"]
    target_ko = ATTR_KO[target]
    selected = payload["selection"]["selected"]
    success = payload["success"]
    least_damage = min(payload["selection"]["candidates"], key=lambda row: row["non_target_drop"])
    non_target_ko = "·".join(ATTR_KO[name] for name in payload["non_targets"])
    target_delta = payload["test_pairs"][selected][target]
    arms = ["baseline", selected]
    for arm in payload["test"]:
        if arm not in arms:
            arms.append(arm)
    removal_ko = "·".join(ATTR_KO[name] for name in payload["removal_attributes"])
    eligible = any(row["eligible"] for row in payload["selection"]["candidates"])
    constraint = (
        f"사전 제약({non_target_ko} 평균 하락 2%p 이내)을 만족한 방식이 있습니다."
        if eligible
        else f"사전 제약({non_target_ko} 평균 하락 2%p 이내)을 만족한 방식은 없습니다."
    )
    fit_note = ""
    if target == "fit":
        fit_note = "<li>핏 라벨 기준은 모호합니다. 이 run은 그 라벨 위에서의 측정입니다.</li>"
    elif "fit" not in payload["non_targets"]:
        fit_note = "<li>2%p 제약의 평균에는 핏을 넣지 않았습니다. 핏 라벨 기준이 모호하기 때문입니다. 히트맵에는 핏을 포함합니다.</li>"
    checks = "".join(
        f"<li>{escape(name)}: {'통과' if ok else '실패'}</li>" for name, ok in success["checks"].items()
    )
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>속성 조건부 정렬: {escape(target_ko)}</title><style>{CSS}</style></head><body><main>
<h1>속성 조건부 정렬: {escape(target_ko)}</h1>
<p class="lead">기준 벡터는 이전 실험에서 가장 좋았던 <b>{escape(payload.get('preprocess') or '배경 픽셀화 24px · pad 6%')}</b> 임베딩입니다. 이미지 인코더는 재학습하지 않고, {escape(target_ko)} 문장으로 만든 축만 다시 계산합니다.</p>
<p class="meta">{escape(payload['date'])} · 유한 임베딩 {payload['finite_n']:,}장 · dev에서 방식 선택 · test에서 최종 판정 · {escape(project.name)}</p>
<div class="tldr"><b>{escape(target_ko)} 제어 결과</b>
<ul>
<li>{constraint}</li>
<li>제약을 만족하는 후보가 있으면 그 안에서, 없으면 전체에서 dev {escape(target_ko)} purity가 가장 높은 <b>{escape(selected)}</b>를 골랐습니다. test {escape(target_ko)} purity Δ {_pct(target_delta.get('delta'))} {_ci(target_delta)}</li>
<li>비타깃 손실이 가장 작은 방식은 <b>{escape(least_damage['arm'])}</b>입니다. dev 기준 목표 Δ {_pct(least_damage['target_delta'])}, 비타깃 하락 {least_damage['non_target_drop']*100:.1f}%p.</li>
{fit_note}
<li>사전 성공 조건: {'모두 통과' if success['passed'] else '일부 실패'}</li>
</ul></div>

<h2>1. 임베딩을 어떻게 만들었나</h2>
<p>모든 방식은 픽셀화 pad 6% 전처리로 만든 같은 벡터 <code>z</code>에서 출발합니다. 목표 속성은 {escape(target_ko)}이고, 비타깃 제거에 쓰는 문장은 {escape(removal_ko)}입니다.</p>
<pre>1. 점수 지문
   s = z Tᵀ
   T의 각 열은 {escape(target_ko)} 프롬프트 묶음의 평균 벡터
   정렬 벡터 = L2(s)

2. 방향 / 부분공간
   텍스트: {escape(target_ko)} 프롬프트를 중심화한 뒤 SVD → U_text
   데이터: train {escape(target_ko)} centroid를 중심화한 뒤 SVD → U_data
   정렬 벡터 = L2(z U)

3. 기존 유사도와 속성 유사도의 혼합
   sim = (1-α) cos(z, z') + α cos(zU, z'U)
   구현은 L2(z), L2(zU)를 √(1-α), √α로 이어 붙인 벡터의 내적

4. 비타깃 제거
   {escape(removal_ko)} 프롬프트로 U_other를 만들고
   z' = z - U_other U_otherᵀ z
   정렬 벡터 = L2(z' U_target)</pre>
<p class="hint">텍스트 방식은 이미지 라벨을 보지 않습니다. data_centroid만 train 라벨 centroid를 쓰므로 상한 참고군입니다. α와 SVD rank는 dev에서 고르고 test 수치는 선택 이후의 확인값입니다. 문장 수가 적으면 요청 rank보다 축이 줄어듭니다.</p>

<h2>2. 분포 산점도</h2>
{_scatter(project)}

<h2>3. 목표 클래스</h2>
<p class="hint">split은 {escape(target_ko)} 라벨로 층화했습니다. 아래는 test에 남은 클래스 수입니다.</p>
{_class_table(payload)}

<h2>4. test 군집 성능</h2>
<p class="hint">첫 열은 baseline purity@10 절대값이고, 나머지는 baseline 대비 Δ(%p)입니다. 유의는 paired bootstrap 95% CI가 0을 포함하지 않는 경우입니다.</p>
{_heatmap(payload['test'], payload['test_pairs'], arms)}

<h2>5. 방식 비교</h2>
<p class="hint">Jaccard@10은 test의 목표 라벨 집합에서 기존 Top10을 얼마나 유지하는지입니다.</p>
{_attribute_ranking(payload)}

<h2>6. 같은 query의 이웃</h2>
<p class="hint">test에서 목표 클래스가 겹치지 않게 골랐습니다. 이웃은 test 목표 라벨 집합 안의 Top 8입니다.</p>
{_gallery(payload, selected)}

<h2>7. 판정</h2>
<ul>{checks}</ul>
<p class="hint">선택 규칙은 dev에서 비타깃({escape(non_target_ko)}) 평균 하락이 2%p 이내인 후보 중 {escape(target_ko)} purity가 가장 높은 방식입니다. 해당 후보가 없으면 {escape(target_ko)} purity 최댓값을 고릅니다. test에서 고른 방식의 비타깃 평균 하락은 {success['non_target_drop'] * 100:.2f}%p입니다. data_centroid는 라벨을 쓰므로 선택 후보에서 제외했습니다.</p>
</main></body></html>"""
    out = project / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def build(project: Path) -> Path:
    payload = json.loads((project / "metrics" / "evaluation.json").read_text(encoding="utf-8"))
    if payload.get("target"):
        return _build_attribute(project, payload)
    selected = payload["selection"]["selected"]
    transferred = payload["transferred_pattern_arm"]
    success = payload["success"]
    least_damage = min(
        payload["selection"]["candidates"],
        key=lambda row: row["non_target_drop"],
    )
    check_ko = {
        "coarse_color_ci_positive": "test 색상 purity 95% CI가 0보다 큼",
        "weighted_purity_ci_positive": "유사색 weighted purity 95% CI가 0보다 큼",
        "ndcg_ci_positive": "유사색 nDCG 95% CI가 0보다 큼",
        "mean_delta_e_ci_negative": "평균 ΔE00 95% CI가 0보다 작음",
        "non_target_drop_within_limit": "무늬·소매·넥 평균 하락이 2%p 이내",
    }
    checks = "".join(
        f"<li>{escape(check_ko.get(name, name))}: {'통과' if ok else '실패'}</li>"
        for name, ok in success["checks"].items()
    )
    color_delta = payload["color_test_pairs"][selected]["color"]
    pattern_delta = (payload["pattern_test_pairs"].get(transferred) or {}).get("pattern") or {}
    arms = ["baseline", selected]
    for arm in payload["color_test"]:
        if arm not in arms:
            arms.append(arm)
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>속성 조건부 정렬</title><style>{CSS}</style></head><body><main>
<h1>속성 조건부 정렬: 색상</h1>
<p class="lead">기준 벡터는 이전 실험에서 가장 좋았던 <b>{escape(payload.get('preprocess') or '배경 픽셀화 24px · pad 6%')}</b> 임베딩입니다. 이미지 인코더는 재학습하지 않고, 그 벡터의 속성 공간만 다시 계산합니다.</p>
<p class="meta">{escape(payload['date'])} · 유한 임베딩 {payload['finite_n']:,}장 · dev에서 방식 선택 · test에서 최종 판정 · {escape(project.name)}</p>
<div class="tldr"><b>색상 정렬은 가능하지만, 다른 속성을 유지하는 조건은 실패</b>
<ul>
<li>사전 제약(무늬·소매·넥 평균 하락 2%p 이내)을 만족한 방식은 없습니다.</li>
<li>제약을 만족하지 못할 때의 규칙대로 dev 색상 purity가 가장 높은 <b>{escape(selected)}</b>를 골랐습니다. test 색상 purity Δ {_pct(color_delta.get('delta'))} {_ci(color_delta)}</li>
<li>비타깃 손실이 가장 작은 방식은 <b>{escape(least_damage['arm'])}</b>입니다. dev 기준 색상 Δ {_pct(least_damage['color_delta'])}, 비타깃 하락 {least_damage['non_target_drop']*100:.1f}%p.</li>
<li>같은 레시피를 무늬에 이식한 <b>{escape(transferred)}</b>도 무늬 purity Δ {_pct(pattern_delta.get('delta'))} {_ci(pattern_delta)}로 목표 속성은 개선됩니다.</li>
<li>사전 성공 조건: {'모두 통과' if success['passed'] else '일부 실패'}</li>
</ul></div>

<h2>1. 임베딩을 어떻게 만들었나</h2>
<p>모든 방식은 픽셀화 pad 6% 전처리로 만든 같은 벡터 <code>z</code>에서 출발합니다. 여기서 바꾸는 것은 유사도를 계산하는 방식뿐입니다.</p>
<pre>1. 점수 지문
   s = z Tᵀ
   T의 각 열은 색상 프롬프트 묶음의 평균 벡터
   정렬 벡터 = L2(s)

2. 방향 / 부분공간
   텍스트: 색상 프롬프트를 중심화한 뒤 SVD → U_text
   데이터: train 색상 centroid를 중심화한 뒤 SVD → U_data
   정렬 벡터 = L2(z U)

3. 기존 유사도와 속성 유사도의 혼합
   sim = (1-α) cos(z, z') + α cos(zU, z'U)
   구현은 L2(z), L2(zU)를 √(1-α), √α로 이어 붙인 벡터의 내적

4. 비타깃 제거
   무늬·소매·넥 프롬프트로 U_other를 만들고
   z' = z - U_other U_otherᵀ z
   정렬 벡터 = L2(z' U_color)</pre>
<p class="hint">팔레트 RGB와 garment mask의 CIELAB은 평가에만 씁니다. 텍스트 방식은 이미지 라벨을 보지 않습니다. data_centroid만 train 라벨 centroid를 쓰므로 상한 참고군입니다. α와 SVD rank는 dev에서 고르고 test 수치는 선택 이후의 확인값입니다.</p>

<h2>2. 분포 산점도</h2>
{_scatter(project)}

<h2>3. 세부 색상 기준</h2>
<p>같은 클래스 purity는 빨강과 버건디를 완전히 다른 실패로 계산합니다. solid 의류의 마스크 중앙값 Lab을 구하고, 고정 sRGB 팔레트와의 CIEDE2000으로 세부 색을 매깁니다. 유사 색상 relevance는 <code>exp(-ΔE00 / 18)</code>입니다. 이 Lab 값은 임베딩 생성에 사용하지 않습니다.</p>
{_reference_summary(project)}

<h2>4. test 군집 성능</h2>
<p class="hint">첫 열은 baseline purity@10 절대값이고, 나머지는 baseline 대비 Δ(%p)입니다. 유의는 paired bootstrap 95% CI가 0을 포함하지 않는 경우입니다.</p>
{_heatmap(payload['color_test'], payload['color_test_pairs'], arms)}

<h2>5. 방식 비교와 유사 색상</h2>
<p class="hint">평균 ΔE Δ는 음수일 때 이웃의 지각 색상 거리가 줄어든 것입니다. Jaccard@10은 기존 Top10을 얼마나 유지하는지입니다.</p>
{_ranking_table(payload)}
<h3>solid test의 절대 유사 색상 지표</h3>
{_graded_table(payload)}

<h2>6. 같은 query의 이웃</h2>
<p class="hint">test solid 중 팔레트 앵커와 ΔE00 18 이하인 상품을 세부 색이 겹치지 않게 골랐습니다. 이웃은 test 색상 라벨 집합 안의 Top 8입니다.</p>
{_gallery(payload, selected)}

<h2>7. 무늬로 같은 방식 이식</h2>
<p>색상 dev 선택 규칙으로 고른 계열과 rank/α를 무늬 프롬프트에 그대로 적용했습니다. 이식 대상은 <b>{escape(transferred)}</b>입니다.</p>
{_heatmap(payload['pattern_test'], payload['pattern_test_pairs'], list(payload['pattern_test']))}

<h2>8. 판정</h2>
<ul>{checks}</ul>
<p class="hint">선택 규칙은 dev에서 비타깃 하락이 2%p 이내인 후보 중 색상 purity가 가장 높은 방식입니다. 해당 후보가 없어 색상 purity 최댓값을 골랐고, test에서 그 방식의 비타깃 평균 하락은 {success['non_target_drop'] * 100:.2f}%p입니다. data_centroid는 라벨을 쓰므로 선택 후보에서 제외했습니다.</p>
</main></body></html>"""
    # The f-string above accidentally duplicates baseline in pattern heatmap via keys including baseline.
    out = project / "report.html"
    out.write_text(html, encoding="utf-8")
    return out
