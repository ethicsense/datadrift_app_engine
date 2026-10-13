"""Render the attribute cluster-quality report (sam3_abcd bank) as a readable HTML page."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

ARMS = ("A", "B", "C", "D")
ARM_INFO = {
    "A": ("원본", "이미지 전체를 그대로 사용 (기준)"),
    "B": ("크롭", "SAM3 박스에 12% 여백을 둔 잘라내기, 마스크 없음"),
    "C": ("소프트 마스크", "크롭 + 경계를 부드럽게 한 마스크, 배경은 흐리게"),
    "D": ("하드 마스크", "크롭 + 마스크 밖을 검정으로 지움"),
}
ATTRS = ("pattern", "sleeve", "neck", "color", "fit")
ATTR_KO = {"pattern": "무늬", "sleeve": "소매", "neck": "넥라인", "color": "색상", "fit": "핏"}
CLASS_KO = {
    "solid": "무지", "stripe": "스트라이프", "check": "체크", "graphic_logo": "그래픽/로고", "other": "기타 무늬",
    "sleeveless": "민소매", "short": "반팔", "long": "긴팔",
    "round": "라운드", "vneck": "V넥", "collar": "칼라", "hood": "후드", "turtleneck": "터틀넥",
    "white_ivory": "화이트/아이보리", "black": "블랙", "gray": "그레이", "navy_blue": "네이비/블루",
    "beige_brown": "베이지/브라운", "chromatic": "유채색",
    "slim": "슬림", "regular": "레귤러", "oversized": "오버사이즈",
}
EXAMPLES = (
    ("3cd64dfc19420475", "그래픽/로고"),
    ("7aaccf9badfc256e", "스트라이프"),
    ("cb0571ddec533d9f", "후드"),
)
PREVIEW_RUN = "pipeline/preprocess/preprocess-abcd-sam3-450/previews"
QUOTA = 100

CSS = """
:root{--fg:#1d2a26;--mute:#5f6d68;--line:#e2e8e4;--bg:#f6f8f6;--pos:#1f7a56;--neg:#b5532a;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:40px 28px 80px}
h1{font-size:28px;letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:20px;margin:48px 0 6px;letter-spacing:-.01em}
h3{font-size:16px;margin:0 0 10px}
p.lead{color:var(--mute);margin:0 0 4px}
p.hint{color:var(--mute);font-size:13px;margin:4px 0 14px}
.meta{color:var(--mute);font-size:13px}
.tldr{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--pos);border-radius:10px;padding:18px 22px;margin:22px 0 0}
.tldr ul{margin:6px 0 0;padding-left:20px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .arm{font-size:12px;color:var(--mute)}
.card .name{font-size:17px;font-weight:700;margin:2px 0 6px}
.card .desc{font-size:13px;color:var(--mute);min-height:40px}
.badge{display:inline-block;font-size:12px;font-weight:700;padding:2px 9px;border-radius:999px;margin-top:8px}
.b-good{background:#e1f3ea;color:var(--pos)}.b-warn{background:#fbe9df;color:var(--neg)}.b-base{background:#eceff1;color:#4a5560}.b-mid{background:#f3f1e1;color:#7a6a1f}
.score{font-size:13px;margin-top:6px}
.ex{display:grid;grid-template-columns:110px repeat(4,1fr);gap:8px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.ex .h{font-size:12px;color:var(--mute);text-align:center;font-weight:600}
.ex .rl{font-size:13px;font-weight:600}
.ex img{width:100%;aspect-ratio:1;object-fit:contain;background:#eef1ef;border-radius:6px}
table{border-collapse:separate;border-spacing:0;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:10px 12px;text-align:center;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{background:#eef2ef;font-size:13px;font-weight:600}
td.l,th.l{text-align:left}
tr:last-child td{border-bottom:none}
.cell b{display:block;font-size:16px}
.cell small{display:block;font-size:11px;color:var(--mute)}
.ns{opacity:.55}
.legend{display:flex;gap:16px;font-size:12px;color:var(--mute);margin:8px 0 0;flex-wrap:wrap}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px}
.attr{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-top:14px}
.attr-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.attr-head .take{font-size:13px;color:var(--mute)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:10px}
.bar-row{display:grid;grid-template-columns:92px 1fr 64px;gap:10px;align-items:center;font-size:13px;margin:7px 0}
.track{position:relative;height:18px;background:#f1f4f2;border-radius:4px}
.track .zero{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:#9aa6a1}
.track .bar{position:absolute;top:3px;height:12px;border-radius:3px}
.track .ci{position:absolute;top:8px;height:2px;background:#3a4743;opacity:.55}
.mini th,.mini td{padding:6px 8px;font-size:13px}
.quota{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.q{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.qrow{display:grid;grid-template-columns:96px 1fr 34px;gap:6px;align-items:center;font-size:12px;margin:5px 0}
.qtrack{position:relative;height:10px;background:#eef1ef;border-radius:3px}
.qfill{position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:#7fb39b}
.qfill.low{background:#e3a07d}
.qline{position:absolute;top:-3px;bottom:-3px;width:2px;background:#33413c}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-top:14px}
summary{cursor:pointer;font-weight:600}
dl{margin:10px 0 0}dt{font-weight:700;margin-top:12px}dd{margin:2px 0 0;color:#3c4945}
code{background:#eef2ef;padding:1px 5px;border-radius:4px;font-size:12px}
@media (max-width:860px){.cards,.quota{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/attribute_cluster_eval-20261002",
    )
    p.add_argument("--bank", default="sam3_abcd")
    return p.parse_args()


def _ko(cls: str) -> str:
    return CLASS_KO.get(cls, cls)


def _sil(arm_attr: dict[str, Any]) -> dict[str, Any]:
    return arm_attr.get("silhouette") or {}


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


def _verdict(pair: dict[str, dict[str, float]]) -> tuple[str, str, list[str], list[str]]:
    wins = [a for a in ATTRS if _sig(pair[a]) > 0]
    losses = [a for a in ATTRS if _sig(pair[a]) < 0]
    if losses:
        return "주의", "b-warn", wins, losses
    if len(wins) >= 3:
        return "추천", "b-good", wins, losses
    return "보통", "b-mid", wins, losses


def _cards(pairwise: dict[str, Any]) -> str:
    out = []
    for arm in ARMS:
        name, desc = ARM_INFO[arm]
        if arm == "A":
            badge, score = "<span class='badge b-base'>기준</span>", "<div class='score'>비교 기준선</div>"
        else:
            label, cls, wins, losses = _verdict(pairwise[f"A_vs_{arm}"])
            badge = f"<span class='badge {cls}'>{label}</span>"
            up = ", ".join(ATTR_KO[a] for a in wins) or "없음"
            down = ", ".join(ATTR_KO[a] for a in losses) or "없음"
            score = (
                f"<div class='score'><span style='color:var(--pos)'>▲ 개선</span> {escape(up)}<br>"
                f"<span style='color:var(--neg)'>▼ 악화</span> {escape(down)}</div>"
            )
        out.append(
            f"<div class='card'><div class='arm'>{arm}</div><div class='name'>{escape(name)}</div>"
            f"<div class='desc'>{escape(desc)}</div>{badge}{score}</div>"
        )
    return f"<div class='cards'>{''.join(out)}</div>"


def _examples(root: Path) -> str:
    rows = ["<div class='h'></div>"] + [f"<div class='h'>{a} · {escape(ARM_INFO[a][0])}</div>" for a in ARMS]
    for item_id, label in EXAMPLES:
        if not all((root / PREVIEW_RUN / a / f"{item_id}.jpg").is_file() for a in ARMS):
            continue
        rows.append(f"<div class='rl'>{escape(label)}</div>")
        rows += [f"<img src='{PREVIEW_RUN}/{a}/{item_id}.jpg' alt='{a}'>" for a in ARMS]
    return f"<div class='ex'>{''.join(rows)}</div>"


def _heatmap(pairwise: dict[str, Any]) -> str:
    scale = max(abs(pairwise[f"A_vs_{arm}"][a]["delta"]) for arm in ARMS[1:] for a in ATTRS)
    head = "<tr><th class='l'>속성</th>" + "".join(
        f"<th>{arm} · {escape(ARM_INFO[arm][0])}</th>" for arm in ARMS[1:]
    ) + "<th>가장 좋은 전처리</th></tr>"
    body = []
    for a in ATTRS:
        cells = []
        best_arm, best_val = "A", 0.0
        for arm in ARMS[1:]:
            d = pairwise[f"A_vs_{arm}"][a]
            sig = _sig(d)
            pp = d["delta"] * 100
            note = "유의" if sig else "차이 불확실"
            cls = "cell" if sig else "cell ns"
            bg = _tint(d["delta"], scale) if sig else "transparent"
            cells.append(
                f"<td class='{cls}' style='background:{bg}'><b>{pp:+.1f}%p</b>"
                f"<small>{note} · [{d['ci_low']*100:+.1f}, {d['ci_high']*100:+.1f}]</small></td>"
            )
            if sig > 0 and d["delta"] > best_val:
                best_arm, best_val = arm, d["delta"]
        best = "A · 원본 유지" if best_arm == "A" else f"{best_arm} · {ARM_INFO[best_arm][0]}"
        body.append(f"<tr><td class='l'><b>{ATTR_KO[a]}</b> <small style='color:var(--mute)'>{a}</small></td>"
                    f"{''.join(cells)}<td><b>{escape(best)}</b></td></tr>")
    legend = (
        "<div class='legend'><span><i class='sw' style='background:rgba(31,122,86,.5)'></i>A보다 좋아짐</span>"
        "<span><i class='sw' style='background:rgba(181,83,42,.5)'></i>A보다 나빠짐</span>"
        "<span><i class='sw' style='background:#fff;border:1px solid #ccc'></i>흐림 = 95% 신뢰구간이 0을 포함 (차이 단정 불가)</span></div>"
    )
    return f"<table>{head}{''.join(body)}</table>{legend}"


def _delta_bars(pairwise: dict[str, Any], arms_data: dict[str, Any], attr: str, scale: float) -> str:
    rows = []
    base = arms_data["A"][attr]["purity_at_k"]
    rows.append(
        f"<div class='bar-row'><span>A · 원본</span><div class='track'><span class='zero'></span></div>"
        f"<span>{base*100:.1f}%</span></div>"
    )
    for arm in ARMS[1:]:
        d = pairwise[f"A_vs_{arm}"][attr]
        sig = _sig(d)
        color = "var(--pos)" if d["delta"] > 0 else "var(--neg)"
        width = abs(d["delta"]) / scale * 50
        left = 50 if d["delta"] >= 0 else 50 - width
        lo = 50 + d["ci_low"] / scale * 50
        hi = 50 + d["ci_high"] / scale * 50
        val = arms_data[arm][attr]["purity_at_k"] * 100
        rows.append(
            f"<div class='bar-row{'' if sig else ' ns'}'><span>{arm} · {escape(ARM_INFO[arm][0])}</span>"
            f"<div class='track'><span class='zero'></span>"
            f"<span class='bar' style='left:{left:.1f}%;width:{width:.1f}%;background:{color}'></span>"
            f"<span class='ci' style='left:{lo:.1f}%;width:{max(hi-lo,0.5):.1f}%'></span></div>"
            f"<span>{val:.1f}% <small>({d['delta']*100:+.1f})</small></span></div>"
        )
    return "".join(rows)


def _class_table(arms_data: dict[str, Any], attr: str) -> str:
    counts = arms_data["A"][attr].get("class_counts") or {}
    per = {arm: _sil(arms_data[arm][attr]).get("per_class") or {} for arm in ARMS}
    classes = [c for c in sorted(counts, key=lambda c: -counts[c]) if c in per["A"]]
    deltas = [per[arm][c] - per["A"][c] for arm in ARMS[1:] for c in classes if c in per[arm]]
    scale = max((abs(x) for x in deltas), default=0.0)
    head = "<tr><th class='l'>라벨</th><th>n</th><th>A 점수</th>" + "".join(
        f"<th>{arm} 변화</th>" for arm in ARMS[1:]
    ) + "</tr>"
    body = []
    for c in classes:
        cells = []
        for arm in ARMS[1:]:
            if c not in per[arm]:
                cells.append("<td>—</td>")
                continue
            dv = per[arm][c] - per["A"][c]
            cells.append(f"<td style='background:{_tint(dv, scale)}'>{dv:+.3f}</td>")
        body.append(
            f"<tr><td class='l'>{escape(_ko(c))}</td><td>{counts[c]}</td>"
            f"<td>{per['A'][c]:+.3f}</td>{''.join(cells)}</tr>"
        )
    return f"<table class='mini'>{head}{''.join(body)}</table>"


def _attr_takeaway(pairwise: dict[str, Any], attr: str) -> str:
    ups = [arm for arm in ARMS[1:] if _sig(pairwise[f"A_vs_{arm}"][attr]) > 0]
    downs = [arm for arm in ARMS[1:] if _sig(pairwise[f"A_vs_{arm}"][attr]) < 0]
    parts = []
    if ups:
        parts.append("개선: " + ", ".join(f"{a}({ARM_INFO[a][0]})" for a in ups))
    if downs:
        parts.append("악화: " + ", ".join(f"{a}({ARM_INFO[a][0]})" for a in downs))
    return " · ".join(parts) or "전처리별 차이가 불확실"


def _attr_sections(pairwise: dict[str, Any], arms_data: dict[str, Any]) -> str:
    scale = max(
        max(abs(pairwise[f"A_vs_{arm}"][a]["ci_low"]), abs(pairwise[f"A_vs_{arm}"][a]["ci_high"]))
        for arm in ARMS[1:] for a in ATTRS
    ) * 1.05
    out = []
    for a in ATTRS:
        out.append(
            f"<div class='attr'><div class='attr-head'><h3>{ATTR_KO[a]} <small style='color:var(--mute)'>{a}</small></h3>"
            f"<span class='take'>{escape(_attr_takeaway(pairwise, a))}</span></div>"
            "<div class='grid2'><div><p class='hint'>이웃 10개 중 같은 라벨 비율 (막대 = A 대비 변화, 가는 선 = 95% 신뢰구간)</p>"
            f"{_delta_bars(pairwise, arms_data, a, scale)}</div>"
            "<div><p class='hint'>라벨별 군집 점수(실루엣). A 점수 대비 변화가 초록이면 그 라벨이 더 또렷하게 모임</p>"
            f"{_class_table(arms_data, a)}</div></div></div>"
        )
    return "".join(out)


def _quota(arms_data: dict[str, Any]) -> str:
    out = []
    for a in ATTRS:
        counts = arms_data["A"][a].get("class_counts") or {}
        top = max(max(counts.values(), default=QUOTA), QUOTA)
        rows = []
        for c in sorted(counts, key=lambda c: -counts[c]):
            n = counts[c]
            rows.append(
                f"<div class='qrow'><span>{escape(_ko(c))}</span><div class='qtrack'>"
                f"<span class='qfill{' low' if n < QUOTA else ''}' style='width:{n/top*100:.1f}%'></span>"
                f"<span class='qline' style='left:{QUOTA/top*100:.1f}%'></span></div><span>{n}</span></div>"
            )
        out.append(f"<div class='q'><h3>{ATTR_KO[a]}</h3>{''.join(rows)}</div>")
    return f"<div class='quota'>{''.join(out)}</div>"


def _kappa(kappa: dict[str, Any]) -> str:
    km = kappa["metrics"]
    cells = "".join(
        f"<td><b>{km[a]['kappa']:.2f}</b><br><small style='color:var(--mute)'>일치 {km[a]['agree']*100:.0f}%</small></td>"
        for a in ATTRS
    )
    head = "".join(f"<th>{ATTR_KO[a]}</th>" for a in ATTRS)
    return f"<table class='mini'><tr>{head}</tr><tr>{cells}</tr></table>"


GLOSSARY = """
<dl>
<dt>이웃 순도 (purity@10) — 이 리포트의 주 지표</dt>
<dd>각 상품에서 임베딩이 가장 가까운 10개 상품을 보고, 그중 같은 라벨(예: 같은 '스트라이프') 비율을 잽니다. 전체 상품 평균입니다. 높을수록 "비슷한 것끼리 모인다"는 뜻입니다.</dd>
<dt>A 대비 변화 (%p)와 신뢰구간</dt>
<dd>같은 상품 집합에서 B/C/D의 순도에서 A의 순도를 뺀 값입니다. 상품을 무작위로 다시 뽑는 부트스트랩을 2,000번 반복해 95% 신뢰구간을 구했습니다. 구간 전체가 0보다 크면 '유의하게 개선', 0보다 작으면 '유의하게 악화', 0을 걸치면 '차이 불확실'입니다.</dd>
<dt>실루엣 (Silhouette, Rousseeuw 1987)</dt>
<dd>상품 하나마다 a = 같은 라벨 상품들과의 평균 거리, b = 가장 가까운 다른 라벨 상품들과의 평균 거리를 구해 s = (b − a) / max(a, b)를 계산합니다(−1~+1). 한 라벨의 s를 평균하면 그 라벨이 얼마나 또렷하게 뭉쳐 있는지 숫자 하나로 나옵니다. 0 근처는 경계가 흐린 상태, 음수는 다른 라벨 쪽에 더 가깝다는 뜻입니다. 거리는 코사인 거리입니다.</dd>
<dt>R-precision, lift, separation gap (원자료에 보관)</dt>
<dd>R-precision은 같은 라벨 개수 R만큼 이웃을 봤을 때의 적중률, lift는 순도를 우연 수준(라벨 비율)으로 나눈 배율, gap은 같은 라벨끼리 평균 유사도에서 다른 라벨과의 평균 유사도를 뺀 값입니다. 결론은 순도와 같은 방향이라 본문에서는 생략했습니다. 값은 <code>metrics/cluster_quality.json</code>에 있습니다.</dd>
<dt>κ (Cohen's kappa)</dt>
<dd>에이전트 라벨과 사람 라벨이 우연을 넘어 얼마나 일치하는지(0~1). 0.8 이상이면 매우 높은 일치로 봅니다.</dd>
</dl>
"""


def _procedure() -> str:
    base = "pipeline"
    rows = [
        ("시드 표본 450장", f"{base}/cohorts/tops-450-20260918"),
        ("SAM3 마스크 (시드)", f"{base}/sam3/sam3-mps-450, sam3-mps-450-expanded, sam3-mps-450-fallback-cache"),
        ("A/B/C/D 임베딩 (시드)", f"{base}/preprocess/preprocess-abcd-sam3-450"),
        ("균형 보충 211장", f"{base}/cohorts/cohort-supp-211-20261002 → sam3/sam3-mps-supp211 → preprocess/preprocess-abcd-sam3-supp211"),
        ("확장 293장 (클래스당 100 목표)", f"{base}/cohorts/cohort-expand-293-20261002 → sam3/sam3-mps-expand293(-fallback, -recovered) → preprocess/preprocess-abcd-sam3-expand293-recovered"),
        ("부족분 보충 49·36·28장", f"{base}/cohorts/cohort-topup-* → sam3/sam3-mps-topup* → preprocess/preprocess-abcd-sam3-topup*"),
        ("라벨", "labels.jsonl, label_chunks*/, label_schema.json"),
        ("검수 50장 κ", "label_review.html, human_labels_review50.jsonl, kappa_review50.json"),
        ("병합 임베딩", "embeddings/sam3_abcd/ (출처: summary.json)"),
        ("지표 원자료", "metrics/cluster_quality.json, metrics/cluster_quality_summary.md"),
        ("폐기·스모크 산출물", "../_archive/"),
    ]
    body = "".join(f"<tr><td class='l'>{escape(s)}</td><td class='l'><code>{escape(p)}</code></td></tr>" for s, p in rows)
    return (
        "<details><summary>절차와 산출물 위치</summary><p class='hint'>경로는 이 리포트 폴더 기준입니다.</p>"
        f"<table class='mini'><tr><th class='l'>단계</th><th class='l'>위치</th></tr>{body}</table></details>"
    )


def build(root: Path, bank_name: str) -> Path:
    metrics = json.loads((root / "metrics" / "cluster_quality.json").read_text(encoding="utf-8"))
    kappa = json.loads((root / "kappa_review50.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "embeddings" / bank_name / "summary.json").read_text(encoding="utf-8"))
    bank = metrics["banks"][bank_name]
    arms_data, pairwise = bank["arms"], bank["pairwise"]

    shortfalls = [
        f"{ATTR_KO[a]} · {_ko(c)} {n}장"
        for a in ATTRS
        for c, n in (arms_data["A"][a].get("class_counts") or {}).items()
        if n < QUOTA
    ]
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>전처리별 속성 군집도 리포트</title><style>{CSS}</style></head><body><main>
<h1>전처리가 속성 군집을 좋게 만드는가</h1>
<p class="lead">SAM3 마스크 기반 전처리 4종(A~D)으로 만든 FashionSigLIP 임베딩에서, 같은 속성 라벨을 가진 상의끼리 얼마나 잘 모이는지 비교했습니다.</p>
<p class="meta">상품 {summary['coverage']['A']:,}장 · 속성 5종 · 생성 {escape(str(metrics.get('created_at', ''))[:10])}</p>

<div class="tldr"><b>결론</b><ul>
<li><b>B 크롭</b>과 <b>C 소프트 마스크</b>는 원본보다 같은 라벨끼리 더 잘 모입니다. 나빠지는 속성이 없습니다.</li>
<li><b>색상</b>은 C가 가장 크게 개선합니다. <b>무늬·넥라인·소매</b>는 B와 C가 비슷합니다.</li>
<li><b>D 하드 마스크</b>는 색상·무늬는 좋아지지만 <b>소매·넥라인·핏이 나빠집니다</b>. 검정 배경이 옷의 형태 단서를 지우는 것으로 보입니다.</li>
<li><b>핏</b>은 어떤 전처리로도 확실히 좋아지지 않습니다.</li>
</ul></div>
{_cards(pairwise)}

<h2>전처리 예시</h2>
<p class="hint">같은 상품을 A~D로 변환한 모습입니다.</p>
{_examples(root)}

<h2>한눈에 비교: 원본(A) 대비 개선폭</h2>
<p class="hint">값 = 각 상품의 가까운 이웃 10개 중 같은 라벨 비율이 A보다 몇 %p 바뀌었는지. 색이 진할수록 변화가 큽니다.</p>
{_heatmap(pairwise)}

<h2>속성별 상세</h2>
<p class="hint">왼쪽은 전처리별 이웃 순도, 오른쪽은 라벨 하나하나의 군집 점수 변화입니다. 예를 들어 '그래픽/로고' 라벨만 따로 비교하려면 무늬 표의 해당 줄을 보면 됩니다.</p>
{_attr_sections(pairwise, arms_data)}

<h2>데이터 신뢰도</h2>
<p class="hint">라벨별 표본 수 (세로선 = 목표 {QUOTA}장). 주황 막대는 목표 미달입니다: {escape(', '.join(shortfalls) or '없음')}. 터틀넥은 원천 풀에 상품이 부족하고, 기타 무늬는 이미지 확인 시 그래픽·무지로 재분류되는 경우가 많았습니다.</p>
{_quota(arms_data)}
<p class="hint" style="margin-top:18px">사람 검수 50장과 에이전트 라벨의 일치도 (κ)</p>
{_kappa(kappa)}

<h2>참고</h2>
<details><summary>지표 설명</summary>{GLOSSARY}</details>
{_procedure()}
</main></body></html>"""
    out = root / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    print(build(args.cohort_dir, args.bank))


if __name__ == "__main__":
    main()
