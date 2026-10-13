"""Render soft-mask pad/blur sweep report (sam3_soft_sweep bank)."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

ARMS = ("A", "B", "C_0", "C_1", "C_2", "C_3", "C_4", "C_blur40")
COMPARE_ARMS = ("B", "C_0", "C_1", "C_2", "C_3", "C_4", "C_blur40")
ARM_INFO = {
    "A": ("원본", "이미지 전체 (기준)"),
    "B": ("크롭", "pad 0.12 · 마스크 없음"),
    "C_0": ("기존 soft", "pad 0.12 · dilate 7 · blur σ=21"),
    "C_1": ("양쪽 패딩 없음", "pad 0 · dilate 0 · blur σ=21"),
    "C_2": ("약한 블러", "pad 0.12 · dilate 7 · blur σ=8"),
    "C_3": ("블러패딩 없음", "pad 0.12 · dilate 0 · blur σ=21"),
    "C_4": ("블러패딩 없음+약한 블러", "pad 0.12 · dilate 0 · blur σ=8"),
    "C_blur40": ("강한 블러(보관)", "pad 0.12 · dilate 7 · blur σ=40 · 첫 스윕 C_3"),
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
PREVIEW_A = "pipeline/preprocess/preprocess-abcd-sam3-450/previews/A"
PREVIEW_B = "pipeline/preprocess/preprocess-abcd-sam3-450/previews/B"
PREVIEW_SOFT = "previews/soft_sweep"
QUOTA = 100

CSS = """
:root{--fg:#1d2a26;--mute:#5f6d68;--line:#e2e8e4;--bg:#f6f8f6;--pos:#1f7a56;--neg:#b5532a;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",system-ui,sans-serif}
main{max-width:1120px;margin:0 auto;padding:40px 28px 80px}
h1{font-size:28px;letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:20px;margin:48px 0 6px;letter-spacing:-.01em}
h3{font-size:16px;margin:0 0 10px}
p.lead{color:var(--mute);margin:0 0 4px}
p.hint{color:var(--mute);font-size:13px;margin:4px 0 14px}
.meta{color:var(--mute);font-size:13px}
.tldr{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--pos);border-radius:10px;padding:18px 22px;margin:22px 0 0}
.tldr ul{margin:6px 0 0;padding-left:20px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .arm{font-size:12px;color:var(--mute)}
.card .name{font-size:17px;font-weight:700;margin:2px 0 6px}
.card .desc{font-size:13px;color:var(--mute);min-height:36px}
.badge{display:inline-block;font-size:12px;font-weight:700;padding:2px 9px;border-radius:999px;margin-top:8px}
.b-good{background:#e1f3ea;color:var(--pos)}.b-warn{background:#fbe9df;color:var(--neg)}.b-base{background:#eceff1;color:#4a5560}.b-mid{background:#f3f1e1;color:#7a6a1f}
.score{font-size:13px;margin-top:6px}
.ex{display:grid;grid-template-columns:110px repeat(8,1fr);gap:8px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto}
.ex .h{font-size:11px;color:var(--mute);text-align:center;font-weight:600}
.ex .rl{font-size:13px;font-weight:600}
.ex img{width:100%;aspect-ratio:1;object-fit:contain;background:#eef1ef;border-radius:6px}
table{border-collapse:separate;border-spacing:0;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:10px 10px;text-align:center;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{background:#eef2ef;font-size:12px;font-weight:600}
td.l,th.l{text-align:left}
tr:last-child td{border-bottom:none}
.cell b{display:block;font-size:15px}
.cell small{display:block;font-size:10px;color:var(--mute)}
.ns{opacity:.55}
.legend{display:flex;gap:16px;font-size:12px;color:var(--mute);margin:8px 0 0;flex-wrap:wrap}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px}
.attr{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-top:14px}
.attr-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.attr-head .take{font-size:13px;color:var(--mute)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:10px}
.bar-row{display:grid;grid-template-columns:108px 1fr 72px;gap:10px;align-items:center;font-size:13px;margin:7px 0}
.track{position:relative;height:18px;background:#f1f4f2;border-radius:4px}
.track .zero{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:#9aa6a1}
.track .bar{position:absolute;top:3px;height:12px;border-radius:3px}
.track .ci{position:absolute;top:8px;height:2px;background:#3a4743;opacity:.55}
.mini th,.mini td{padding:6px 8px;font-size:12px}
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
@media (max-width:960px){.cards,.quota{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cohort-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/garment_segmentation/attribute_cluster_eval-20261002",
    )
    p.add_argument("--bank", default="sam3_soft_sweep")
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
    wins = [a for a in ATTRS if a in pair and _sig(pair[a]) > 0]
    losses = [a for a in ATTRS if a in pair and _sig(pair[a]) < 0]
    if losses:
        return "주의", "b-warn", wins, losses
    if len(wins) >= 3:
        return "추천", "b-good", wins, losses
    return "보통", "b-mid", wins, losses


def _auto_tldr(pairwise: dict[str, Any]) -> str:
    lines: list[str] = []
    # Best arm per attr among COMPARE_ARMS
    best_by_attr: dict[str, str] = {}
    for a in ATTRS:
        best_arm, best_val = "A", 0.0
        for arm in COMPARE_ARMS:
            d = pairwise.get(f"A_vs_{arm}", {}).get(a)
            if not d:
                continue
            if _sig(d) > 0 and d["delta"] > best_val:
                best_arm, best_val = arm, d["delta"]
        best_by_attr[a] = best_arm

    c0_vs = []
    for arm in ("C_1", "C_2", "C_3", "C_4", "C_blur40"):
        better, worse = 0, 0
        for a in ATTRS:
            d_new = pairwise.get(f"A_vs_{arm}", {}).get(a)
            d_c0 = pairwise.get("A_vs_C_0", {}).get(a)
            if not d_new or not d_c0:
                continue
            if d_new["delta"] > d_c0["delta"] + 1e-6:
                better += 1
            elif d_new["delta"] < d_c0["delta"] - 1e-6:
                worse += 1
        c0_vs.append((arm, better, worse))

    wins_c0 = sum(1 for a in ATTRS if a in pairwise.get("A_vs_C_0", {}) and _sig(pairwise["A_vs_C_0"][a]) > 0)
    lines.append(
        f"<li><b>C_0 (기존 soft)</b>는 A 대비 속성 {wins_c0}/5에서 유의하게 개선됩니다. "
        f"B와 함께 기준 비교군입니다.</li>"
    )
    for arm, better, worse in c0_vs:
        name = ARM_INFO[arm][0]
        if better > worse:
            lines.append(
                f"<li><b>{arm} ({name})</b>는 C_0보다 A 대비 이득이 큰 속성이 {better}개, "
                f"작은 속성이 {worse}개입니다.</li>"
            )
        elif worse > better:
            lines.append(
                f"<li><b>{arm} ({name})</b>는 C_0보다 A 대비 이득이 작은 속성이 {worse}개로, "
                f"기존 soft를 넘어서지 못합니다.</li>"
            )
        else:
            lines.append(f"<li><b>{arm} ({name})</b>는 C_0과 전반적으로 비슷한 수준입니다.</li>")

    champs = {}
    for a, arm in best_by_attr.items():
        champs.setdefault(arm, []).append(ATTR_KO[a])
    champ_txt = ", ".join(f"{ARM_INFO[arm][0]}({arm}): {'/'.join(attrs)}" for arm, attrs in champs.items() if arm != "A")
    if champ_txt:
        lines.append(f"<li>속성별 A 대비 최고: {escape(champ_txt)}.</li>")
    return "".join(lines)


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


def _preview_path(root: Path, arm: str, item_id: str) -> Path | None:
    if arm == "A":
        p = root / PREVIEW_A / f"{item_id}.jpg"
    elif arm == "B":
        p = root / PREVIEW_B / f"{item_id}.jpg"
    else:
        p = root / PREVIEW_SOFT / arm / f"{item_id}.jpg"
    return p if p.is_file() else None


def _examples(root: Path) -> str:
    rows = ["<div class='h'></div>"] + [
        f"<div class='h'>{a}<br>{escape(ARM_INFO[a][0])}</div>" for a in ARMS
    ]
    for item_id, label in EXAMPLES:
        paths = [_preview_path(root, a, item_id) for a in ARMS]
        if not all(paths):
            continue
        rows.append(f"<div class='rl'>{escape(label)}</div>")
        for arm, path in zip(ARMS, paths):
            assert path is not None
            rel = path.relative_to(root).as_posix()
            rows.append(f"<img src='{rel}' alt='{arm}'>")
    return f"<div class='ex'>{''.join(rows)}</div>"


def _heatmap(pairwise: dict[str, Any]) -> str:
    scale = max(
        abs(pairwise[f"A_vs_{arm}"][a]["delta"])
        for arm in COMPARE_ARMS
        for a in ATTRS
        if a in pairwise.get(f"A_vs_{arm}", {})
    )
    head = "<tr><th class='l'>속성</th>" + "".join(
        f"<th>{arm}<br>{escape(ARM_INFO[arm][0])}</th>" for arm in COMPARE_ARMS
    ) + "<th>최고</th></tr>"
    body = []
    for a in ATTRS:
        cells = []
        best_arm, best_val = "A", 0.0
        for arm in COMPARE_ARMS:
            d = pairwise[f"A_vs_{arm}"][a]
            sig = _sig(d)
            pp = d["delta"] * 100
            note = "유의" if sig else "불확실"
            cls = "cell" if sig else "cell ns"
            bg = _tint(d["delta"], scale) if sig else "transparent"
            cells.append(
                f"<td class='{cls}' style='background:{bg}'><b>{pp:+.1f}%p</b>"
                f"<small>{note} · [{d['ci_low']*100:+.1f}, {d['ci_high']*100:+.1f}]</small></td>"
            )
            if sig > 0 and d["delta"] > best_val:
                best_arm, best_val = arm, d["delta"]
        best = "A · 원본" if best_arm == "A" else f"{best_arm} · {ARM_INFO[best_arm][0]}"
        body.append(
            f"<tr><td class='l'><b>{ATTR_KO[a]}</b></td>{''.join(cells)}"
            f"<td><b>{escape(best)}</b></td></tr>"
        )
    legend = (
        "<div class='legend'><span><i class='sw' style='background:rgba(31,122,86,.5)'></i>A보다 좋아짐</span>"
        "<span><i class='sw' style='background:rgba(181,83,42,.5)'></i>A보다 나빠짐</span>"
        "<span><i class='sw' style='background:#fff;border:1px solid #ccc'></i>흐림 = 차이 불확실</span></div>"
    )
    return f"<table>{head}{''.join(body)}</table>{legend}"


def _delta_bars(pairwise: dict[str, Any], arms_data: dict[str, Any], attr: str, scale: float) -> str:
    rows = []
    base = arms_data["A"][attr]["purity_at_k"]
    rows.append(
        f"<div class='bar-row'><span>A · 원본</span><div class='track'><span class='zero'></span></div>"
        f"<span>{base*100:.1f}%</span></div>"
    )
    for arm in COMPARE_ARMS:
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
    deltas = [per[arm][c] - per["A"][c] for arm in COMPARE_ARMS for c in classes if c in per[arm]]
    scale = max((abs(x) for x in deltas), default=0.0)
    head = "<tr><th class='l'>라벨</th><th>n</th><th>A</th>" + "".join(
        f"<th>{arm}</th>" for arm in COMPARE_ARMS
    ) + "</tr>"
    body = []
    for c in classes:
        cells = []
        for arm in COMPARE_ARMS:
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
    ups = [arm for arm in COMPARE_ARMS if _sig(pairwise[f"A_vs_{arm}"][attr]) > 0]
    downs = [arm for arm in COMPARE_ARMS if _sig(pairwise[f"A_vs_{arm}"][attr]) < 0]
    parts = []
    if ups:
        parts.append("개선: " + ", ".join(f"{a}({ARM_INFO[a][0]})" for a in ups))
    if downs:
        parts.append("악화: " + ", ".join(f"{a}({ARM_INFO[a][0]})" for a in downs))
    return " · ".join(parts) or "전처리별 차이가 불확실"


def _attr_sections(pairwise: dict[str, Any], arms_data: dict[str, Any]) -> str:
    scale = max(
        max(abs(pairwise[f"A_vs_{arm}"][a]["ci_low"]), abs(pairwise[f"A_vs_{arm}"][a]["ci_high"]))
        for arm in COMPARE_ARMS
        for a in ATTRS
    ) * 1.05
    out = []
    for a in ATTRS:
        out.append(
            f"<div class='attr'><div class='attr-head'><h3>{ATTR_KO[a]} <small style='color:var(--mute)'>{a}</small></h3>"
            f"<span class='take'>{escape(_attr_takeaway(pairwise, a))}</span></div>"
            "<div class='grid2'><div><p class='hint'>이웃 순도 (막대 = A 대비, 선 = 95% CI)</p>"
            f"{_delta_bars(pairwise, arms_data, a, scale)}</div>"
            "<div><p class='hint'>라벨별 실루엣 변화 (A 대비)</p>"
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
<dt>이웃 순도 (purity@10)</dt>
<dd>가까운 이웃 10개 중 같은 속성 라벨 비율. 높을수록 같은 라벨끼리 모입니다.</dd>
<dt>A 대비 Δ와 신뢰구간</dt>
<dd>부트스트랩 2,000회 95% CI. 구간이 0을 안 지나면 유의한 개선/악화입니다.</dd>
<dt>스윕 설계</dt>
<dd>C_0=기존 soft. C_1=박스·블러패딩 제거. C_2=약한 블러. C_3=크롭 유지·dilate 0. C_4=C_3+약한 블러. <b>C_blur40</b>=첫 스윕의 강한 블러(옛 C_3)를 별도 stem으로 보관. stem을 재정의할 때 덮어쓰지 않도록 archive에 백업합니다.</dd>
<dt>실루엣</dt>
<dd>라벨별 뭉침 점수(−1~+1). 표의 숫자는 A 대비 변화입니다.</dd>
</dl>
"""


def _procedure() -> str:
    rows = [
        ("기존 ABCD bank", "embeddings/sam3_abcd/"),
        ("스윕 bank", "embeddings/sam3_soft_sweep/ (C_1~C_4, C_blur40)"),
        ("임베딩 백업", "embeddings/sam3_soft_sweep/archive/"),
        ("지표 아카이브", "metrics/archive/"),
        ("설정", "experiments/garment_segmentation/config.preprocess.soft_sweep.example.yaml"),
        ("예제 미리보기", "previews/soft_sweep/"),
        ("지표", "metrics/cluster_quality.json → banks.sam3_soft_sweep"),
        ("이 리포트", "report_soft_sweep.html"),
        ("1차 ABCD 리포트", "report.html"),
    ]
    body = "".join(
        f"<tr><td class='l'>{escape(s)}</td><td class='l'><code>{escape(p)}</code></td></tr>"
        for s, p in rows
    )
    return (
        "<details><summary>절차와 산출물 위치</summary>"
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
    cov = summary.get("coverage") or {}
    n_cov = cov.get("C_1") or cov.get("A") or 0
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Soft-mask 패딩·블러 스윕 리포트</title><style>{CSS}</style></head><body><main>
<h1>패딩과 블러를 바꾸면 soft mask가 더 좋아질까</h1>
<p class="lead">기존 B·C_0에 더해 C_1~C_4 soft-mask 변형을 비교하고, 첫 스윕의 강한 블러(σ=40)는 <b>C_blur40</b>으로 별도 보관·포함합니다.</p>
<p class="meta">상품 {n_cov:,}장 · bank {escape(bank_name)} · 생성 {escape(str(metrics.get('created_at', ''))[:10])}</p>

<div class="tldr"><b>결론</b><ul>
{_auto_tldr(pairwise)}
</ul></div>
{_cards(pairwise)}

<h2>전처리 예시</h2>
<p class="hint">C_3·C_4는 dilate 0, C_blur40은 첫 스윕 강한 블러(σ=40) 보관본입니다.</p>
{_examples(root)}

<h2>한눈에 비교: 원본(A) 대비 개선폭</h2>
<p class="hint">이웃 10개 중 같은 라벨 비율이 A보다 몇 %p 바뀌었는지입니다.</p>
{_heatmap(pairwise)}

<h2>속성별 상세</h2>
{_attr_sections(pairwise, arms_data)}

<h2>데이터 신뢰도</h2>
<p class="hint">라벨별 표본 수 (세로선 = 목표 {QUOTA}장). 미달: {escape(', '.join(shortfalls) or '없음')}.</p>
{_quota(arms_data)}
<p class="hint" style="margin-top:18px">사람 검수 50장 κ</p>
{_kappa(kappa)}

<h2>참고</h2>
<details><summary>지표·스윕 설명</summary>{GLOSSARY}</details>
{_procedure()}
</main></body></html>"""
    out = root / "report_soft_sweep.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    print(build(args.cohort_dir, args.bank))


if __name__ == "__main__":
    main()
