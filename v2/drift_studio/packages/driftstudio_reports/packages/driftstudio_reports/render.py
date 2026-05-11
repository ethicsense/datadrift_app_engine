from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

from driftstudio_spec import Plan, ReportFormat, StepType


_TEMPLATE_DIR = Path(__file__).parent / "templates"

_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)

# 웹 `EmbeddingProjectionCard`와 동일 팔레트
_CLUSTER_COLORS = [
    "#7c3aed",
    "#22c55e",
    "#06b6d4",
    "#f59e0b",
    "#ef4444",
    "#3b82f6",
    "#a855f7",
    "#84cc16",
    "#14b8a6",
    "#e11d48",
    "#6366f1",
    "#f97316",
]


def _json_pretty(payload: Any) -> str:
    if payload is None:
        return "{}"
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _report_package_version() -> str:
    try:
        from importlib.metadata import version

        return version("driftstudio_reports")
    except Exception:
        return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _normalize_cluster_id(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _cluster_sort_key(cid: Any) -> tuple:
    if cid == -1:
        return (2, 0, "")
    if isinstance(cid, int):
        return (0, cid, "")
    return (1, 0, str(cid))


def _cluster_label(cid: Any) -> str:
    if cid == -1:
        return "noise"
    if isinstance(cid, int):
        return f"cluster_{cid}"
    return str(cid)


def _to_float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for item in values:
        if isinstance(item, (int, float)):
            out.append(float(item))
    return out


def _hist_svg(
    title: str,
    hist: dict[str, Any],
    *,
    x_label: str,
    caption: str,
    width: int = 760,
    height: int = 240,
) -> str | None:
    bins = _to_float_list(hist.get("bins"))
    counts = _to_float_list(hist.get("counts"))
    if len(bins) < 2 or not counts:
        return None

    n = min(len(counts), len(bins) - 1)
    counts = counts[:n]
    bins = bins[: n + 1]
    max_count = max(counts) if counts else 0
    if max_count <= 0:
        return None

    margin_left, margin_right, margin_top, margin_bottom = 52, 14, 28, 52
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom
    bar_gap = 2
    bar_w = max(2.0, (inner_w - bar_gap * (n - 1)) / max(1, n))

    bars: list[str] = []
    for i, c in enumerate(counts):
        h = (c / max_count) * inner_h
        x = margin_left + i * (bar_w + bar_gap)
        y = margin_top + (inner_h - h)
        bars.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' height='{h:.2f}' fill='#2563eb' opacity='0.85' />"
        )

    x_label_start = bins[0]
    x_label_end = bins[-1]
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"<text x='{margin_left}' y='16' font-size='12' font-weight='600' fill='#111827'>{title}</text>"
        f"<text x='{margin_left}' y='{height - 10}' font-size='10' fill='#6b7280'>{caption}</text>"
        f"<text x='8' y='{margin_top + inner_h / 2}' font-size='10' fill='#6b7280' transform='rotate(-90 8 {margin_top + inner_h / 2})'>count</text>"
        f"<text x='{margin_left + inner_w / 2 - 40}' y='{margin_top + inner_h + 38}' font-size='10' fill='#6b7280'>{x_label}</text>"
        f"<line x1='{margin_left}' y1='{margin_top + inner_h}' x2='{width - margin_right}' y2='{margin_top + inner_h}' stroke='#9ca3af'/>"
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + inner_h}' stroke='#9ca3af'/>"
        + "".join(bars)
        + f"<text x='{margin_left}' y='{margin_top + inner_h + 18}' font-size='10' fill='#6b7280'>{x_label_start:.3g}</text>"
        + f"<text x='{width - margin_right - 40}' y='{margin_top + inner_h + 18}' font-size='10' fill='#6b7280'>{x_label_end:.3g}</text>"
        + f"<text x='{width - margin_right - 36}' y='{margin_top + 12}' font-size='10' fill='#6b7280'>max {int(max_count)}</text>"
        + f"<title>{title}</title>"
        + "</svg>"
    )


def _dict_bar_svg(title: str, values: dict[str, Any], *, width: int = 760, height: int = 220) -> str | None:
    numeric_items = [(str(k), float(v)) for k, v in values.items() if isinstance(v, (int, float))]
    if not numeric_items:
        return None
    numeric_items.sort(key=lambda x: x[1], reverse=True)
    numeric_items = numeric_items[:20]
    max_value = max(v for _, v in numeric_items)
    if max_value <= 0:
        return None

    margin_left, margin_right, margin_top, margin_bottom = 130, 12, 28, 24
    inner_w = width - margin_left - margin_right
    row_h = 20
    needed_h = margin_top + margin_bottom + row_h * len(numeric_items)
    height = max(height, needed_h)

    bars: list[str] = []
    labels: list[str] = []
    for i, (name, val) in enumerate(numeric_items):
        y = margin_top + i * row_h
        w = (val / max_value) * inner_w
        bars.append(
            f"<rect x='{margin_left}' y='{y + 3}' width='{w:.2f}' height='12' fill='#059669' opacity='0.85' />"
            f"<text x='{margin_left + w + 6:.2f}' y='{y + 13}' font-size='11' fill='#374151'>{val:.3g}</text>"
        )
        labels.append(f"<text x='8' y='{y + 13}' font-size='11' fill='#374151'>{name[:28]}</text>")

    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"<text x='8' y='16' font-size='12' font-weight='600' fill='#111827'>{title}</text>"
        + "".join(labels)
        + "".join(bars)
        + f"<title>{title}</title>"
        + "</svg>"
    )


def _vertical_bar_svg(
    title: str,
    values: dict[str, Any],
    *,
    color: str = "#6366f1",
    width: int = 760,
    height: int = 280,
) -> str | None:
    items = [(str(k), float(v)) for k, v in values.items() if isinstance(v, (int, float))]
    if not items:
        return None
    items.sort(key=lambda x: x[0])
    if len(items) > 16:
        items = items[:16]
    max_val = max(v for _, v in items)
    if max_val <= 0:
        return None

    margin_left, margin_right, margin_top, margin_bottom = 44, 14, 24, 86
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom
    n = len(items)
    slot = inner_w / max(1, n)
    bar_w = max(8.0, slot * 0.62)
    tick_step = max(1, math.ceil(n / 6))

    parts: list[str] = [
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        f"<text x='{margin_left}' y='16' font-size='12' font-weight='600' fill='#111827'>{title}</text>",
    ]
    for g in range(4):
        y = margin_top + (inner_h * g / 3.0)
        parts.append(
            f"<line x1='{margin_left}' y1='{y:.2f}' x2='{margin_left + inner_w}' y2='{y:.2f}' stroke='#e5e7eb' stroke-dasharray='3 3'/>"
        )
    parts.extend(
        [
            f"<line x1='{margin_left}' y1='{margin_top + inner_h}' x2='{margin_left + inner_w}' y2='{margin_top + inner_h}' stroke='#9ca3af'/>",
            f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + inner_h}' stroke='#9ca3af'/>",
            f"<text x='8' y='{margin_top + 8}' font-size='10' fill='#6b7280'>{max_val:.3g}</text>",
        ]
    )
    for i, (name, val) in enumerate(items):
        h = (val / max_val) * inner_h
        x = margin_left + i * slot + (slot - bar_w) / 2.0
        y = margin_top + inner_h - h
        cx = x + bar_w / 2.0
        parts.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' height='{h:.2f}' fill='{color}' opacity='0.9'/>"
        )
        parts.append(
            f"<text x='{cx:.2f}' y='{y - 4:.2f}' font-size='9' text-anchor='middle' fill='#4b5563'>{val:.3g}</text>"
        )
        if i % tick_step == 0:
            label = name[:18]
            parts.append(
                f"<text x='{cx:.2f}' y='{margin_top + inner_h + 8:.2f}' font-size='9' fill='#6b7280' transform='rotate(-25 {cx:.2f} {margin_top + inner_h + 8:.2f})' text-anchor='end'>{label}</text>"
            )
    parts.append(f"<title>{title}</title></svg>")
    return "".join(parts)


def _hist_counts_svg(
    title: str,
    bins: list[float],
    counts: list[float],
    *,
    color: str,
    caption: str,
    width: int = 280,
    height: int = 172,
) -> str | None:
    if len(bins) < 2 or not counts:
        return None
    n = min(len(counts), len(bins) - 1)
    counts = counts[:n]
    bins = bins[: n + 1]
    max_count = max(counts) if counts else 0
    if max_count <= 0:
        return None
    margin_left, margin_right, margin_top, margin_bottom = 30, 8, 20, 28
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom
    bar_gap = 1
    bar_w = max(2.0, (inner_w - bar_gap * (n - 1)) / max(1, n))
    bars: list[str] = []
    for i, c in enumerate(counts):
        h = (c / max_count) * inner_h
        x = margin_left + i * (bar_w + bar_gap)
        y = margin_top + (inner_h - h)
        bars.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' height='{h:.2f}' fill='{color}' opacity='0.88' />")
    return (
        f"<svg width='100%' height='{height}' preserveAspectRatio='xMinYMin meet' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"<text x='{margin_left}' y='16' font-size='12' font-weight='600' fill='#111827'>{title}</text>"
        f"<line x1='{margin_left}' y1='{margin_top + inner_h}' x2='{width - margin_right}' y2='{margin_top + inner_h}' stroke='#9ca3af'/>"
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + inner_h}' stroke='#9ca3af'/>"
        + "".join(bars)
        + f"<text x='{margin_left}' y='{height - 20}' font-size='9' fill='#6b7280'>[{bins[0]:.3g}, {bins[-1]:.3g}]</text>"
        + f"<text x='{margin_left + 84}' y='{height - 20}' font-size='8.5' fill='#6b7280'>{caption}</text>"
        + f"<title>{title}</title></svg>"
    )


def _embedding_projection_split_svg(
    title: str,
    points: list[dict[str, Any]],
    *,
    caption: str,
    width: int = 760,
    height: int = 380,
) -> str | None:
    if not points:
        return None
    coords: list[tuple[float, float, str]] = []
    for p in points[:4000]:
        x = p.get("x")
        y = p.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        split = str(p.get("split") or "")
        color = "#7c3aed" if split == "base" else "#84cc16" if split == "target" else "#6366f1"
        coords.append((float(x), float(y), color))
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    margin_left, margin_right, margin_top, margin_bottom = 48, 14, 52, 42
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def px(x: float) -> float:
        return margin_left + ((x - x_min) / (x_max - x_min)) * inner_w

    def py(y: float) -> float:
        return margin_top + inner_h - ((y - y_min) / (y_max - y_min)) * inner_h

    dots = "".join(f"<circle cx='{px(x):.2f}' cy='{py(y):.2f}' r='2.0' fill='{c}' opacity='0.70'/>" for x, y, c in coords)
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
        f"<text x='{margin_left}' y='18' font-size='13' font-weight='600' fill='#111827'>{title}</text>"
        f"<text x='{margin_left}' y='34' font-size='10' fill='#6b7280'>{caption}</text>"
        f"<rect x='{margin_left}' y='40' width='10' height='10' fill='#7c3aed'/><text x='{margin_left + 14}' y='49' font-size='10' fill='#374151'>base</text>"
        f"<rect x='{margin_left + 64}' y='40' width='10' height='10' fill='#84cc16'/><text x='{margin_left + 78}' y='49' font-size='10' fill='#374151'>target</text>"
        f"<line x1='{margin_left}' y1='{margin_top + inner_h}' x2='{margin_left + inner_w}' y2='{margin_top + inner_h}' stroke='#9ca3af'/>"
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + inner_h}' stroke='#9ca3af'/>"
        + dots
        + f"<text x='{margin_left}' y='{height - 10}' font-size='10' fill='#6b7280'>x [{x_min:.2f}, {x_max:.2f}] · y [{y_min:.2f}, {y_max:.2f}]</text>"
        + f"<title>{title}</title></svg>"
    )


def _cluster_sizes_svg(title: str, clusters: list[dict[str, Any]], *, width: int = 760, height: int = 260) -> str | None:
    if not clusters:
        return None
    items = []
    for c in clusters:
        cid = _normalize_cluster_id(c.get("id"))
        if cid is None:
            continue
        sz = c.get("size")
        if not isinstance(sz, (int, float)):
            continue
        items.append((_cluster_label(cid), float(sz)))
    if not items:
        return None
    items.sort(key=lambda x: x[1], reverse=True)
    values = {k: v for k, v in items[:24]}
    return _dict_bar_svg(title, values, width=width, height=max(height, 28 + 24 + 20 * len(values)))


def _scatter_svg_clustered(
    title: str,
    points: list[dict[str, Any]],
    *,
    projection_method: str,
    cluster_method: str | None,
    n_clusters: Any,
    sampling: dict[str, Any] | None,
    width: int = 760,
    height: int = 400,
) -> str | None:
    if not points:
        return None

    reduced = points[:2000]
    has_cluster = any(p.get("cluster") is not None and p.get("cluster") is not False for p in reduced)

    margin_left, margin_right, margin_top, margin_bottom = 48, 14, 72, 44
    legend_w = 170
    plot_width = width - margin_left - margin_right - (legend_w if has_cluster else 0)
    inner_w = plot_width
    inner_h = height - margin_top - margin_bottom

    coords: list[tuple[float, float, str, Any]] = []
    if has_cluster:
        cluster_map: dict[Any, list[tuple[float, float]]] = {}
        for p in reduced:
            if not isinstance(p.get("x"), (int, float)) or not isinstance(p.get("y"), (int, float)):
                continue
            cid = _normalize_cluster_id(p.get("cluster"))
            if cid is None:
                continue
            cluster_map.setdefault(cid, []).append((float(p["x"]), float(p["y"])))

        if not cluster_map:
            has_cluster = False
        else:
            ordered_cids = sorted(cluster_map.keys(), key=_cluster_sort_key)
            cid_to_color = {cid: _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i, cid in enumerate(ordered_cids)}
            for cid in ordered_cids:
                col = cid_to_color[cid]
                for x, y in cluster_map[cid]:
                    coords.append((x, y, col, cid))

    if not has_cluster:
        for p in reduced:
            if not isinstance(p.get("x"), (int, float)) or not isinstance(p.get("y"), (int, float)):
                continue
            coords.append((float(p["x"]), float(p["y"]), _CLUSTER_COLORS[0], None))

    if not coords:
        return None

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def px(x: float) -> float:
        return margin_left + ((x - x_min) / (x_max - x_min)) * inner_w

    def py(y: float) -> float:
        return margin_top + inner_h - ((y - y_min) / (y_max - y_min)) * inner_h

    dots = "".join(
        f"<circle cx='{px(x):.2f}' cy='{py(y):.2f}' r='2.2' fill='{fill}' opacity='0.72' />" for x, y, fill, _ in coords
    )

    sample_note = ""
    if sampling and isinstance(sampling, dict):
        n = sampling.get("n")
        cap = sampling.get("cap")
        total = sampling.get("total")
        if n is not None and cap is not None and total is not None:
            sample_note = f"표시 샘플 {n} / 상한 {cap} (전체 {total})"

    subtitle = (
        f"투영: {projection_method}"
        + (f" · 클러스터: {cluster_method or '-'}" + (f" (k={n_clusters})" if n_clusters is not None else ""))
        + (f" · {sample_note}" if sample_note else "")
    )

    parts: list[str] = [
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        f"<text x='{margin_left}' y='20' font-size='13' font-weight='600' fill='#111827'>{title}</text>",
        f"<text x='{margin_left}' y='38' font-size='10' fill='#6b7280'>{subtitle}</text>",
        f"<text x='8' y='{margin_top + inner_h / 2}' font-size='10' fill='#6b7280' transform='rotate(-90 8 {margin_top + inner_h / 2})'>PC2</text>",
        f"<text x='{margin_left + inner_w / 2 - 24}' y='{height - 12}' font-size='10' fill='#6b7280'>PC1</text>",
        f"<line x1='{margin_left}' y1='{margin_top + inner_h}' x2='{margin_left + inner_w}' y2='{margin_top + inner_h}' stroke='#9ca3af'/>",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + inner_h}' stroke='#9ca3af'/>",
        dots,
        f"<text x='{margin_left}' y='{margin_top + inner_h + 22}' font-size='10' fill='#6b7280'>x [{x_min:.2f}, {x_max:.2f}]</text>",
        f"<text x='{margin_left + inner_w - 120}' y='{margin_top + inner_h + 22}' font-size='10' fill='#6b7280'>y [{y_min:.2f}, {y_max:.2f}]</text>",
    ]

    if has_cluster:
        lx = margin_left + inner_w + 12
        ly = margin_top
        parts.append(f"<text x='{lx}' y='{ly}' font-size='11' font-weight='600' fill='#374151'>범례</text>")
        ordered_cids = sorted({c[3] for c in coords if c[3] is not None}, key=_cluster_sort_key)
        cid_to_color = {}
        for i, cid in enumerate(ordered_cids):
            cid_to_color[cid] = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        row = 0
        for cid in ordered_cids:
            col = cid_to_color[cid]
            yy = ly + 16 + row * 18
            parts.append(
                f"<rect x='{lx}' y='{yy - 10}' width='12' height='12' fill='{col}' opacity='0.85' />"
                f"<text x='{lx + 18}' y='{yy}' font-size='10' fill='#374151'>{_cluster_label(cid)}</text>"
            )
            row += 1
            if row > 14:
                break

    parts.append(f"<title>{title}</title></svg>")
    return "".join(parts)


def _build_kpi_rows(summary: dict[str, Any], eda_base: dict[str, Any] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not isinstance(summary, dict):
        return rows

    priority_keys = [
        ("files_analyzed", "분석 파일 수"),
        ("num_files", "파일 수"),
        ("texts_analyzed", "텍스트 수"),
        ("series_analyzed", "시계열 수"),
        ("embeddings_extracted", "임베딩 추출 수"),
        ("n_clusters", "클러스터 수"),
        ("avg_width", "평균 너비(px)"),
        ("avg_height", "평균 높이(px)"),
        ("avg_size_mb", "평균 파일 크기(MB)"),
        ("modality", "모달리티"),
        ("status", "상태"),
    ]
    used: set[str] = set()
    for key, label in priority_keys:
        if key in summary and summary[key] is not None:
            rows.append({"label": label, "value": str(summary[key])})
            used.add(key)

    if isinstance(eda_base, dict):
        mid = eda_base.get("modality")
        if mid and "modality" not in used:
            rows.append({"label": "모달리티(원본)", "value": str(mid)})

    for k, v in summary.items():
        if k in used or v is None:
            continue
        if isinstance(v, (dict, list)):
            continue
        if len(rows) >= 14:
            break
        rows.append({"label": k, "value": str(v)})
    return rows


def _executive_bullets(
    modality: str | None,
    summary: dict[str, Any],
    eda_base: dict[str, Any] | None,
    name: str,
) -> list[str]:
    bullets: list[str] = []
    if modality:
        bullets.append(f"데이터 모달리티는 {modality} 입니다.")

    nf = summary.get("num_files") or summary.get("files_analyzed") or (eda_base or {}).get("files_analyzed")
    if nf is not None:
        bullets.append(f"분석 대상 파일(또는 레코드) 수는 {nf} 입니다.")

    if summary.get("avg_width") is not None and summary.get("avg_height") is not None:
        bullets.append(
            f"이미지 해상도 평균은 약 {float(summary['avg_width']):.0f}×{float(summary['avg_height']):.0f}px 입니다."
        )

    if summary.get("n_clusters") is not None:
        bullets.append(f"임베딩 기반 자동 클러스터 수는 {summary['n_clusters']} 입니다.")

    if summary.get("embeddings_extracted") is not None:
        bullets.append(f"임베딩이 추출된 샘플은 {summary['embeddings_extracted']} 건입니다.")

    bullets.append(f"데이터셋 경로(표시명): {name}")
    if not bullets:
        bullets.append("요약 지표가 제한적입니다. 원시 Summary는 부록을 참고하세요.")
    return bullets


def _quality_notes(summary: dict[str, Any], modality: str | None) -> list[str]:
    notes: list[str] = []
    if modality == "vision_image":
        w = summary.get("avg_width")
        h = summary.get("avg_height")
        if isinstance(w, (int, float)) and isinstance(h, (int, float)):
            ar = w / h if h else 0
            if ar > 1.4:
                notes.append("가로가 긴 이미지 비율이 높을 수 있습니다. 크롭/리사이즈 정책을 점검하세요.")
            elif ar < 0.75:
                notes.append("세로가 긴 이미지 비율이 높을 수 있습니다.")
        sz = summary.get("avg_size_mb")
        if isinstance(sz, (int, float)) and sz > 5:
            notes.append("평균 파일 크기가 큽니다. 학습/배포 파이프라인에서 I/O 병목 가능성이 있습니다.")
    if not notes:
        notes.append("자동 품질 코멘트: 분포 차트를 보고 이상치 구간(극단 bin)이 있는지 확인하세요.")
    return notes


def _label_from_path(p: str | None) -> str:
    if not p:
        return "—"
    try:
        return Path(p).name or str(p)
    except Exception:
        return str(p)


def _drift_status_for_ui(overall: Any) -> str:
    """
    웹 Drift 요약 카드(drift.status.v1)와 동일 구간.
    RuntimeRunner가 raw의 vision status(0.15/0.25)와 별도로 부여하는 표시용 상태.
    """
    try:
        o = float(overall)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if o >= 1.0:
        return "CRITICAL"
    if o >= 0.7:
        return "WARNING"
    return "NORMAL"


def _compact_drift_for_appendix(drift: dict[str, Any]) -> dict[str, Any]:
    """
    부록 JSON만 축약. 차트용 원본 drift는 그대로 두고, 복사본에서 대용량 필드 제거.
    - attribute_values_* : 파일(또는 레코드)마다 스칼라가 모여 지표당 수천~수만 개.
    - embedding_index_* : 경로·메타·인덱스 구조가 길어질 수 있음.
    """
    out = {
        k: v
        for k, v in drift.items()
        if k not in ("attribute_values_ref", "attribute_values_cur", "embedding_index_ref", "embedding_index_cur")
    }
    omitted: list[str] = []
    for k in ("attribute_values_ref", "attribute_values_cur", "embedding_index_ref", "embedding_index_cur"):
        if k in drift:
            omitted.append(k)
    if omitted:
        out["_omitted_large_fields"] = omitted
    return out


def _build_drift_kpi_rows(drift: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    st = _drift_status_for_ui(drift.get("overall_score"))
    if drift.get("overall_score") is not None:
        rows.append(
            {
                "label": "드리프트 · 종합 점수",
                "value": f"{float(drift['overall_score']):.4f} ({st})",
            }
        )
    else:
        rows.append({"label": "드리프트 · 상태", "value": str(st)})

    if drift.get("embedding_drift") is not None:
        rows.append({"label": "드리프트 · 임베딩", "value": f"{float(drift['embedding_drift']):.4f}"})
    if drift.get("attribute_drift_overall") is not None:
        rows.append(
            {"label": "드리프트 · 속성 평균(PSI 등)", "value": f"{float(drift['attribute_drift_overall']):.4f}"}
        )
    if drift.get("size_drift") is not None:
        rows.append({"label": "드리프트 · 크기 분포", "value": f"{float(drift['size_drift']):.4f}"})

    for key, label in [
        ("files_added", "파일 추가(타깃 기준)"),
        ("files_removed", "파일 삭제(타깃 기준)"),
        ("files_common", "공통 파일 수"),
    ]:
        if drift.get(key) is not None:
            rows.append({"label": label, "value": str(drift[key])})

    if drift.get("modality"):
        rows.append({"label": "드리프트 · 모달리티", "value": str(drift["modality"])})

    det = drift.get("embedding_drift_detailed")
    if isinstance(det, dict):
        for key, label in [
            ("mmd", "MMD (single)"),
            ("mmd_multiscale", "MMD (multi-scale)"),
            ("mean_shift", "Mean shift"),
            ("wasserstein", "Wasserstein"),
            ("psi", "PSI (avg)"),
        ]:
            if det.get(key) is not None:
                try:
                    rows.append({"label": f"임베딩 · {label}", "value": f"{float(det[key]):.4f}"})
                except (TypeError, ValueError):
                    rows.append({"label": f"임베딩 · {label}", "value": str(det[key])})
    return rows


def _drift_executive_bullets(
    drift: dict[str, Any],
    *,
    base_label: str,
    target_label: str,
) -> list[str]:
    bullets: list[str] = []
    bullets.append(f"비교 대상: Base «{base_label}» vs Target «{target_label}».")

    st = _drift_status_for_ui(drift.get("overall_score"))
    if drift.get("overall_score") is not None:
        bullets.append(
            f"종합 드리프트 점수는 {float(drift['overall_score']):.4f}이며, 상태는 {st} 입니다. "
            f"(웹 UI Drift 요약과 동일: 0.7 미만 정상, 0.7 이상 1.0 미만 주의, 1.0 이상 위험.)"
        )
    else:
        bullets.append(f"드리프트 상태: {st}.")

    ad = drift.get("attribute_drifts")
    if isinstance(ad, dict) and ad:
        ranked = sorted(
            ((str(k), float(v)) for k, v in ad.items() if isinstance(v, (int, float))),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        if ranked:
            top_s = ", ".join(f"{k}={v:.3f}" for k, v in ranked)
            bullets.append(f"속성별 드리프트 상위: {top_s}.")

    if any(drift.get(k) is not None for k in ("files_added", "files_removed", "files_common")):
        bullets.append(
            f"파일 변화: 추가 {drift.get('files_added', '—')}, "
            f"삭제 {drift.get('files_removed', '—')}, 공통 {drift.get('files_common', '—')}."
        )

    if drift.get("embedding_drift") is not None:
        bullets.append(f"임베딩 공간 드리프트(앙상블) 점수: {float(drift['embedding_drift']):.4f}.")

    return bullets


def _drift_quality_notes(drift: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    st = str(_drift_status_for_ui(drift.get("overall_score")))
    if st == "CRITICAL":
        notes.append(
            "종합 점수가 위험 구간입니다. 배포/재학습 전에 데이터 수집 파이프라인·라벨·전처리 변경 여부를 점검하세요."
        )
    elif st == "WARNING":
        notes.append("주의 구간입니다. 속성·임베딩 차트에서 변화가 큰 항목을 우선 검토하세요.")

    ad = drift.get("attribute_drifts")
    if isinstance(ad, dict):
        for k, v in ad.items():
            if isinstance(v, (int, float)) and float(v) >= 0.25:
                notes.append(f"속성 «{k}»의 드리프트가 높습니다({float(v):.3f}). 해당 지표 정의와 수집 조건을 확인하세요.")

    if not notes:
        notes.append("드리프트 수치는 분포 차이를 요약합니다. 도메인 기준으로 허용치를 별도 정의하는 것이 좋습니다.")
    return notes


def _build_drift_chart_blocks(drift: dict[str, Any]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    ad = drift.get("attribute_drifts")
    if isinstance(ad, dict) and ad:
        num_ad = {str(k): float(v) for k, v in ad.items() if isinstance(v, (int, float))}
        svg = _vertical_bar_svg("Attribute Drifts", num_ad, color="#6366f1", height=290)
        if svg:
            blocks.append(
                {
                    "title": "Attribute Drifts",
                    "caption": "페이지 카드와 동일하게 항목별 막대 차트로 표시합니다.",
                    "svg": svg,
                }
            )

    det = drift.get("embedding_drift_detailed")
    if isinstance(det, dict):
        ns = det.get("normalized_scores")
        if isinstance(ns, dict) and ns:
            svg2 = _vertical_bar_svg(
                "Embedding Drift · normalized_scores",
                {str(k): float(v) for k, v in ns.items() if isinstance(v, (int, float))},
                color="#6366f1",
                height=290,
            )
            if svg2:
                blocks.append(
                    {
                        "title": "Embedding Drift · normalized_scores",
                        "caption": "페이지 Embedding Drift 카드의 normalized_scores 차트 형식을 맞춘 표시입니다.",
                        "svg": svg2,
                    }
                )

    return blocks


def _build_attribute_distribution_blocks(dist_payload: dict[str, Any] | None) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    if not isinstance(dist_payload, dict):
        return blocks
    metrics = dist_payload.get("metrics")
    if not isinstance(metrics, dict):
        return blocks
    for metric_name, metric in list(metrics.items())[:6]:
        if not isinstance(metric, dict):
            continue
        base = metric.get("base") if isinstance(metric.get("base"), dict) else {}
        target = metric.get("target") if isinstance(metric.get("target"), dict) else {}
        bins = base.get("bins")
        base_counts = base.get("counts")
        target_counts = target.get("counts")
        if not isinstance(bins, list) or not isinstance(base_counts, list) or not isinstance(target_counts, list):
            continue
        try:
            bins_f = [float(x) for x in bins]
            base_f = [float(x) for x in base_counts]
            target_f = [float(x) for x in target_counts]
        except (TypeError, ValueError):
            continue
        base_svg = _hist_counts_svg(
            f"{metric_name} · base",
            bins_f,
            base_f,
            color="#7c3aed",
            caption="base histogram",
            width=280,
            height=172,
        )
        target_svg = _hist_counts_svg(
            f"{metric_name} · target",
            bins_f,
            target_f,
            color="#84cc16",
            caption="target histogram",
            width=280,
            height=172,
        )
        if base_svg and target_svg:
            blocks.append(
                {
                    "title": f"Attribute Distributions: {metric_name}",
                    "caption": f"score={metric.get('score', '-')} · method={metric.get('method', '-')}",
                    "svg_left": base_svg,
                    "svg_right": target_svg,
                    "left_label": "base",
                    "right_label": "target",
                }
            )
        elif base_svg:
            blocks.append(
                {
                    "title": f"Attribute Distributions: {metric_name} (base)",
                    "caption": f"score={metric.get('score', '-')} · method={metric.get('method', '-')}",
                    "svg": base_svg,
                }
            )
        elif target_svg:
            blocks.append(
                {
                    "title": f"Attribute Distributions: {metric_name} (target)",
                    "caption": f"score={metric.get('score', '-')} · method={metric.get('method', '-')}",
                    "svg": target_svg,
                }
            )
    return blocks


def _build_embedding_projection_blocks(proj_payload: dict[str, Any] | None) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    if not isinstance(proj_payload, dict):
        return blocks
    points = proj_payload.get("points")
    if not isinstance(points, list):
        return blocks
    method = str(proj_payload.get("method") or "pca").upper()
    sampling = proj_payload.get("sampling") if isinstance(proj_payload.get("sampling"), dict) else {}
    sample_caption = ""
    if sampling:
        n = sampling.get("n")
        base_count = sampling.get("base_count")
        target_count = sampling.get("target_count")
        if n is not None and base_count is not None and target_count is not None:
            sample_caption = f"method={method} · sampled n={n} (base={base_count}, target={target_count})"
    svg = _embedding_projection_split_svg(
        f"Embedding Projection ({method})",
        points,
        caption=sample_caption or f"method={method}",
    )
    if svg:
        blocks.append(
            {
                "title": "Embedding Projection",
                "caption": "base/target 분포를 2D 투영으로 비교합니다.",
                "svg": svg,
            }
        )
    return blocks


def _plan_includes_drift_step(plan: Plan) -> bool:
    return any(getattr(s, "type", None) == StepType.drift for s in (plan.steps or []))


def _target_summary_bullets(eda_target: dict[str, Any] | None) -> list[str]:
    if not isinstance(eda_target, dict):
        return []
    summary = eda_target.get("summary")
    if not isinstance(summary, dict):
        return []
    lines: list[str] = []
    nf = summary.get("num_files") or summary.get("files_analyzed")
    if nf is not None:
        lines.append(f"Target 요약: 분석 파일(또는 레코드) 수 {nf}.")
    if summary.get("n_clusters") is not None:
        lines.append(f"Target 임베딩 클러스터 수: {summary['n_clusters']}.")
    return lines


def render_report(
    *,
    out_dir: Path,
    plan: Plan,
    eda_base: Optional[dict[str, Any]],
    eda_target: Optional[dict[str, Any]],
    drift: Optional[dict[str, Any]],
    drift_attribute_distributions: Optional[dict[str, Any]] = None,
    drift_embedding_projection: Optional[dict[str, Any]] = None,
    formats: list[ReportFormat],
) -> dict[str, Optional[str]]:
    """
    최소 리포트 렌더러.

    - HTML: 항상 생성
    - PDF: WeasyPrint 기반 생성 (실패하면 None)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    name = Path(plan.base_path or "dataset").name
    rows, cols = (0, 0)
    if eda_base and isinstance(eda_base.get("shape"), (list, tuple)) and len(eda_base["shape"]) == 2:
        rows, cols = int(eda_base["shape"][0]), int(eda_base["shape"][1])

    summary = (eda_base or {}).get("summary", {}) if isinstance(eda_base, dict) else {}
    if not isinstance(summary, dict):
        summary = {}

    modality = (eda_base or {}).get("modality") if isinstance(eda_base, dict) else None
    if not modality and plan.modality:
        modality = str(plan.modality)

    missing_rate = (eda_base or {}).get("missing_rate", {}) if isinstance(eda_base, dict) else {}
    if not isinstance(missing_rate, dict):
        missing_rate = {}

    distributions = (eda_base or {}).get("distributions", {}) if isinstance(eda_base, dict) else {}
    distributions_basic = (eda_base or {}).get("distributions_basic", {}) if isinstance(eda_base, dict) else {}
    distributions_attributes = (eda_base or {}).get("distributions_attributes", {}) if isinstance(eda_base, dict) else {}
    label_distributions = summary.get("label_distributions", {}) if isinstance(summary, dict) else {}

    chart_blocks: list[dict[str, str]] = []
    for source_name, source in [
        ("Distributions", distributions),
        ("Basic Distributions", distributions_basic),
        ("Attribute Distributions", distributions_attributes),
    ]:
        if not isinstance(source, dict):
            continue
        for metric_name, hist in source.items():
            if not isinstance(hist, dict):
                continue
            x_lab = f"{metric_name} (bin 경계)"
            cap = f"{metric_name} 값 구간별 빈도(파일/레코드 수). 막대 높이는 해당 bin에 속한 개수입니다."
            svg = _hist_svg(f"{source_name}: {metric_name}", hist, x_label=x_lab, caption=cap)
            if svg:
                chart_blocks.append({"title": f"{source_name}: {metric_name}", "caption": cap, "svg": svg})

    if isinstance(label_distributions, dict):
        for label_name, label_dist in label_distributions.items():
            if not isinstance(label_dist, dict):
                continue
            cap = "클래스(또는 라벨)별 빈도 상위 항목입니다."
            svg = _dict_bar_svg(f"Label Distribution: {label_name}", label_dist)
            if svg:
                chart_blocks.append({"title": f"Label Distribution: {label_name}", "caption": cap, "svg": svg})

    embedding_projection = (eda_base or {}).get("embedding_projection", {}) if isinstance(eda_base, dict) else {}
    embedding_clustering = (eda_base or {}).get("embedding_clustering", {}) if isinstance(eda_base, dict) else {}

    cluster_proj = None
    if isinstance(embedding_clustering, dict):
        cp = embedding_clustering.get("projection")
        if isinstance(cp, dict) and isinstance(cp.get("points"), list):
            cluster_proj = cp

    proj_points: list[dict[str, Any]] = []
    proj_method = "pca"
    if isinstance(embedding_projection, dict):
        proj_method = str(embedding_projection.get("method") or "pca")
        if isinstance(embedding_projection.get("points"), list):
            proj_points = embedding_projection["points"]

    scatter_points = cluster_proj["points"] if cluster_proj else proj_points
    cluster_method = None
    n_clusters = None
    sampling = None
    if isinstance(embedding_clustering, dict):
        cluster_method = str(embedding_clustering.get("method") or "kmeans")
        n_clusters = embedding_clustering.get("n_clusters")
    if cluster_proj and isinstance(cluster_proj.get("sampling"), dict):
        sampling = cluster_proj["sampling"]

    if scatter_points:
        svg = _scatter_svg_clustered(
            "임베딩 2D 투영",
            scatter_points,
            projection_method=proj_method.upper(),
            cluster_method=cluster_method,
            n_clusters=n_clusters,
            sampling=sampling,
        )
        if svg:
            cap = "포인트 색은 클러스터 ID 기준입니다(웹 UI와 동일 팔레트). 클러스터가 없으면 단색입니다."
            chart_blocks.append(
                {
                    "title": f"Embedding 2D ({proj_method})",
                    "caption": cap,
                    "svg": svg,
                }
            )

    clusters_list: list[dict[str, Any]] = []
    if isinstance(embedding_clustering, dict) and isinstance(embedding_clustering.get("clusters"), list):
        clusters_list = embedding_clustering["clusters"]
    if clusters_list:
        csvg = _cluster_sizes_svg("클러스터 크기(샘플 수)", clusters_list)
        if csvg:
            chart_blocks.append(
                {
                    "title": "Cluster sizes",
                    "caption": "각 클러스터에 할당된 샘플 수입니다.",
                    "svg": csvg,
                }
            )

    plan_dump = plan.model_dump()
    eda_kpi_rows = _build_kpi_rows(summary, eda_base if isinstance(eda_base, dict) else None)
    eda_executive_bullets = _executive_bullets(
        modality, summary, eda_base if isinstance(eda_base, dict) else None, name
    )
    eda_quality_notes = _quality_notes(summary, str(modality) if modality else None)

    drift_d: dict[str, Any] = drift if isinstance(drift, dict) else {}
    base_label = _label_from_path(plan.base_path)
    target_label = _label_from_path(plan.target_path) if plan.target_path else "—"
    use_drift_template = _plan_includes_drift_step(plan)

    rows_cols_note = ""
    if modality in ("vision_image", "image") or (not rows and not cols and modality):
        rows_cols_note = "표 형태의 행/열 개념이 없는 모달리티이거나, 요약에 shape가 없습니다."
    elif not rows and not cols:
        rows_cols_note = "EDA 요약에 shape 정보가 없습니다."

    missing_note = ""
    if not missing_rate:
        missing_note = "결측률 객체가 비어 있습니다. 모달리티에 따라 결측 정의가 적용되지 않았을 수 있습니다."

    generated_at = _utc_now_iso()
    report_version = _report_package_version()
    plan_name = plan_dump.get("name", "—")
    plan_modality = str(plan_dump.get("modality", "—"))

    if use_drift_template:
        drift_template = _env.get_template("drift_report.html")
        drift_executive_bullets = (
            _drift_executive_bullets(drift_d, base_label=base_label, target_label=target_label)
            + _target_summary_bullets(eda_target if isinstance(eda_target, dict) else None)
        )
        drift_appendix_explanation = (
            "원시 drift.raw와 동일한 수치·요약이되, "
            "attribute_values_ref/cur(파일별 속성값 배열)와 embedding_index_ref/cur(임베딩 인덱스 전문)은 "
            "부록 길이·PDF 용량 때문에 JSON에서 완전히 제거했습니다. 전체 원본은 drift.raw 아티팩트를 참고하세요."
        )
        html = drift_template.render(
            base_label=base_label,
            target_label=target_label,
            generated_at=generated_at,
            report_version=report_version,
            drift_executive_bullets=drift_executive_bullets,
            drift_kpi_rows=_build_drift_kpi_rows(drift_d),
            drift_quality_notes=_drift_quality_notes(drift_d),
            chart_blocks_drift=(
                _build_drift_chart_blocks(drift_d)
                + _build_attribute_distribution_blocks(drift_attribute_distributions)
                + _build_embedding_projection_blocks(drift_embedding_projection)
            ),
            drift_appendix_explanation=drift_appendix_explanation,
            drift_json=_json_pretty(_compact_drift_for_appendix(drift_d)),
            plan_name=plan_name,
            plan_modality=plan_modality,
        )
    else:
        eda_template = _env.get_template("eda_report.html")
        html = eda_template.render(
            id="report",
            name=name,
            rows=rows,
            cols=cols,
            modality=modality or "—",
            generated_at=generated_at,
            report_version=report_version,
            plan_name=plan_name,
            plan_modality=plan_modality,
            executive_bullets=eda_executive_bullets,
            kpi_rows=eda_kpi_rows,
            quality_notes=eda_quality_notes,
            rows_cols_note=rows_cols_note,
            missing_note=missing_note,
            missing_json=_json_pretty(missing_rate),
            summary_json=_json_pretty(summary),
            chart_blocks=chart_blocks,
            plan=plan_dump,
            eda_base=eda_base,
            eda_target=eda_target,
            drift=drift,
        )

    html_path = out_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")

    pdf_path: Optional[Path] = out_dir / "report.pdf"
    pdf_error: str | None = None
    if ReportFormat.pdf in formats:
        try:
            HTML(string=html, base_url=str(out_dir.resolve())).write_pdf(str(pdf_path))
            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                pdf_error = "PDF 파일이 생성되지 않았거나 크기가 0입니다."
                pdf_path = None
        except Exception as e:
            pdf_error = str(e)
            pdf_path = None
    else:
        pdf_path = None

    out: dict[str, Any] = {"html": str(html_path), "pdf": str(pdf_path) if pdf_path else None}
    if pdf_error and not out["pdf"]:
        out["error"] = pdf_error
    return out
