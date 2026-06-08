from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup

from .models import NormalizedDataset


def render_report(dataset: NormalizedDataset, analysis: dict[str, Any], output_path: Path) -> Path:
    env = Environment(
        loader=PackageLoader("silhouette_outliner", "templates"),
        autoescape=select_autoescape(("html", "xml", "j2")),
    )
    env.filters["currency"] = _currency
    env.filters["percent"] = _percent
    env.filters["percent_or_dash"] = _percent_or_dash
    env.filters["tojson"] = _tojson
    template = env.get_template("report.html.j2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        template.render(dataset=dataset.to_dict(), analysis=analysis),
        encoding="utf-8",
    )
    return output_path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _currency(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _percent(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}%"


def _tojson(value: Any) -> Markup:
    # Embedded payloads sit inside <script type="application/json">, which is NOT
    # HTML-decoded by the browser. autoescape would convert quotes to entities and
    # break JSON.parse; mark the result safe so Jinja keeps it verbatim. We still
    # neutralize "</" to defang any accidental script-closing sequence.
    raw = json.dumps(value, ensure_ascii=False)
    raw = raw.replace("</", "<\\/")
    return Markup(raw)


def _percent_or_dash(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.1f}%"
