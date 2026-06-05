"""Shared helpers for product-name wordcloud notebooks."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- defaults (notebooks may override before import) ---
MIN_TOKEN_LEN = 2
WORDCLOUD_MAX_WORDS = 120
SKIP_CATEGORY_CODES = frozenset({"000000"})
FILTER_GENERIC_TOKENS = True
AUTO_FILTER_UBIQUITOUS = True
UBIQUITOUS_MIN_CATEGORY_RATIO = 0.75
UBIQUITOUS_MIN_COUNT_PER_CATEGORY = 25

NOISE_STOPWORDS = frozenset({
    "color", "colors", "colour", "colours",
    "size", "sizes", "option", "options",
    "the", "and", "for", "with", "men", "women", "unisex",
    "new", "ver", "edition", "set", "pack",
    "블랙", "화이트", "블루", "레드", "그린", "베이지", "그레이", "브라운", "핑크",
    "black", "white", "blue", "red", "green", "beige", "gray", "grey", "brown", "pink",
})

GENERIC_GARMENT_STOPWORDS = frozenset({
    "티셔츠", "티", "셔츠", "tee", "tshirt", "tshirts", "shirt", "shirts",
    "팬츠", "pants", "pant", "trousers", "trouser", "jeans", "진",
    "bag", "bags", "백", "가방", "백팩", "backpack", "숄더백", "토트백", "토트", "tote",
    "신발", "shoes", "shoe", "스니커즈", "sneakers", "sneaker",
    "스커트", "skirt", "skirts", "드레스", "dress", "dresses", "원피스",
    "모자", "cap", "caps", "hat", "hats",
    "양말", "삭스", "socks", "sock",
    "상의", "하의", "아우터", "outer", "의류", "패션",
    "acc", "accessory", "accessories", "악세서리", "액세서리",
})

CATEGORY_GENERIC_STOPWORDS: dict[str, frozenset[str]] = {
    "001000": frozenset({"상의", "top", "tops", "맨투맨"}),
    "002000": frozenset({"아우터", "outer", "재킷", "jacket", "jackets", "coat", "coats"}),
    "003000": frozenset({"바지", "bottom", "bottoms"}),
    "004000": frozenset({"bag", "bags", "가방", "백", "백팩", "backpack"}),
}

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]

WINDOW_DISPLAY_ORDER = ("1d", "1w", "rt", "1m")

_HANGUL = re.compile(r"[가-힣]{2,}")
_ENGLISH = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_PURE_ASCII_SKU = re.compile(r"^[A-Z0-9][A-Z0-9:\-]{5,}$")
_SPLIT = re.compile(r"[\s\[\]\(\)_\-/·,:]+")

FONT_PATH: str | None = None
MPL_FONT: fm.FontProperties | None = None


def resolve_font_path() -> str | None:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return path
    return None


def configure_matplotlib_korean(font_path: str | None) -> fm.FontProperties | None:
    if not font_path:
        return None
    try:
        fm.fontManager.addfont(font_path)
    except (ValueError, OSError):
        pass
    prop = fm.FontProperties(fname=font_path)
    family = prop.get_name()
    plt.rcParams["font.family"] = family
    plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return prop


def init_fonts() -> tuple[str | None, fm.FontProperties | None]:
    global FONT_PATH, MPL_FONT
    FONT_PATH = resolve_font_path()
    MPL_FONT = configure_matplotlib_korean(FONT_PATH)
    return FONT_PATH, MPL_FONT


def list_run_dirs(runs_dir: Path) -> list[Path]:
    candidates = []
    for path in runs_dir.iterdir():
        if path.is_dir() and (path / "normalized.json").is_file():
            candidates.append(path)
    return sorted(candidates, key=lambda p: (p / "normalized.json").stat().st_mtime, reverse=True)


def resolve_run_dir(runs_dir: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        run_dir = run_dir.resolve()
        if not (run_dir / "normalized.json").is_file():
            raise FileNotFoundError(f"normalized.json 없음: {run_dir}")
        return run_dir
    runs = list_run_dirs(runs_dir)
    if not runs:
        raise FileNotFoundError(f"{runs_dir} 아래에 normalized.json이 있는 run이 없습니다.")
    return runs[0]


def load_items(run_dir: Path) -> list[dict]:
    with (run_dir / "normalized.json").open(encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("normalized.json: items가 리스트가 아닙니다.")
    return items


def product_display_name(item: dict) -> str:
    return (item.get("product_clean") or item.get("product") or "").strip()


def category_key(item: dict) -> tuple[str, str]:
    code = str(item.get("category_code") or item.get("category_major_code") or "")
    label = str(item.get("category_label") or item.get("category_parent_label") or code)
    return code, label


def window_key(item: dict) -> tuple[str, str]:
    wid = str(item.get("ranking_window_id") or "")
    label = str(item.get("ranking_window_label") or wid)
    return wid, label


def product_items(items: list[dict]) -> list[dict]:
    return [it for it in items if it.get("sub_pan") == "product"]


def label_anchor_tokens(label: str) -> frozenset[str]:
    anchors: set[str] = set()
    for m in _HANGUL.finditer(label):
        if len(m.group()) >= MIN_TOKEN_LEN:
            anchors.add(m.group())
    for m in _ENGLISH.finditer(label):
        w = m.group().lower()
        if len(w) >= MIN_TOKEN_LEN:
            anchors.add(w)
    return frozenset(anchors)


def discover_ubiquitous_tokens(
    raw_by_cat: dict[tuple[str, str], Counter],
    *,
    min_category_ratio: float = UBIQUITOUS_MIN_CATEGORY_RATIO,
    min_count_per_category: int = UBIQUITOUS_MIN_COUNT_PER_CATEGORY,
) -> frozenset[str]:
    if not raw_by_cat:
        return frozenset()
    n_cats = len(raw_by_cat)
    need = max(2, int(n_cats * min_category_ratio + 0.999))
    token_cat_hits: dict[str, int] = defaultdict(int)
    for ctr in raw_by_cat.values():
        for tok, count in ctr.items():
            if count >= min_count_per_category:
                token_cat_hits[tok] += 1
    return frozenset(tok for tok, hits in token_cat_hits.items() if hits >= need)


def generic_stopwords_for(
    category_code: str,
    category_label: str,
    ubiquitous: frozenset[str] | None = None,
) -> frozenset[str]:
    code = str(category_code)
    words = set(GENERIC_GARMENT_STOPWORDS)
    words |= CATEGORY_GENERIC_STOPWORDS.get(code, frozenset())
    words |= label_anchor_tokens(category_label)
    if ubiquitous:
        words |= ubiquitous if isinstance(ubiquitous, (set, frozenset)) else frozenset(ubiquitous)
    return frozenset(words)


def tokenize_product_name(name: str) -> list[str]:
    if not name:
        return []
    tokens: list[str] = []
    for chunk in _SPLIT.split(name):
        chunk = chunk.strip()
        if not chunk or len(chunk) < MIN_TOKEN_LEN:
            continue
        if _PURE_ASCII_SKU.match(chunk):
            continue
        for m in _HANGUL.finditer(chunk):
            w = m.group()
            if w not in NOISE_STOPWORDS:
                tokens.append(w)
        for m in _ENGLISH.finditer(chunk):
            w = m.group().lower()
            if len(w) >= MIN_TOKEN_LEN and w not in NOISE_STOPWORDS:
                tokens.append(w)
    return tokens


def build_category_counters(
    items: list[dict],
    *,
    filter_generic: bool = FILTER_GENERIC_TOKENS,
    auto_ubiquitous: bool = AUTO_FILTER_UBIQUITOUS,
) -> tuple[dict[tuple[str, str], Counter], frozenset[str]]:
    raw_by_cat: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for item in product_items(items):
        code, label = category_key(item)
        if code in SKIP_CATEGORY_CODES:
            continue
        name = product_display_name(item)
        if not name:
            continue
        for tok in tokenize_product_name(name):
            raw_by_cat[(code, label)][tok] += 1

    ubiquitous: frozenset[str] = frozenset()
    if filter_generic and auto_ubiquitous:
        ubiquitous = discover_ubiquitous_tokens(raw_by_cat)

    by_cat: dict[tuple[str, str], Counter] = {}
    for key, ctr in raw_by_cat.items():
        code, label = key
        if filter_generic:
            stop = generic_stopwords_for(code, label, ubiquitous)
            by_cat[key] = Counter({t: c for t, c in ctr.items() if t not in stop})
        else:
            by_cat[key] = ctr
    return by_cat, ubiquitous


def build_category_window_counters(
    items: list[dict],
    *,
    filter_generic: bool = FILTER_GENERIC_TOKENS,
    auto_ubiquitous: bool = AUTO_FILTER_UBIQUITOUS,
) -> tuple[dict[tuple[str, str, str, str], Counter], frozenset[str]]:
    """Key: (category_code, category_label, window_id, window_label)."""
    raw_by_cat, ubiquitous = build_category_counters(
        items, filter_generic=False, auto_ubiquitous=False
    )
    if filter_generic and auto_ubiquitous and not ubiquitous:
        ubiquitous = discover_ubiquitous_tokens(raw_by_cat)

    raw: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    for item in product_items(items):
        code, label = category_key(item)
        if code in SKIP_CATEGORY_CODES:
            continue
        wid, wlabel = window_key(item)
        name = product_display_name(item)
        if not name:
            continue
        for tok in tokenize_product_name(name):
            raw[(code, label, wid, wlabel)][tok] += 1

    out: dict[tuple[str, str, str, str], Counter] = {}
    for key, ctr in raw.items():
        code, label, _, _ = key
        if filter_generic:
            stop = generic_stopwords_for(code, label, ubiquitous)
            out[key] = Counter({t: c for t, c in ctr.items() if t not in stop})
        else:
            out[key] = ctr
    return out, ubiquitous


def counter_for_scope(
    items: list[dict],
    *,
    category_code: str | None = None,
    ranking_window_id: str | None = None,
    ubiquitous: frozenset[str] | None = None,
) -> Counter:
    ctr: Counter = Counter()
    for it in product_items(items):
        code, label = category_key(it)
        if code in SKIP_CATEGORY_CODES:
            continue
        if category_code and code != category_code:
            continue
        if ranking_window_id and str(it.get("ranking_window_id") or "") != ranking_window_id:
            continue
        name = product_display_name(it)
        if not name:
            continue
        stop = generic_stopwords_for(code, label, ubiquitous) if FILTER_GENERIC_TOKENS else frozenset()
        for tok in tokenize_product_name(name):
            if tok in stop:
                continue
            ctr[tok] += 1
    return ctr


def ubiquitous_for_items(items: list[dict]) -> frozenset[str]:
    _, ub = build_category_counters(product_items(items))
    return ub


def make_wordcloud(
    counter: Counter,
    *,
    width: int = 960,
    height: int = 540,
) -> WordCloud:
    freq = dict(counter.most_common(WORDCLOUD_MAX_WORDS))
    kwargs = dict(
        width=width,
        height=height,
        background_color="white",
        max_words=WORDCLOUD_MAX_WORDS,
        colormap="viridis",
        prefer_horizontal=0.85,
    )
    if FONT_PATH:
        kwargs["font_path"] = FONT_PATH
    return WordCloud(**kwargs).generate_from_frequencies(freq)


def _title_font_kwargs() -> dict:
    return {"fontproperties": MPL_FONT} if MPL_FONT else {}


def build_window_panels_by_category(
    cw_counters: dict[tuple[str, str, str, str], Counter],
    window_order: tuple[str, ...] | list[str] | None = None,
) -> dict[str, tuple[str, list[tuple[str, Counter]]]]:
    """카테고리 코드 → (라벨, [(윈도우 부제, Counter), ...])."""
    order = list(window_order or WINDOW_DISPLAY_ORDER)
    raw: dict[str, tuple[str, list[tuple[str, str, Counter]]]] = {}
    for key, ctr in cw_counters.items():
        code, label, wid, wlabel = key
        if code not in raw:
            raw[code] = (label, [])
        raw[code][1].append((wid, wlabel, ctr))

    out: dict[str, tuple[str, list[tuple[str, Counter]]]] = {}
    for code, (label, entries) in raw.items():
        by_wid = {wid: (wlabel, ctr) for wid, wlabel, ctr in entries}
        panels: list[tuple[str, Counter]] = []
        for wid in order:
            if wid in by_wid:
                wlabel, ctr = by_wid[wid]
                panels.append((f"{wlabel} ({wid})", ctr))
        for wid, (wlabel, ctr) in sorted(by_wid.items()):
            if wid in order:
                continue
            panels.append((f"{wlabel} ({wid})", ctr))
        out[code] = (label, panels)
    return out


def plot_wordcloud_grid(
    panels: list[tuple[str, Counter]],
    *,
    suptitle: str,
    ncols: int = 2,
    figsize_per_cell: tuple[float, float] = (7.0, 4.5),
    save_path: Path | None = None,
    wordcloud_width: int = 960,
    wordcloud_height: int = 540,
) -> None:
    if not panels:
        print("표시할 패널이 없습니다.")
        return
    n = len(panels)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
    )
    axes_flat = [axes] if n == 1 else list(axes.flatten())
    title_font = _title_font_kwargs()
    for ax, (subtitle, ctr) in zip(axes_flat, panels):
        if not ctr:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", **title_font)
            ax.set_title(subtitle, **title_font)
            ax.axis("off")
            continue
        wc = make_wordcloud(ctr, width=wordcloud_width, height=wordcloud_height)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{subtitle} — n={sum(ctr.values()):,}", **title_font)
        ax.axis("off")
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14, y=1.02, **title_font)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_wordcloud_column(
    panels: list[tuple[str, Counter]],
    *,
    suptitle: str,
    figsize_per_cell: tuple[float, float] = (14.0, 6.5),
    save_path: Path | None = None,
    wordcloud_width: int = 1280,
    wordcloud_height: int = 720,
) -> None:
    """윈도우별 워드클라우드를 세로 1열(큰 패널)로 표시."""
    plot_wordcloud_grid(
        panels,
        suptitle=suptitle,
        ncols=1,
        figsize_per_cell=figsize_per_cell,
        save_path=save_path,
        wordcloud_width=wordcloud_width,
        wordcloud_height=wordcloud_height,
    )


def sort_window_panels(
    panels: list[tuple[str, Counter]],
    window_ids: list[str],
) -> list[tuple[str, Counter]]:
    order = {wid: i for i, wid in enumerate(window_ids)}

    def sort_key(panel: tuple[str, Counter]) -> tuple[int, str]:
        subtitle = panel[0]
        for wid in window_ids:
            if wid in subtitle:
                return (order.get(wid, 999), subtitle)
        return (999, subtitle)

    return sorted(panels, key=sort_key)
