from .datasets import DATASET_CATALOG, download_all, load_olist, load_superstore, load_uci_online_retail_ii
from .kaggle_data import (
    build_all_kaggle_panels,
    download_kaggle_all,
    load_favorita_panels,
    load_instacart_panels,
    load_m5_panels,
)
from .metrics import commerce_daily_metrics, entity_churn, mix_shares
from .drift import (
    compare_windows,
    js_divergence,
    ks_wasserstein,
    psi,
    revenue_decomposition,
    rolling_drift,
    total_variation,
)
from .plotting import configure_matplotlib, finish_figure, format_date_axis, plot_metric_stack, style_area

__all__ = [
    "DATASET_CATALOG",
    "download_all",
    "download_kaggle_all",
    "build_all_kaggle_panels",
    "load_olist",
    "load_superstore",
    "load_uci_online_retail_ii",
    "load_m5_panels",
    "load_favorita_panels",
    "load_instacart_panels",
    "commerce_daily_metrics",
    "entity_churn",
    "mix_shares",
    "compare_windows",
    "js_divergence",
    "ks_wasserstein",
    "psi",
    "revenue_decomposition",
    "rolling_drift",
    "total_variation",
    "configure_matplotlib",
    "finish_figure",
    "format_date_axis",
    "plot_metric_stack",
    "style_area",
]
