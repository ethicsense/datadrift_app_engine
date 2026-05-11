from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from .midi_parser import MidiParsed, ticks_to_seconds


def compute_midi_metrics(parsed: MidiParsed) -> Dict[str, Any]:
    notes = parsed.notes
    ppq = parsed.ppq
    tempos = parsed.tempos

    if not notes:
        return {
            "num_notes": 0,
            "duration_sec": 0.0,
        }

    start_tick = min(n.start_tick for n in notes)
    end_tick = max(n.end_tick for n in notes)
    duration_ticks = max(0, end_tick - start_tick)
    duration_sec = float(ticks_to_seconds(duration_ticks, ppq, tempos))
    if duration_sec <= 0:
        duration_sec = 0.0

    pitches = np.array([n.pitch for n in notes], dtype=np.float32)
    vels = np.array([n.velocity for n in notes], dtype=np.float32)
    durs = np.array([n.duration_tick for n in notes], dtype=np.float32)

    # polyphony proxy: count note-ons per tick bucket (coarse)
    # bucket size = ppq/8 (32nd note) to avoid huge tick resolution
    bucket = max(1, int(ppq // 8))
    onset_bins = Counter(int((n.start_tick - start_tick) // bucket) for n in notes)
    polyphony_vals = np.array(list(onset_bins.values()), dtype=np.float32) if onset_bins else np.array([1.0])

    pitch_class = Counter(int(p) % 12 for p in pitches.tolist())

    notes_per_sec = float(len(notes) / duration_sec) if duration_sec > 0 else float(len(notes))

    return {
        "num_notes": int(len(notes)),
        "duration_sec": float(duration_sec),
        "notes_per_sec": float(notes_per_sec),
        "pitch_min": int(np.min(pitches)),
        "pitch_max": int(np.max(pitches)),
        "pitch_range": int(np.max(pitches) - np.min(pitches)),
        "pitch_mean": float(np.mean(pitches)),
        "pitch_std": float(np.std(pitches)),
        "velocity_mean": float(np.mean(vels)),
        "velocity_std": float(np.std(vels)),
        "duration_tick_mean": float(np.mean(durs)),
        "duration_tick_std": float(np.std(durs)),
        "polyphony_bucket_mean": float(np.mean(polyphony_vals)),
        "polyphony_bucket_max": float(np.max(polyphony_vals)),
        "pitch_class_hist": {str(k): int(v) for k, v in sorted(pitch_class.items())},
    }


def compute_psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """
    Population Stability Index (PSI) for numeric distributions.
    """
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0

    # use ref quantiles to make stable bins
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(ref, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_hist / max(1, ref_hist.sum())
    cur_pct = cur_hist / max(1, cur_hist.sum())

    ref_pct = np.clip(ref_pct, eps, 1.0)
    cur_pct = np.clip(cur_pct, eps, 1.0)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_js_divergence(p_counts: Counter, q_counts: Counter, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence for categorical distributions.
    """
    keys = set(p_counts.keys()) | set(q_counts.keys())
    if not keys:
        return 0.0
    p = np.array([p_counts.get(k, 0) for k in keys], dtype=np.float64)
    q = np.array([q_counts.get(k, 0) for k in keys], dtype=np.float64)
    p = p / max(eps, p.sum())
    q = q / max(eps, q.sum())
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * (_kl(p, m) + _kl(q, m))

