"""Generate pad×blur variant_grid YAML snippet (5×6 grid, reuse from prior runs)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SC = REPO / "artifacts/garment_segmentation/soft_consistency_pad_blur-20261007/embeddings/soft_consistency"
P06 = REPO / "artifacts/garment_segmentation/soft_pad06_blur_fine-20261007/embeddings/soft_consistency"

PADS = (0.0, 0.06, 0.12, 0.20, 0.30)
BLURS = (4.0, 8.0, 12.0, 21.0, 32.0, 48.0)


def _name(pad: float, blur: float) -> str:
    return f"p{int(round(pad * 1000)):03d}_b{int(round(blur)):03d}"


def _reuse(pad: float, blur: float) -> str | None:
    rel = lambda p: str(p.relative_to(REPO))
    if pad == 0.0 and blur == 21.0 and (SC / "pad_r0.npy").is_file():
        return rel(SC / "pad_r0.npy")
    if pad == 0.06 and blur == 21.0 and (SC / "pad_r06.npy").is_file():
        return rel(SC / "pad_r06.npy")
    if pad == 0.12 and blur == 21.0 and (SC / "soft_ref.npy").is_file():
        return rel(SC / "soft_ref.npy")
    if pad == 0.20 and blur == 21.0 and (SC / "pad_r20.npy").is_file():
        return rel(SC / "pad_r20.npy")
    if pad == 0.30 and blur == 21.0 and (SC / "pad_r30.npy").is_file():
        return rel(SC / "pad_r30.npy")
    if pad == 0.12 and blur == 4.0 and (SC / "blur_s04.npy").is_file():
        return rel(SC / "blur_s04.npy")
    if pad == 0.12 and blur == 8.0 and (SC / "blur_s08.npy").is_file():
        return rel(SC / "blur_s08.npy")
    if pad == 0.06:
        stem = f"blur_s{int(blur):02d}" if blur < 10 else f"blur_s{int(blur)}"
        if blur == 48:
            stem = "blur_s48"
        p = P06 / f"{stem}.npy"
        if p.is_file():
            return rel(p)
    return None


def build_grid() -> list[dict]:
    rows = []
    for pad in PADS:
        for blur in BLURS:
            name = _name(pad, blur)
            row: dict = {
                "name": name,
                "pad_ratio": pad,
                "dilate_px": 7,
                "background_blur_sigma": blur,
            }
            reuse = _reuse(pad, blur)
            if reuse:
                row["reuse_npy"] = reuse
            rows.append(row)
    return rows


def main() -> None:
    grid = build_grid()
    n_reuse = sum(1 for r in grid if r.get("reuse_npy"))
    print(json.dumps({"n": len(grid), "reuse": n_reuse, "encode": len(grid) - n_reuse}, indent=2))
    example = REPO / "experiments/garment_segmentation/config.preprocess.soft_pad_blur_grid.example.yaml"
    base = yaml.safe_load(example.read_text(encoding="utf-8")) if example.is_file() else {}
    base["variant_grid"] = grid
    out = REPO / "experiments/garment_segmentation/config.preprocess.soft_pad_blur_grid.yaml"
    out.write_text(
        yaml.dump(base, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
