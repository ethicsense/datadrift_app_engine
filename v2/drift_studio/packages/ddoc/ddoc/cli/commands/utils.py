"""CLI shared utilities (minimal).

정리 목표:
- analyze / plugin 커맨드만 남기는 CLI 리빌드에 맞춰,
  legacy workspace(DVC/params/snapshot/exp) 유틸을 제거합니다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import yaml


def _pretty(x: Any) -> str:
    """JSON 또는 객체를 보기 좋게 출력"""
    try:
        # ensure_ascii=False를 사용하여 한글 깨짐 방지
        return json.dumps(x, indent=2, ensure_ascii=False)
    except Exception:
        return str(x)

def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("_") or "dataset"


def _snapshot_id_for(path: Path) -> str:
    try:
        p = str(path.resolve())
    except Exception:
        p = str(path)
    return "path:" + hashlib.sha1(p.encode("utf-8")).hexdigest()[:16]


def _fingerprint_dir(root: Path) -> str:
    """
    가벼운 fingerprint:
    - 파일 내용까지 읽지 않고 (rel_path, size, mtime) 기반으로 해시 생성
    - drift_studio_runtime/runner.py와 동일 계열의 전략
    """
    h = hashlib.sha256()
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        # 숨김/정크 제외
        dirnames[:] = [d for d in dirnames if d not in {"__MACOSX"} and not d.startswith(".")]
        for fn in sorted(filenames):
            if fn in {".DS_Store", "Thumbs.db"} or fn.startswith("._"):
                continue
            fp = Path(dirpath) / fn
            try:
                rel = fp.relative_to(root).as_posix()
                st = fp.stat()
            except Exception:
                continue
            h.update(rel.encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
            h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()


def _cleanup_extracted(dest: Path) -> None:
    # 최소 정크 제거 (Studio 쪽 zip_resolver의 축약 버전)
    junk_dirs = {"__MACOSX", ".git", ".svn", "__pycache__", ".idea"}
    junk_files = {".DS_Store", "Thumbs.db", ".gitkeep", ".gitignore"}

    for root, dirs, files in os.walk(dest, topdown=False):
        for d in list(dirs):
            if d in junk_dirs:
                p = Path(root) / d
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)
        for f in list(files):
            if f in junk_files or f.startswith("._"):
                p = Path(root) / f
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass


def resolve_dataset_input(input_path: str, *, work_dir: Path) -> Path:
    """
    input_path:
    - 디렉토리: 그대로 사용
    - .zip: work_dir/_inputs 아래로 압축해제 후 디렉토리 경로 반환
    """
    p = Path(input_path)
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if p.is_dir():
        return p.resolve()

    if p.is_file() and p.suffix.lower() == ".zip":
        dest = work_dir / "_inputs" / f"{p.stem}_extracted"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(dest)
        _cleanup_extracted(dest)

        # 단일 루트 폴더 중첩 평탄화(흔한 케이스)
        items = [x for x in dest.iterdir() if x.name not in {"__MACOSX"} and not x.name.startswith(".")]
        if len(items) == 1 and items[0].is_dir():
            # zip stem과 같거나, 이중 중첩이면 contents를 위로 올림
            inner = items[0]
            inner_items = [x for x in inner.iterdir() if not x.name.startswith(".") and x.name != "__MACOSX"]
            if inner.name == p.stem or (len(inner_items) == 1 and inner_items[0].is_dir()):
                for child in inner.iterdir():
                    shutil.move(str(child), str(dest / child.name))
                try:
                    inner.rmdir()
                except Exception:
                    pass
        return dest.resolve()

    raise ValueError(f"dataset 입력은 디렉토리 또는 .zip 이어야 합니다: {input_path}")


def load_ddoc_yaml(dataset_dir: Path) -> dict[str, Any]:
    meta_path = dataset_dir / "ddoc.yaml"
    if not meta_path.exists():
        raise ValueError(f"ddoc.yaml not found at dataset root: {meta_path}")
    try:
        payload = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise ValueError(f"ddoc.yaml parse failed: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError("ddoc.yaml must be a mapping/object")
    return payload


def infer_modality(dataset_dir: Path) -> str:
    cfg = load_ddoc_yaml(dataset_dir)
    modality = (cfg.get("modality") or "").strip()
    if not modality:
        raise ValueError("ddoc.yaml missing required field: modality")
    return str(modality)


def dataset_identity(dataset_dir: Path) -> tuple[str, str]:
    """
    Returns (snapshot_id, data_hash)
    """
    snapshot_id = _snapshot_id_for(dataset_dir)
    data_hash = _fingerprint_dir(dataset_dir)
    return snapshot_id, data_hash

