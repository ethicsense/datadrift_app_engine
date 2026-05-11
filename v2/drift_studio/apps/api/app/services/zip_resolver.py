import os
import shutil
import zipfile

# NOTE:
# - drift_studio v2 정책: "내장 분석 로직"은 제거합니다.
# - 이 모듈은 ZIP 업로드를 위한 유틸로서,
#   1) 압축 해제
#   2) 불필요 파일/폴더 정리
#   3) (필요 시) 단일 폴더 중첩 평탄화
#   4) UI용 트리/간단 통계 생성
# 만 담당합니다.


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
TEXT_EXT = {".txt"}
CSV_EXT = {".csv"}
JSON_EXT = {".json"}
XML_EXT = {".xml"}
YAML_EXT = {".yaml", ".yml"}
JUNK_FOLDERS = {"__MACOSX", ".git", ".svn", "__pycache__", ".idea"}
JUNK_FILES = {".DS_Store", "Thumbs.db", ".gitkeep", ".gitignore"}
JUNK_PREFIXES = ("._",)


def _is_junk(name: str) -> bool:
    if name in JUNK_FOLDERS or name in JUNK_FILES:
        return True
    if any(name.startswith(prefix) for prefix in JUNK_PREFIXES):
        return True
    return False


def _cleanup_extracted(directory: str) -> None:
    # 루트 레벨 junk 폴더 먼저 제거
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if item in JUNK_FOLDERS and os.path.isdir(item_path):
            shutil.rmtree(item_path)

    # 재귀 제거
    for root, dirs, files in os.walk(directory, topdown=False):
        for d in list(dirs):
            if d in JUNK_FOLDERS:
                dir_path = os.path.join(root, d)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
        for f in list(files):
            if _is_junk(f):
                fp = os.path.join(root, f)
                if os.path.exists(fp):
                    os.remove(fp)


def _move_contents_up(src_dir: str, dest_dir: str) -> None:
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dest_dir, name)
        if os.path.exists(dst):
            # 충돌 시 뒤에 suffix 부여
            base, ext = os.path.splitext(name)
            i = 1
            while os.path.exists(dst):
                dst = os.path.join(dest_dir, f"{base}_{i}{ext}")
                i += 1
        shutil.move(src, dst)
    # 비어있으면 제거
    try:
        os.rmdir(src_dir)
    except Exception:
        pass


def _is_double_nested(folder: str) -> bool:
    items = [i for i in os.listdir(folder) if not i.startswith(".") and not _is_junk(i)]
    if len(items) != 1:
        return False
    return os.path.isdir(os.path.join(folder, items[0]))


def _flatten_structure(directory: str, zip_stem: str) -> None:
    items = [i for i in os.listdir(directory) if not i.startswith(".") and i not in JUNK_FOLDERS]
    if len(items) != 1:
        return
    single = os.path.join(directory, items[0])
    if not os.path.isdir(single):
        return
    if items[0] == zip_stem or _is_double_nested(single):
        _move_contents_up(single, directory)


def extract_zip_dataset(zip_path: str, dest: str | None = None) -> str:
    """
    ZIP을 압축 해제하고 정리/평탄화까지 수행합니다.
    - dest가 이미 존재하면 삭제 후 재추출합니다(재시도 안전).
    """
    if dest is None:
        dest = f"{zip_path}_extracted"

    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    _cleanup_extracted(dest)
    _flatten_structure(dest, os.path.splitext(os.path.basename(zip_path))[0])
    return dest


def _build_tree(root_dir: str, max_depth: int = 6, max_entries: int = 2000) -> dict:
    """
    UI용 단순 트리.
    """
    root_dir = os.path.abspath(root_dir)
    tree = {"name": os.path.basename(root_dir), "type": "dir", "children": []}
    entries = 0

    def walk(cur_dir: str, node: dict, depth: int):
        nonlocal entries
        if depth > max_depth or entries >= max_entries:
            return
        try:
            names = sorted(os.listdir(cur_dir))
        except Exception:
            return
        for name in names:
            if entries >= max_entries:
                return
            if _is_junk(name):
                continue
            full = os.path.join(cur_dir, name)
            if os.path.isdir(full):
                child = {"name": name, "type": "dir", "children": []}
                node["children"].append(child)
                entries += 1
                walk(full, child, depth + 1)
            else:
                node["children"].append({"name": name, "type": "file"})
                entries += 1

    walk(root_dir, tree, 0)
    return tree


def summarize_extracted_dir(extracted_dir: str) -> dict:
    """
    분석이 아닌, 업로드 후 UI 미리보기를 위한 최소 통계.
    """
    stats = {
        "total_files": 0,
        "image_files": 0,
        "video_files": 0,
        "text_files": 0,
        "csv_files": 0,
        "json_files": 0,
        "xml_files": 0,
        "yaml_files": 0,
        "subdirs": set(),
    }
    sample_image = None
    ddoc_yaml_rel = None

    for root, dirs, files in os.walk(extracted_dir):
        dirs[:] = [d for d in dirs if not _is_junk(d)]

        rel = os.path.relpath(root, extracted_dir)
        if rel != ".":
            top = rel.split(os.sep)[0]
            if not _is_junk(top):
                stats["subdirs"].add(top)

        for f in files:
            if _is_junk(f):
                continue
            stats["total_files"] += 1
            ext = os.path.splitext(f)[1].lower()
            full = os.path.join(root, f)
            if ext in IMAGE_EXT:
                stats["image_files"] += 1
                if sample_image is None:
                    sample_image = full
            elif ext in VIDEO_EXT:
                stats["video_files"] += 1
            elif ext in TEXT_EXT:
                stats["text_files"] += 1
            elif ext in CSV_EXT:
                stats["csv_files"] += 1
            elif ext in JSON_EXT:
                stats["json_files"] += 1
            elif ext in XML_EXT:
                stats["xml_files"] += 1
            elif ext in YAML_EXT:
                stats["yaml_files"] += 1

            # ddoc.yaml 위치(상대경로) 기록 - 루트/하위 모두 탐색
            if f.lower() == "ddoc.yaml" and ddoc_yaml_rel is None:
                ddoc_yaml_rel = os.path.relpath(full, extracted_dir)

    stats["subdirs"] = sorted(list(stats["subdirs"]))
    return {"stats": stats, "sample_image": sample_image, "ddoc_yaml": ddoc_yaml_rel}


def analyze_zip_dataset(zip_path: str) -> dict:
    """
    (호환용) 과거의 'ZIP 구조 자동분석' API 이름을 유지합니다.
    지금은 포맷 판정/내장 분석을 하지 않고, 압축 해제 + 트리/통계만 제공합니다.
    """
    extracted = extract_zip_dataset(zip_path)
    tree = _build_tree(extracted)
    summary = summarize_extracted_dir(extracted)
    return {
        "zip_type": "dataset",  # UI 호환용 placeholder (실제 모달리티는 ddoc.yaml이 결정)
        "root_dir": extracted,
        "tree": tree,
        "stats": summary["stats"],
        "sample_image": summary["sample_image"],
        "ddoc_yaml": summary.get("ddoc_yaml"),
    }


