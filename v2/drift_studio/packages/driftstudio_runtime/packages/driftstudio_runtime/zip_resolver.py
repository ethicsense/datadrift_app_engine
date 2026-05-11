import os
import shutil
import zipfile
import glob
import yaml  # pip install pyyaml

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
TEXT_EXT = {".txt", ".md", ".jsonl"}
CSV_EXT = {".csv", ".tsv"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}

# 제거할 불필요한 파일/폴더
JUNK_FOLDERS = {"__MACOSX", ".git", ".svn", "__pycache__", ".idea"}
JUNK_FILES = {".DS_Store", "Thumbs.db", ".gitkeep", ".gitignore"}
JUNK_PREFIXES = ("._",)  # macOS 리소스 포크 파일


def _is_junk(name: str) -> bool:
    """불필요한 파일/폴더인지 확인"""
    if name in JUNK_FOLDERS or name in JUNK_FILES:
        return True
    if any(name.startswith(prefix) for prefix in JUNK_PREFIXES):
        return True
    return False


# ------------------------------------------------------------
# ZIP 해제 + 정리 + 평탄화
# ------------------------------------------------------------
def _extract_zip(zip_path: str, dest: str = None) -> str:
    """
    ZIP 파일을 압축 해제하고 정리합니다.
    
    Args:
        zip_path: ZIP 파일 경로
        dest: 압축 해제 대상 디렉토리 (None이면 자동 생성)
    
    Returns:
        압축 해제된 디렉토리 경로
    """
    if dest is None:
        dest = f"{zip_path}_extracted"
    
    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        
        # 압축 해제 후 정리
        _cleanup_extracted(dest)
        _flatten_structure(dest, os.path.splitext(os.path.basename(zip_path))[0])
    
    return dest


def extract_zip_dataset(zip_path: str, dest: str) -> str:
    """
    driftstudio_runtime 용 공개 API.
    - ZIP 파일을 dest로 압축 해제 + 정리 + 평탄화 후, 압축 해제된 디렉토리 경로를 반환합니다.
    """
    # dest가 이미 존재하면, 이전 실행/재시도 흔적이므로 제거 후 재추출
    if os.path.exists(dest):
        shutil.rmtree(dest)
    return _extract_zip(zip_path, dest=dest)


def _cleanup_extracted(directory: str):
    """
    압축 해제된 디렉토리에서 불필요한 파일/폴더 제거
    
    제거 대상:
    - __MACOSX/ 폴더
    - .DS_Store 파일
    - ._ 로 시작하는 파일 (macOS 리소스 포크)
    - Thumbs.db (Windows)
    """
    # 1. 루트 레벨 불필요한 폴더 먼저 제거
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if item in JUNK_FOLDERS and os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"🧹 Removed junk folder: {item}")
    
    # 2. 재귀적으로 불필요한 파일/폴더 제거
    for root, dirs, files in os.walk(directory, topdown=False):
        # 불필요한 폴더 제거
        for d in dirs:
            if d in JUNK_FOLDERS:
                dir_path = os.path.join(root, d)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
        
        # 불필요한 파일 제거
        for f in files:
            if _is_junk(f):
                file_path = os.path.join(root, f)
                if os.path.exists(file_path):
                    os.remove(file_path)


def _flatten_structure(directory: str, zip_stem: str):
    """
    이중 중첩 구조를 평탄화합니다.
    
    예시:
        Before: extracted/archive/archive/train/
        After:  extracted/train/
    
    Args:
        directory: 압축 해제된 디렉토리
        zip_stem: ZIP 파일 이름 (확장자 제외)
    """
    # 숨김 파일/불필요한 파일 제외한 항목 목록
    items = [
        item for item in os.listdir(directory) 
        if not item.startswith('.') and item not in JUNK_FOLDERS
    ]
    
    # 정확히 하나의 폴더만 있는 경우 평탄화 검토
    if len(items) == 1:
        single_item = os.path.join(directory, items[0])
        if os.path.isdir(single_item):
            # ZIP 파일명과 같거나, 이중 중첩인 경우 평탄화
            if items[0] == zip_stem or _is_double_nested(single_item):
                print(f"📂 Flattening nested structure: {items[0]}/")
                _move_contents_up(single_item, directory)


def _is_double_nested(folder: str) -> bool:
    """
    이중 중첩 구조인지 확인합니다.
    
    폴더 안에 하나의 폴더만 있고, 그 폴더가 실제 데이터를 담고 있는지 확인
    """
    items = [
        item for item in os.listdir(folder) 
        if not item.startswith('.') and item not in JUNK_FOLDERS
    ]
    
    if len(items) == 1 and os.path.isdir(os.path.join(folder, items[0])):
        # 하위 폴더에 실제 데이터가 있는지 확인
        sub_folder = os.path.join(folder, items[0])
        sub_items = [
            item.lower() for item in os.listdir(sub_folder) 
            if not item.startswith('.')
        ]
        data_indicators = ['images', 'labels', 'train', 'valid', 'test', 'data.yaml', 'annotations']
        return any(indicator in sub_items for indicator in data_indicators)
    
    return False


def _move_contents_up(source_dir: str, target_dir: str):
    """
    source_dir의 내용물을 target_dir로 이동하고 source_dir 삭제
    """
    import uuid
    
    # 임시 디렉토리로 먼저 이동 (이름 충돌 방지)
    temp_name = f"_temp_{uuid.uuid4().hex[:8]}"
    temp_dir = os.path.join(os.path.dirname(source_dir), temp_name)
    os.rename(source_dir, temp_dir)
    
    # 내용물을 상위로 이동
    for item in os.listdir(temp_dir):
        src = os.path.join(temp_dir, item)
        dst = os.path.join(target_dir, item)
        
        # 충돌 시 덮어쓰기
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        
        shutil.move(src, dst)
    
    # 빈 임시 디렉토리 제거
    if os.path.exists(temp_dir):
        try:
            os.rmdir(temp_dir)
        except OSError:
            shutil.rmtree(temp_dir)


# ------------------------------------------------------------
# Tree view 생성기 (불필요한 파일/폴더 필터링)
# ------------------------------------------------------------
def _build_tree(path: str) -> dict:
    """
    디렉토리 트리 구조를 생성합니다.
    불필요한 파일/폴더(__MACOSX, .DS_Store 등)는 제외합니다.
    """
    name = os.path.basename(path)
    
    # 불필요한 항목 필터링
    if _is_junk(name):
        return None
    
    if os.path.isfile(path):
        return {"name": name}

    children = []
    for item in sorted(os.listdir(path)):
        # 불필요한 항목 스킵
        if _is_junk(item):
            continue
        
        full = os.path.join(path, item)
        child = _build_tree(full)
        if child is not None:
            children.append(child)

    return {"name": name, "children": children}


# ------------------------------------------------------------
# Roboflow용 EDA (class, split별 통계)
# ------------------------------------------------------------
def analyze_roboflow(info: dict) -> dict:
    root = info["root_dir"]
    data_yaml = os.path.join(root, "data.yaml")

    class_names: list[str] = []
    if os.path.exists(data_yaml):
        try:
            with open(data_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            names = data.get("names")
            if isinstance(names, list):
                class_names = names
            elif isinstance(names, dict):
                # dict일 경우 index 순으로 정렬
                class_names = [name for _, name in sorted(names.items(), key=lambda x: int(x[0]))]
        except Exception:
            class_names = []

    splits = {}
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue

        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")

        image_files = []
        label_files = []

        if os.path.isdir(images_dir):
            for r, d, files in os.walk(images_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in IMAGE_EXT:
                        image_files.append(os.path.join(r, f))

        if os.path.isdir(labels_dir):
            for r, d, files in os.walk(labels_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() == ".txt":
                        label_files.append(os.path.join(r, f))

        class_counts = {}
        # YOLO-style 라벨(txt)이면 간단히 class index 통계
        for lf in label_files:
            try:
                with open(lf, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_idx = int(float(parts[0]))
                        class_counts[cls_idx] = class_counts.get(cls_idx, 0) + 1
            except Exception:
                continue

        # class index → class name 매핑
        class_counts_named = {}
        for idx, cnt in class_counts.items():
            if 0 <= idx < len(class_names):
                cname = class_names[idx]
            else:
                cname = f"class_{idx}"
            class_counts_named[cname] = cnt

        splits[split] = {
            "num_images": len(image_files),
            "num_labels": len(label_files),
            "class_counts": class_counts_named,
        }

    return {
        "classes": class_names,
        "splits": splits,
    }


# ------------------------------------------------------------
# ZIP 구조 자동 분석기 (YOLO / Roboflow / COCO / VOC / Bundles)
# ------------------------------------------------------------
def analyze_zip_dataset(zip_path: str) -> dict:
    """
    ZIP 내부 구조 분석 → zip_type 판별 (yolo / roboflow / coco / voc / bundles)
    또한 전체 트리뷰 + sample image도 포함
    """
    extracted = _extract_zip(zip_path)
    tree = _build_tree(extracted)

    stats = {
        "total_files": 0,
        "image_files": 0,
        "text_files": 0,
        "csv_files": 0,
        "video_files": 0,
        "json_files": 0,
        "xml_files": 0,
        "subdirs": set(),
    }

    sample_image = None

    for root, dirs, files in os.walk(extracted):
        # 불필요한 디렉토리 탐색 스킵 (in-place 수정)
        dirs[:] = [d for d in dirs if not _is_junk(d)]
        
        rel = os.path.relpath(root, extracted)
        if rel != ".":
            top = rel.split(os.sep)[0]
            # 불필요한 폴더 제외
            if not _is_junk(top):
                stats["subdirs"].add(top)

        for f in files:
            # 불필요한 파일 스킵
            if _is_junk(f):
                continue
            
            stats["total_files"] += 1
            ext = os.path.splitext(f)[1].lower()
            full = os.path.join(root, f)

            if ext in IMAGE_EXT:
                stats["image_files"] += 1
                if sample_image is None:
                    sample_image = full
            elif ext in TEXT_EXT:
                stats["text_files"] += 1
            elif ext in CSV_EXT:
                stats["csv_files"] += 1
            elif ext in VIDEO_EXT:
                stats["video_files"] += 1
            elif ext == ".json":
                stats["json_files"] += 1
            elif ext == ".xml":
                stats["xml_files"] += 1

    stats["subdirs"] = sorted(list(stats["subdirs"]))

    def has_dir(path_rel: str) -> bool:
        return os.path.isdir(os.path.join(extracted, path_rel))

    has_data_yaml = os.path.exists(os.path.join(extracted, "data.yaml"))

    # ------ Roboflow (train/valid/test + images/labels + data.yaml) ------
    if has_data_yaml and (has_dir("train") or has_dir("valid") or has_dir("test")):
        roboflow_like = True
        for split in ["train", "valid", "test"]:
            split_path = os.path.join(extracted, split)
            if os.path.isdir(split_path):
                imgd = os.path.join(split_path, "images")
                lbld = os.path.join(split_path, "labels")
                # 일부 split이 없어도 전체적으로는 Roboflow 라고 간주
                if not (os.path.isdir(imgd) and os.path.isdir(lbld)):
                    roboflow_like = False
                    break
        if roboflow_like:
            return {
                "zip_type": "roboflow",
                "root_dir": extracted,
                "stats": stats,
                "tree": tree,
                "sample_image": sample_image,
            }

    # ------ YOLO (images + labels) ------
    if has_dir("images") and has_dir("labels"):
        return {
            "zip_type": "yolo",
            "root_dir": extracted,
            "stats": stats,
            "tree": tree,
            "sample_image": sample_image,
        }

    # ------ Pascal VOC ------
    if has_dir("Annotations") and has_dir("JPEGImages"):
        xmls = glob.glob(os.path.join(extracted, "Annotations", "*.xml"))
        jpgs = glob.glob(os.path.join(extracted, "JPEGImages", "*.jpg"))
        if xmls and jpgs:
            return {
                "zip_type": "voc",
                "root_dir": extracted,
                "stats": stats,
                "tree": tree,
                "sample_image": sample_image,
            }

    # ------ COCO ------
    if has_dir("images") and has_dir("annotations"):
        jsons = glob.glob(os.path.join(extracted, "annotations", "*.json"))
        if jsons:
            return {
                "zip_type": "coco",
                "root_dir": extracted,
                "stats": stats,
                "tree": tree,
                "sample_image": sample_image,
            }

    # ------ Bundles ------
    if stats["image_files"] > 0 and stats["csv_files"] == 0 and stats["text_files"] == 0:
        ztype = "image_bundle"
    elif stats["csv_files"] > 0:
        ztype = "tabular_bundle"
    elif stats["text_files"] > 0:
        ztype = "text_bundle"
    else:
        ztype = "unknown_zip"

    return {
        "zip_type": ztype,
        "root_dir": extracted,
        "stats": stats,
        "tree": tree,
        "sample_image": sample_image,
    }

# ---------------------------------------------------------------------
# YOLO / VOC / COCO 분석기 (EDA 상세)
# ---------------------------------------------------------------------

def analyze_yolo(info):
    """
    YOLO format:
        extracted/
            images/
            labels/
            data.yaml
    """
    tree = info.get("tree", {})
    stats = info.get("stats", {})
    classes = []

    # data.yaml 분석 (클래스 이름)
    yaml_path = None
    for f in stats.get("all_files", []):
        if f.endswith("data.yaml"):
            yaml_path = f
            break

    if yaml_path and os.path.exists(yaml_path):
        import yaml
        with open(yaml_path, "r") as y:
            ydata = yaml.safe_load(y)
            classes = ydata.get("names", []) or ydata.get("names", {})

    return {
        "format": "yolo",
        "classes": classes,
        "num_images": stats.get("image_files", 0),
        "num_labels": stats.get("text_files", 0),
        "tree": tree,
    }


def analyze_voc(info):
    """
    VOC format:
        extracted/
            Annotations/*.xml
            JPEGImages/*.jpg
            ImageSets/Main/*.txt
    """
    stats = info.get("stats", {})
    tree = info.get("tree", {})

    # VOC XML 파싱으로 class 통계
    import xml.etree.ElementTree as ET

    class_count = {}

    for f in stats.get("all_files", []):
        if f.endswith(".xml"):
            try:
                tree_xml = ET.parse(f)
                root = tree_xml.getroot()

                for obj in root.findall("object"):
                    name = obj.find("name").text.strip()
                    class_count[name] = class_count.get(name, 0) + 1
            except Exception:
                pass

    return {
        "format": "voc",
        "num_annotations": sum(class_count.values()),
        "class_distribution": class_count,
        "num_images": stats.get("image_files", 0),
        "tree": tree,
    }


def analyze_coco(info):
    """
    COCO format:
        extracted/
            annotations/*.json
            train2017/
            val2017/
    """
    stats = info.get("stats", {})
    tree = info.get("tree", {})

    import json

    class_count = {}
    total_annotations = 0

    # COCO annotation 파일 찾기
    for f in stats.get("all_files", []):
        if f.endswith(".json") and "annotation" in f.lower():
            try:
                with open(f, "r") as j:
                    data = json.load(j)

                categories = {c["id"]: c["name"] for c in data.get("categories", [])}

                for ann in data.get("annotations", []):
                    cls_id = ann["category_id"]
                    cls_name = categories.get(cls_id, "unknown")
                    class_count[cls_name] = class_count.get(cls_name, 0) + 1
                    total_annotations += 1
            except Exception:
                pass

    return {
        "format": "coco",
        "num_annotations": total_annotations,
        "class_distribution": class_count,
        "num_images": stats.get("image_files", 0),
        "tree": tree,
    }