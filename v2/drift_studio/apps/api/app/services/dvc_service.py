"""
저장소(artifact) 서비스

주의:
- drift_studio v2에서는 workspace/git/dvc 기능을 제외합니다.
- 다만 v1 코드/스키마 호환을 위해 모듈 파일명(`dvc_service.py`)과 필드명(`dvc_path`)은 당분간 유지합니다.
- 여기서 말하는 "dvc_path"는 실제 DVC가 아니라 **로컬 아티팩트 저장 경로**입니다.
"""

import os
import uuid
import shutil

# 로컬 아티팩트 저장 루트 (repo 루트 기준 상대경로)
# - v1의 dvc_storage 명칭을 제거하고, drift_studio 의미에 맞게 변경
BASE_DATA_DIR = os.environ.get("DRIFT_STUDIO_DATA_DIR", "storage/datasets")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# =============================================================
# 1) 업로드 파일 저장만 담당하는 함수
# =============================================================
def save_uploaded_dataset(uploaded_file):
    dataset_id = str(uuid.uuid4())
    dataset_dir = os.path.join(BASE_DATA_DIR, dataset_id)
    ensure_dir(dataset_dir)

    raw_path = os.path.join(dataset_dir, uploaded_file.filename)

    with open(raw_path, "wb") as f:
        shutil.copyfileobj(uploaded_file.file, f)

    return dataset_id, raw_path


# =============================================================
# 2) (호환용) 단일 파일 post-process hook
# =============================================================
def dvc_add_file(file_path: str):
    """
    v1 호환을 위한 함수명이지만 v2에서는 DVC를 사용하지 않습니다.
    현재는 파일 존재 여부만 검증하고 그대로 반환합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    return file_path


# =============================================================
# 3) ZIP 데이터셋 처리 (압축 해제 + 분석 + 폴더 dvc add)
# =============================================================
def process_zip_dataset(dataset_id: str, zip_path: str):
    """
    ZIP 파일을 처리:
      1) zip_resolver를 통해 압축 해제 + 정리 + 평탄화
      2) ZIP 구조 분석 결과 반환
    
    Note: 
      - 압축 해제, 불필요한 파일 제거(__MACOSX, .DS_Store 등), 
        이중 구조 평탄화 로직은 zip_resolver._extract_zip에서 처리됨
      - DVC는 현재 사용하지 않음 (2차 작업에서 ddoc 연동 시 처리)
    """
    from app.services.zip_resolver import analyze_zip_dataset

    # ZIP 분석 (내부적으로 압축 해제 + 정리 + 평탄화 수행)
    info = analyze_zip_dataset(zip_path)

    return info


# =============================================================
# 4) DVC 버전 조회 (UI 용)
# =============================================================
def get_dvc_versions(dataset_id: str):
    """
    v1 호환용 placeholder.
    v2에서는 workspace/git/dvc 기능을 제외하므로 항상 단일 "v1"(논리 버전)만 반환합니다.
    """
    return [{"version": "v1"}]