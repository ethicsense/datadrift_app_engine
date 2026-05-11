import os
import shutil

from sqlalchemy.orm import Session

from app.models import Dataset
from app.services.dvc_service import (
    save_uploaded_dataset,
    process_zip_dataset,
    BASE_DATA_DIR
)
from app.utils.json_sanitize import clean_json_value
from driftstudio_runtime.infer import validate_dataset_dir_or_raise


def create_dataset(db: Session, uploaded_file):
    # 1) 공통 저장 로직 (dataset 폴더만 생성)
    dataset_id, raw_original_path = save_uploaded_dataset(uploaded_file)

    filename = uploaded_file.filename
    ext = os.path.splitext(filename)[1].lower()

    dataset_dir = os.path.dirname(raw_original_path)

    # ============================================================
    # v2 규약: 업로드는 ZIP만 허용 + ZIP 내부에 ddoc.yaml 필수
    # ============================================================
    if ext != ".zip":
        # 저장된 원본 파일 정리
        try:
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
        except Exception:
            pass
        raise ValueError("모든 데이터셋은 .zip 형식으로 업로드되어야 합니다(ddoc.yaml 포함 필수)")

    info = process_zip_dataset(dataset_id, raw_original_path)
    extracted_dir = info.get("root_dir")
    if not extracted_dir:
        raise ValueError("ZIP 압축 해제 경로(root_dir)를 확인할 수 없습니다")

    # 메타/파일구성 검증 (불일치 시 처리 중단)
    meta = validate_dataset_dir_or_raise(extracted_dir)

    preview = {
        # zip_type은 legacy UI 호환용 필드였고,
        # v2 규약에서는 ddoc.yaml의 modality가 유일한 판정 근거입니다.
        "zip_type": meta.modality,
        "tree": info.get("tree"),
        "stats": info.get("stats"),
        "sample_image": info.get("sample_image"),
        "meta": meta.model_dump(),
    }

    # UI 호환을 위해 type은 zip 유지, 실제 처리 루트는 extracted_dir로 고정
    dtype = "zip"
    dvc_path = extracted_dir
    rows, cols = 0, 0

    # ============================================================
    # Dataset DB 저장
    # ============================================================

    dataset = Dataset(
        id=dataset_id,
        name=filename,
        type=dtype,
        dvc_path=dvc_path,
        version="v1",
        preview=clean_json_value(preview),
        rows=rows,
        cols=cols,
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset


def delete_dataset(db: Session, dataset_id: str):
    """
    데이터셋과 관련된 모든 데이터를 삭제합니다.
    
    1. 관련 EDA 결과 삭제
    2. 관련 Drift 결과 삭제 (base 또는 target으로 사용된 경우)
    3. 파일 시스템에서 데이터셋 폴더 삭제
    4. Dataset DB 레코드 삭제
    """
    # 1) 데이터셋 존재 확인
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return {"success": False, "message": "데이터셋을 찾을 수 없습니다."}

    deleted_info = {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "files_deleted": False
    }

    # 2) 파일 시스템에서 데이터셋 폴더 삭제
    dataset_dir = os.path.join(BASE_DATA_DIR, dataset_id)
    if os.path.exists(dataset_dir):
        try:
            shutil.rmtree(dataset_dir)
            deleted_info["files_deleted"] = True
        except Exception as e:
            # 파일 삭제 실패해도 DB는 삭제 진행
            deleted_info["files_deleted"] = False
            deleted_info["file_error"] = str(e)

    # 3) Dataset 레코드 삭제
    db.delete(dataset)
    db.commit()

    return {"success": True, "deleted": deleted_info}
