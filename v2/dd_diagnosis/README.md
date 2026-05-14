# DD Diagnosis

datadrift_app_engine **v1** 웹앱(Flask + FiftyOne 등)을 **v2** 안에서 단독으로 돌리기 위해 이식한 구성입니다.

**최종 수정: 2026-05-12**

## 구성

| 경로 | 설명 |
|------|------|
| `backend/` | Flask 앱 (`app.py`, `config.py`, `templates/`, `static/` 등) |
| `yolo_cam/` | YOLO Grad-CAM(EigenCAM) 보조 모듈 |
| `run_webapp.py` | 권장 CLI 엔트리 (루트에 `yolo_cam`을 path에 포함) |
| `run_webapp.sh` | venv가 있으면 활성화 후 `run_webapp.py` 실행 |
| `setup_env.sh` | 가상환경 생성 + `requirements.txt` + CLIP 설치 |
| `requirements.txt` | Python 의존성 목록 |

## 요구 사항

- **Python 3.10+** 권장 (Torch / FiftyOne 호환)
- macOS / Linux (Windows는 경로·쉘만 조정하면 동일 개념)

## 설치

프로젝트 루트(`dd_diagnosis/`)에서:

```bash
chmod +x setup_env.sh run_webapp.sh
./setup_env.sh
```

- 이미 `.venv`가 있으면 **재생성하지 않고** 그 안에 패키지만 다시 설치합니다.
- CLIP 설치를 잠시 생략하려면: `SKIP_CLIP=1 ./setup_env.sh`  
  (CAM·임베딩 등 `import clip` 경로는 실패할 수 있습니다.)
- 특정 인터프리터로 venv를 만들려면: `PYTHON=python3.12 ./setup_env.sh`

## 실행

```bash
source .venv/bin/activate
./run_webapp.sh
```

또는:

```bash
source .venv/bin/activate
python run_webapp.py --port 5555 --host 0.0.0.0
```

- 기본 Flask 포트는 `backend/config.py`의 `flask_port`(기본 **5555**)입니다.
- FiftyOne 보조 포트는 `--fiftyone-port` 또는 설정 파일에서 조정합니다.

`backend/`만 단독으로 `python app.py ...` 실행할 때는 `yolo_cam`이 PYTHONPATH에 없을 수 있으므로, **루트의 `run_webapp.py` / `run_webapp.sh` 사용을 권장**합니다.

## 런타임 디렉터리

앱 기동 시 `init_app` → `config.ensure_directories()`로 예를 들어 다음이 생성됩니다.

- `backend/db/`, `backend/models/`, `backend/logs/`
- `backend/datasets/uploads/`, `backend/datasets/exported_datasets/`
- `backend/static/cam_results/`, `backend/static/perturbation_results/`

Git에는 넣지 않도록 `.gitignore`에 포함되어 있습니다.

## 참고

- 상위 레포 **v1**의 전체 CLI·패키지명은 `ddoc` 등으로 통합되어 있었고, 본 디렉터리는 **웹 진단 앱만** 분리한 형태입니다.
- 의존성 버전을 v1과 동일하게 고정하려면 `v1/requirements.txt`를 기준으로 이 파일을 조정하면 됩니다.
