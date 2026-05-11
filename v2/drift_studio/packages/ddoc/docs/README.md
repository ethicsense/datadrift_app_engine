# ddoc (drift_studio v2 minimal)

`ddoc`는 drift_studio(v2)에서 **플러그인 기반 EDA/Drift 분석을 실행하고, 결과 캐시를 저장**하기 위한 최소 런타임/CLI 패키지입니다.

## 제공 기능

- **CLI**
  - `ddoc analyze eda`: 단일 데이터셋 EDA 실행 + 캐시 저장
  - `ddoc analyze drift`: 두 데이터셋 간 drift 실행 + 캐시/산출물 저장
  - `ddoc plugin list/info`: 설치된 플러그인(엔트리포인트) 조회
  - `--version`, `--help`
- **플러그인 시스템**
  - `importlib.metadata`의 entry points(`group="ddoc"`)로 플러그인을 로드합니다.
  - 모달리티별 provider는 `ddoc/core/analysis_facade.py`에서 결정됩니다.

## 설치(로컬 개발)

drift_studio 리포지토리 루트에서:

```bash
./venv/bin/python -m pip install -e packages/ddoc
./venv/bin/python -m pip install -e packages/ddoc/plugins/ddoc-plugin-text
./venv/bin/python -m pip install -e packages/ddoc/plugins/ddoc-plugin-timeseries
./venv/bin/python -m pip install -e packages/ddoc/plugins/ddoc-plugin-audio-wave
./venv/bin/python -m pip install -e packages/ddoc/plugins/ddoc-plugin-audio-midi
./venv/bin/python -m pip install -e packages/ddoc/plugins/ddoc-plugin-vision-image
./venv/bin/python -m pip install -e packages/ddoc/plugins/ddoc-plugin-vision-video
```

## 데이터셋 입력 규약

- CLI 입력은 **디렉토리** 또는 **`.zip`** 입니다.
- 데이터셋 루트에 **`ddoc.yaml`이 필수**입니다.
- 모달리티는 `ddoc.yaml`의 `modality`로 결정됩니다.

## 사용 예시

### EDA

```bash
./venv/bin/python -m ddoc.cli.main analyze eda /path/to/dataset_dir --out analysis
./venv/bin/python -m ddoc.cli.main analyze eda /path/to/dataset.zip --out analysis
```

### Drift

```bash
./venv/bin/python -m ddoc.cli.main analyze drift /path/to/base_dir /path/to/target_dir --out analysis
./venv/bin/python -m ddoc.cli.main analyze drift /path/to/base.zip /path/to/target.zip --out analysis
```

## 출력물/캐시 위치

- `--out <dir>` 아래에 결과가 생성됩니다.
  - EDA: `<out>/eda/<dataset_name>/...`
  - Drift: `<out>/drift/<base_name>__<target_name>/...`
  - `.zip` 입력의 경우 임시 압축해제 디렉토리: `<out>/_inputs/...`
- 플러그인은 내부적으로 `ddoc.core.cache_service.CacheService`를 사용해 `.ddoc/cache`에 캐시를 저장할 수 있습니다.

## 현재 디렉토리 구조(요약)

- `ddoc/cli/`: 최소 CLI(analyze/plugin)
- `ddoc/core/`: 플러그인 로더/파사드/캐시/스키마
- `plugins/`: hookspecs
- `plugins/*`: 모달리티 플러그인 패키지들