# driftstudio_spec

drift_studio의 **데이터셋 규약(입력 포맷)과 스키마 계약(spec)**을 정의합니다.

## 핵심 원칙(중요)

- **모든 데이터셋 입력은 `.zip` 파일 1개로 전달**합니다.
- `.zip` 내부에는 반드시 **메타파일 `ddoc.yaml`(루트 위치)**이 포함되어야 합니다.
- `ddoc.yaml`이 없거나, 값이 스키마/규약을 만족하지 않으면 **분석/처리를 수행하지 않습니다(= fallback 없음)**.
- 텍스트/시계열은 **CSV만 지원**하며, **분석 대상 컬럼은 `ddoc.yaml`에서 명시**해야 합니다.

## ZIP 구조 규약

### 공통

- ZIP 최상위(압축 해제 루트)에 `ddoc.yaml`이 존재해야 합니다.
- `ddoc.yaml` 외의 파일/폴더는 자유지만, 아래 모달리티별 요구사항을 만족해야 합니다.

### 예시(공통)

```
dataset.zip
└── ddoc.yaml
└── (data files...)
```

## ddoc.yaml 스키마(요약)

- 공통 필드
  - `schema_version`: 정수 (현재 1)
  - `name`: (선택) 표시용 이름
  - `modality`: `"vision" | "text" | "timeseries" | "audio" | "video"`
  - `data`: modality별 설정
- **주의**: 정의되지 않은 키가 있으면 검증 실패합니다(extra=forbid).

## 모달리티별 ddoc.yaml 템플릿

### Vision (이미지)

- **요구사항**
  - `data.data_dir` 아래에 이미지 파일이 1개 이상 존재해야 함
  - 이미지 확장자: `.png/.jpg/.jpeg/.bmp/.gif/.webp/.tiff`

```yaml
schema_version: 1
name: "my-vision-dataset"
modality: vision
data:
  data_dir: "."   # 압축 해제 루트 기준 상대경로
```

### Text (CSV만 지원)

- **요구사항**
  - `data.csv`는 반드시 `.csv`
  - `data.columns`는 1개 이상(중복 금지)
  - CSV에 해당 컬럼이 실제로 존재해야 함

```yaml
schema_version: 1
name: "my-text-dataset"
modality: text
data:
  csv: "data.csv"
  columns:
    - "text"
    - "title"
```

### TimeSeries (CSV만 지원)

- **요구사항**
  - `data.csv`는 반드시 `.csv`
  - `data.timestamp_column` 필수(빈 값 불가)
  - `data.numeric_columns` 또는 `data.categorical_columns` 중 최소 1개는 필수
  - `timestamp_column`이 numeric/categorical 컬럼 목록에 포함되면 안 됨
  - numeric/categorical 간 중복 불가

```yaml
schema_version: 1
name: "my-timeseries-dataset"
modality: timeseries
data:
  csv: "timeseries.csv"
  timestamp_column: "ts"
  numeric_columns:
    - "value"
    - "price"
  categorical_columns:
    - "region"
```

### Audio (MIDI만 지원)

- **요구사항**
  - `data.data_dir` 아래에 `.mid` 또는 `.midi` 파일이 1개 이상 존재해야 함

```yaml
schema_version: 1
name: "my-audio-midi-dataset"
modality: audio
data:
  data_dir: "audio"
```

### Video (범용 비디오 파일)

- **요구사항**
  - `data.data_dir` 아래에 비디오 파일이 1개 이상 존재해야 함
  - 지원 확장자: `.mp4/.mov/.avi/.mkv/.webm/.m4v`

```yaml
schema_version: 1
name: "my-video-dataset"
modality: video
data:
  data_dir: "video"
```

## 참고 구현(검증/파서)

- 스키마 모델: `packages/driftstudio_spec/packages/driftstudio_spec/dataset_meta.py`
- 런타임 검증: `packages/driftstudio_runtime/packages/driftstudio_runtime/infer.py`
