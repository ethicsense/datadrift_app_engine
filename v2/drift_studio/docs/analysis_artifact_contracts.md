# 분석 지표/시각화 아티팩트 규약 (2026-01-21)

이 문서는 Drift Studio v2의 분석 지표 재구성과 시각화 개선을 위한 **아티팩트 계약**을 정의합니다.
기존 결과와의 호환성은 보장하지 않으며, 재분석을 전제로 합니다.

## EDA 아티팩트

### 기본 정보 분포
- type: `eda.distributions.basic.v1`
- 목적: 데이터셋 기본 정보(크기/해상도) 분포 관측
- payload:
  - `distributions.size_mb`
  - `distributions.width`
  - `distributions.height`

### 속성 지표 분포
- type: `eda.distributions.attributes.v1`
- 목적: 이미지 속성 지표 분포 관측
- payload:
  - `distributions.brightness`
  - `distributions.exposure`
  - `distributions.contrast`
  - `distributions.dynamic_range`
  - `distributions.colorfulness`
  - `distributions.edge_density`
  - `distributions.sharpness`
  - `distributions.entropy`
  - `distributions.estimated_noise_sigma`

#### 속성 지표 해석 가이드 (2026-04-27 업데이트)
- `brightness`: grayscale 평균 밝기.
- `exposure`: 18% gray 기준의 log-average luminance 노출 바이어스(EV-like).
  - 0에 가까우면 중간 노출, 양수는 밝은(과노출 경향), 음수는 어두운(저노출 경향).
- `contrast`: grayscale 표준편차(전역 대비).
- `dynamic_range`: `P99 - P1` 유효 톤 폭(극단치/아웃라이어 영향을 완화한 범위).
- `colorfulness`: LAB `a*`, `b*` 채널 표준편차 결합.
- `edge_density`: Canny 엣지 맵의 픽셀 비율(고정 threshold 기반).
- `sharpness`: Laplacian 분산(blur에 민감한 선명도 지표).
- `entropy`: grayscale 히스토그램 Shannon entropy.
- `gaussian_noise_level`: Wavelet-MAD 기반 추정 노이즈 표준편차(`HH` 대역).
  - 사용자 표시명은 `estimated_noise_sigma`를 권장하며, `gaussian_noise_level`은 하위호환 키로 유지.

### 제거 지표
- `noise_level`은 `gaussian_noise_level`의 alias이므로 제거
- `quality_score`는 해석 불명확한 휴리스틱이므로 제거

### 임베딩 클러스터링(EDA)
- type: `embedding.clustering.v1`
- 목적: 이미지/텍스트 임베딩의 **클러스터링 결과**를 UI에서 바로 시각화(클러스터 수/크기/산점도 분포)하기 위한 표준 payload 제공
- payload 예시:
  - `method`: `"kmeans"` 등
  - `n_clusters`: number
  - `clusters`: `[{id, size, avg_similarity, min_similarity, max_similarity, top_similar_files}]`
  - `projection.method`: `"pca"` (2D)
  - `projection.points`: `[{x, y, cluster, id}]` (`cluster`는 클러스터 id, `id`는 파일 키/상대경로)
  - `projection.sampling`: `{cap, n, total, seed, strategy}`

## Drift 아티팩트

### 속성 지표 분포 비교
- type: `drift.attribute_distributions.v1`
- 목적: base vs target 분포 비교를 핵심 시각화로 제공
- payload 예시:
  - `metrics.<metric>.base` (histogram)
  - `metrics.<metric>.target` (histogram)
  - `metrics.<metric>.score` (drift 점수, 숫자 표기)
  - `metrics.<metric>.method` (예: `psi`)

### 임베딩 오버레이 프로젝션
- type: `drift.embedding.projection.2d.v1`
- 목적: base/target 임베딩을 동일 좌표계에서 PCA 2D로 투영
- 샘플링:
  - `n = min(2000, len(base), len(target))`
  - 고정 시드 랜덤 샘플링(재현성 보장)
- payload 예시:
  - `method: "pca"`
  - `points: [{x, y, split}]` (`split`은 `base` 또는 `target`)
  - `sampling` 메타: cap, n, base/target count, seed

## Trainlog (MLflow) 아티팩트

### Run 목록
- type: `trainlog.mlflow.runs.index.v1`
- payload:
  - `[{run_id, run_name, experiment_id, user_id, status, start_time, end_time, artifact_uri, tags}]`

### Params 인덱스
- type: `trainlog.mlflow.params.index.v1`
- payload:
  - `{run_id: {param_key: value}}`

### Metrics 인덱스
- type: `trainlog.mlflow.metrics.index.v1`
- payload:
  - `{run_id: {metric_name: [{timestamp, value, step}]}}`

### Artifacts 인덱스
- type: `trainlog.mlflow.artifacts.index.v1`
- payload:
  - `{run_id: [{path, size_bytes, ext, kind}]}`

### MLflow UI 가이드
- type: `trainlog.mlflow.mlflow_ui.guide.v1`
- payload:
  - `{tracking_dir, command, note}`

### Preview 이미지
- type: `trainlog.mlflow.preview.image.v1`
- payload:
  - `{run_id, path, mime, data}` (data는 base64)

### Drift 집계(분포 비교)
- type: `trainlog.mlflow.drift.aggregate.v1`
- payload:
  - `metrics.<metric>.base/target` (n, mean, median, std, min, max)
  - `metrics.<metric>.delta_mean`, `metrics.<metric>.normalized_delta`
  - `params.changed_ratio`, `params.changed_keys`

### Drift 매칭 페어
- type: `trainlog.mlflow.drift.matched_pairs.v1`
- payload:
  - `[{signature, base_run_id, target_run_id, delta_final_metrics, curve_mse}]`

