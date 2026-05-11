# datadrift_app_engine

## 주소
- https://github.com/keti-datadrift/datadrift_app_engine.git

## 개요
- 데이터 드리프트 관리 기술의 기반 프레임워크입니다.
- 개발 및 유지 관리 기관 : __한국전자기술연구원(KETI)__
- 최종 검토 기관 : 한국전자기술연구원(KETI)

## Acknowledgements (사사)
- 이 연구는 2024년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임 (No. RS-2024-00337489, 분석 모델의 성능저하 극복을 위한 데이터 드리프트 관리 기술 개발)
- This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2024-00337489, Development of data drift management technology to overcome performance degradation of AI analysis models)

## 주요 기능

### 1. 이미지 분석 (Image Analysis)
- 이미지 메타데이터 추출 (크기, 형식, 해상도 등)
- 이미지 품질 분석 (노이즈 레벨, 선명도)
- 다양한 형식 지원 (JPG, PNG, JPEG 등)

### 2. 임베딩 기반 분석 (Embedding-based Analysis)
- CLIP 모델을 사용한 이미지 임베딩 추출
- 임베딩 기반 클러스터링 분석
- 유사도 계산 및 유사 이미지 검색

### 3. 데이터 드리프트 분석 (Data Drift Analysis)
- 데이터셋 간 비교 분석
- MMD (Maximum Mean Discrepancy) 계산
- 시각화 및 보고서 생성

### 4. 웹 애플리케이션 (Web Application)
- Flask 기반 웹 인터페이스
- 실시간 데이터 분석 및 시각화
- FiftyOne 통합

## 모듈 구조

```
datadrift_app_engine/
├── main.py                          # 메인 CLI 인터페이스
├── data_utils/                      # 데이터 분석 모듈
│   ├── __init__.py
│   ├── embedding_manager.py        # 임베딩 관리 클래스
│   └── attribute_analyzer.py       # 속성 분석 클래스
├── cache_utils/                     # 캐시 관리 모듈
│   ├── __init__.py
│   └── cache_manager.py
├── report_generator/                # 보고서 생성 모듈
│   ├── __init__.py
│   ├── create_report.py
│   └── report_layout.py
├── flask_webapp/                    # 웹 애플리케이션
│   ├── app.py
│   └── utils.py
└── examples/                        # 사용 예시
    └── data_analysis_example.py
```

## 데이터 유틸리티 모듈

### AttributeAnalyzer 클래스

이미지 속성 분석을 관리하는 클래스입니다.

```python
from data_utils import AttributeAnalyzer

# 초기화
analyzer = AttributeAnalyzer()

# 단일 이미지 분석
attributes = analyzer.analyze_image_attributes('path/to/image.jpg')

# 디렉토리 분석
results = analyzer.analyze_directory('path/to/images', ['jpg', 'png'])
summary = analyzer.get_summary_statistics(results)
```

### EmbeddingManager 클래스

임베딩 추출 및 분석을 관리하는 핵심 클래스입니다.

```python
from data_utils import EmbeddingManager

# 초기화
manager = EmbeddingManager(device='cuda')  # GPU 사용
manager.load_model("ViT-B/16")

# 단일 파일 임베딩 추출
result = manager.extract_embedding('path/to/image.jpg')

# 클러스터링 분석 (캐시에서 임베딩 데이터 로드 후)
clustering_result = manager.perform_clustering(
    embeddings_data, file_names, file_paths,
    n_clusters=5, method='kmeans'
)

# Centroid 유사도 정보 접근
centroids = clustering_result['centroids']
centroid_similarities = clustering_result['centroid_similarities']

# 클러스터별 상세 정보
for cluster_id, stats in clustering_result['cluster_stats'].items():
    print(f"Cluster {cluster_id}:")
    print(f"  Size: {stats['size']}")
    print(f"  Average similarity: {stats['avg_similarity']:.4f}")
    print(f"  Top similar files: {stats['top_similar_files']}")
```

### 편의 함수

```python
from data_utils import (
    run_attribute_analysis, 
    run_embedding_extraction, 
    run_clustering_analysis
)

# 속성 분석
attribute_results = run_attribute_analysis(
    directories=['path/to/images'],
    formats=['jpg', 'png']
)

# 임베딩 추출
extraction_results = run_embedding_extraction(
    directories=['path/to/images'],
    formats=['jpg', 'png'],
    device='cuda'
)

# 클러스터링 분석
clustering_results = run_clustering_analysis(
    directories=['path/to/images'],
    formats=['jpg', 'png'],
    method='kmeans'
)
```

### 지원하는 클러스터링 방법

1. **K-means**: 가장 일반적인 클러스터링 방법
2. **DBSCAN**: 밀도 기반 클러스터링
3. **Hierarchical**: 계층적 클러스터링

## DAE Training Pipeline Tutorial
데이터 샘플링 및 모델 학습 파이프라인 앱 설치 및 구동

### Installation
```
# Clone repo
git clone $GITHUB_URL

# Create conda env
conda create -n $ENV_NAME python==3.12.5
conda activate $ENV_NAME

# Install packages
cd $GITHUB_DIR
pip install -r requirements.txt

# Install CLIP openai
pip install git+https://github.com/openai/CLIP.git

# Run pipeline App
bash run.sh
```

### CLI 사용법

#### 1. 이미지 분석
```bash
# 기본 분석 (속성 분석 + 임베딩 추출)
python main.py analysis /path/to/image/directory

# 특정 형식만 분석
python main.py analysis /path/to/directory --format jpg png

# 루트 디렉토리의 모든 하위 디렉토리 분석
python main.py analysis -r /path/to/root/directory
```

#### 2. 클러스터링 분석
```bash
# 기본 클러스터링 (K-means, 자동 클러스터 수 결정)
python main.py clustering /path/to/directory

# 특정 클러스터 수 지정
python main.py clustering /path/to/directory --n-clusters 5

# 다른 클러스터링 방법 사용
python main.py clustering /path/to/directory --method dbscan
python main.py clustering /path/to/directory --method hierarchical
```

#### 3. 보고서 생성
```bash
# HTML 보고서 생성
python main.py report /path/to/directory --mode html
```

#### 4. 웹 애플리케이션 실행
```bash
# 기본 포트(5555)로 실행
python main.py app

# 특정 포트로 실행
python main.py app --port 8080

# 디버그 모드로 실행
python main.py app --debug
```

### Python API 사용법

```python
import sys
sys.path.append('/path/to/datadrift_app_engine')

from data_utils import AttributeAnalyzer, EmbeddingManager

# 속성 분석
analyzer = AttributeAnalyzer()
attribute_results = analyzer.analyze_directory('/path/to/images', ['jpg', 'png'])

# 임베딩 매니저 초기화
manager = EmbeddingManager(device='cuda')
manager.load_model("ViT-B/16")

# 임베딩 추출
results = manager.extract_embeddings(
    directories=['/path/to/images'],
    formats=['jpg', 'png']
)

# 클러스터링 분석
clustering_results = manager.perform_clustering(
    directories=['/path/to/images'],
    formats=['jpg', 'png'],
    n_clusters=3,
    method='kmeans'
)

# 유사도 계산
embeddings = np.array([...])  # 임베딩 배열
similarity_matrix = manager.calculate_similarity_matrix(embeddings)
similar_pairs = manager.find_similar_pairs(similarity_matrix, threshold=0.8)
```

## 캐시 시스템

모든 분석 결과는 효율적인 캐시 시스템을 통해 저장되며, 개별 파일 단위로 변경을 감지하여 필요한 파일만 재분석합니다.

### 캐시 시스템 특징

- **파일 해시 기반 변경 감지**: MD5 해시를 사용하여 파일 변경 여부를 정확히 감지
- **개별 파일 단위 처리**: 전체 디렉토리를 다시 분석하지 않고 변경된 파일만 처리
- **중복 분석 방지**: 이미 분석된 파일은 건너뛰어 처리 시간 단축
- **증분 업데이트**: 새로운 파일이나 변경된 파일만 추가로 분석

### 캐시 위치 및 키

- **캐시 위치**: 각 디렉토리별로 `.cache` 폴더에 저장
- **캐시 키**:
  - `image_analysis`: 이미지 속성 분석 결과
  - `image_drift_content`: 임베딩 데이터
  - `clustering_analysis`: 클러스터링 결과 (임베딩 해시 기반 유효성 검사)

### 캐시 동작 방식

1. **속성 분석**: 파일별 해시를 확인하여 변경된 파일만 분석
2. **임베딩 추출**: 파일별 해시를 확인하여 변경된 파일만 임베딩 추출
3. **클러스터링**: 임베딩 데이터의 전체 해시를 확인하여 변경 시에만 재계산

### 성능 최적화

- **첫 실행**: 모든 파일을 분석하여 캐시 생성
- **재실행**: 변경된 파일만 분석하여 처리 시간 대폭 단축
- **클러스터링**: 임베딩 데이터가 변경되지 않으면 기존 결과 재사용

## 시각화

클러스터링 결과는 `create_report` 모듈에서 시각화되어 HTML 보고서에 포함됩니다:

- 클러스터 분포 시각화 (PCA 2D)
- 클러스터 크기 분포 차트
- 클러스터 중심점 표시 (K-means의 경우)

## 요구사항

- Python 3.8+
- PyTorch
- CLIP (OpenAI)
- scikit-learn
- matplotlib
- seaborn
- PIL (Pillow)
- numpy
- pandas

## 라이선스

이 프로젝트는 한국전자기술연구원(KETI)에서 개발한 데이터 드리프트 관리 기술 프레임워크입니다.
