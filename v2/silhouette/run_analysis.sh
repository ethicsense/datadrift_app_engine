# 분석 데이터셋 생성. 멀티모달을 켜면 대표 이미지 1장만 임베딩 대상으로 사용.
python -m analytics.pipeline.build_dataset \
    --data-dir data \
    --output-dir output/analytics


## 도커 기반 실행
# docker compose --profile pipeline run --rm analytics-pipeline \
#   sh -lc "pip install -r requirements.txt && \
#   python -m analytics.pipeline.build_dataset \
#     --data-dir data \
#     --output-dir output/analytics \
#     --start-date 2026-02-20 \
#     --end-date 2026-02-26 \
#     --qdrant-url http://qdrant:6333"