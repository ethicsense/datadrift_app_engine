FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY analytics /app/analytics

ENV SILHOUETTE_OUTPUT_DIR=/app/output/analytics
ENV SILHOUETTE_DATASETS_ROOT=/app/output

CMD ["uvicorn", "analytics.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
