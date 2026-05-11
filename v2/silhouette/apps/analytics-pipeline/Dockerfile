FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY analytics /app/analytics
COPY scripts /app/scripts

CMD ["python", "-m", "analytics.pipeline.build_dataset", "--data-dir", "data", "--output-dir", "output/analytics", "--qdrant-url", "http://qdrant:6333"]
