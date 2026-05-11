FROM python:3.11-slim

WORKDIR /app

# system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libcairo2 \
    libpango-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# NLTK data location (plugins may rely on punkt/stopwords)
ENV NLTK_DATA=/usr/local/share/nltk_data

# Optional: ddoc modality plugins install (comma-separated; folder suffix list)
# - 예: text,timeseries,vision-image,vision-video,audio-wave,audio-midi
# - 기본값: 텍스트/시계열 + 오디오(wave/midi). vision-video는 필요 시만 켭니다(리소스 큼).
ARG DDOC_PLUGINS="text,timeseries,audio-wave,audio-midi"

# copy repo parts needed for API + local packages install
# NOTE: build context is repo root (docker-compose.yml 기준)
COPY apps/api /repo/apps/api
COPY packages /repo/packages

WORKDIR /repo/apps/api

# install API + local packages
# NOTE: 로컬 패키지들은 PyPI에 배포되지 않았으므로, 먼저 editable로 설치해
# API 의존성 해석 시 "이미 설치됨"으로 만족시키도록 합니다.
RUN pip install --no-cache-dir -e /repo/packages/driftstudio_spec \
    && pip install --no-cache-dir -e /repo/packages/driftstudio_runtime \
    && pip install --no-cache-dir -e /repo/packages/driftstudio_reports \
    && pip install --no-cache-dir -e /repo/packages/ddoc \
    && if [ -n "$DDOC_PLUGINS" ]; then \
         for p in $(echo "$DDOC_PLUGINS" | tr ',' ' '); do \
           echo "Installing plugin: ddoc-plugin-$p"; \
           pip install --no-cache-dir -e "/repo/packages/ddoc/plugins/ddoc-plugin-$p" || echo "WARNING: Failed to install ddoc-plugin-$p"; \
         done; \
       fi \
    && pip install --no-cache-dir -e .

# Debug: 플러그인 로드 상태 확인
RUN python - <<'PY' || true
import sys
sys.path.insert(0, '/repo/packages/ddoc')
sys.path.insert(0, '/repo/packages/driftstudio_spec/packages')
sys.path.insert(0, '/repo/packages/driftstudio_runtime/packages')

try:
    from ddoc.core.plugins import get_plugin_manager
    pmgr = get_plugin_manager()
    pmgr.load_entrypoints()
    
    print("=== Loaded plugins ===")
    for plugin in pmgr.get_plugins():
        name = pmgr.pm.get_name(plugin)
        print(f"  {name}: {plugin.__class__.__name__}")
    
    print("\n=== EDA hook implementations ===")
    hook = pmgr.pm.hook.eda_run
    for impl in hook.get_hookimpls():
        plugin_obj = impl.plugin
        plugin_name = pmgr.pm.get_name(plugin_obj) if plugin_obj else None
        print(f"  {plugin_name}: {impl.function.__name__}")
except Exception as e:
    print(f"Error checking plugins: {e}")
    import traceback
    traceback.print_exc()
PY

# Best-effort: preload common NLTK resources used by text plugin.
# (Build 환경에서 네트워크가 막혀있을 수 있으므로 실패해도 빌드는 진행합니다.)
RUN python - <<'PY' || true
import os
try:
    import nltk
except Exception as e:
    print("NLTK not available, skipping resource download:", e)
    raise SystemExit(0)

download_dir = os.environ.get("NLTK_DATA") or "/usr/local/share/nltk_data"
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.download(pkg, download_dir=download_dir, quiet=True)
        print(f"Downloaded NLTK resource: {pkg} -> {download_dir}")
    except Exception as e:
        print(f"Failed to download NLTK resource {pkg}: {e}")
PY

EXPOSE 18765

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18765"]


