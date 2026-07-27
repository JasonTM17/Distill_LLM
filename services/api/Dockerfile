# Build stage: compile llama-cpp-python (CPU) and collect wheels
FROM python:3.14-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# CPU-only wheel index first (prebuilt llama-cpp-python), source build as fallback
RUN pip wheel --wheel-dir /wheels \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    -r requirements.txt

# Runtime stage: slim, non-root, no compilers
FROM python:3.14-slim

ARG REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/JasonTM17/distill-gpt55" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${REVISION}"

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 65532 --user-group --no-create-home app

WORKDIR /srv
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl \
    && rm -rf /wheels

COPY app/ ./app/

ENV MODEL_PATH=/models/model.gguf \
    PYTHONUNBUFFERED=1

USER 65532:65532
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
