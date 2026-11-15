# ==============================================================================
# Customer Intelligence Engine (CIE) - Multi-Stage Production Dockerfile
# Base: Python 3.12-slim
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build Dependencies
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -e ".[api,tracking]"

# ------------------------------------------------------------------------------
# Stage 2: Minimal Secure Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Security: Create non-root system user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -M appuser

WORKDIR /app

# Install pre-built wheels from builder stage
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application assets, configurations, and models
COPY config/ /app/config/
COPY models/ /app/models/
COPY src/ /app/src/
# Copy precomputed customer backtesting store
COPY data/processed/clv/test_predictions_2011-09-10.parquet /app/data/processed/clv/
COPY data/processed/features/features_test_2011-09-10.parquet /app/data/processed/features/

ENV PYTHONPATH="/app:/app/src:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CIE_PORT=8000 \
    CIE_HOST=0.0.0.0

# Set ownership to non-root user
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Native Python healthcheck (zero curl/wget dependency)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health', timeout=4)" || exit 1

# Production ASGI server command
CMD ["uvicorn", "cie.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
