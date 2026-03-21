# ── Build stage ──────────────────────────────────────────────────────────────
# Installs all dependencies into a clean layer so the final image doesn't
# carry pip caches or build tools.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps (needed for some wheels, e.g. lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed to resolve dependencies first (better layer caching)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install core package + serving extras into a prefix we can copy cleanly
RUN pip install --upgrade pip \
 && pip install --prefix=/install -e ".[serve]"


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Work Step Time Prediction API" \
      org.opencontainers.image.description="FastAPI service predicting 14 automotive repair work steps and their durations." \
      org.opencontainers.image.source="https://github.com/danielHelmke377/work-step-time-prediction"

# libgomp is needed at runtime by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy source package so editable install resolves correctly
COPY src/ ./src/

# Copy the API application layer
COPY app/ ./app/

# ── Model artifact handling ────────────────────────────────────────────────────
# models/ is NOT baked into the image. It must be mounted at runtime:
#
#   docker run -v $(pwd)/models:/app/models ...
#
# The service starts in degraded mode if models/ is empty or missing; this is
# intentional — GET /health returns 503 until a trained artifact is mounted.
# See README.md → Running with Docker for the full workflow.
#
# To pre-bake artifacts into the image (e.g. for a demo snapshot), uncomment:
# COPY models/ ./models/

# Create models directory so the path resolution in predictor.py works
RUN mkdir -p models

# ── Runtime configuration ─────────────────────────────────────────────────────
# Override any of these at runtime with -e / --env-file:
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The port Uvicorn listens on (override with -e PORT=xxxx)
ENV PORT=8000

EXPOSE $PORT

# Health check so Docker / Compose knows when the service is ready
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

# Start with a single Uvicorn worker (scale out with --workers or a proxy)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
