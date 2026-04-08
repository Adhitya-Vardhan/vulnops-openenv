# --- Dockerfile for VulnOps Benchmark ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy everything needed to build the module
COPY pyproject.toml uv.lock README.md ./
COPY server/ ./server/

# Install dependencies (without dev)
RUN uv sync --frozen --no-dev --no-editable

# --- Runtime Stage ---
FROM python:3.11-slim

WORKDIR /app

# Copy remaining files
COPY models.py inference.py ./
COPY server/ ./server/
COPY data/ ./data/

# Copy virtualenv from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user and set permissions for Hugging Face Spaces
RUN useradd -m -u 1000 user && \
    mkdir -p /tmp && \
    chown -R user:user /app /tmp

USER 1000

# Expose port (HF Spaces defaults to 7860)
EXPOSE 7860

# Default command: Start the environment server
# Use uvicorn to serve the VulnOps FastAPI application on port 7860
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
