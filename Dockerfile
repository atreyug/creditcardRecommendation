# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Metadata
LABEL maintainer="OpenEnv Credit Card Env"
LABEL description="Credit Card Recommendation System — OpenEnv + FastAPI"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY credit_card_env/ ./credit_card_env/
COPY app.py .
COPY inference.py .
COPY openenv.yaml .

# Copy data directory (optional — will fallback to synthetic data if absent)
COPY data/ ./data/

# Expose port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_NAME=gpt-4o-mini
ENV API_BASE_URL=https://api.openai.com/v1

# Default command: start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
