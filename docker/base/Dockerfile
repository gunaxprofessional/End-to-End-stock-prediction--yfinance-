FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install base requirements
COPY requirements/base.txt /app/requirements/base.txt
RUN pip install --no-cache-dir -r /app/requirements/base.txt

# Set environment variables
ENV APP_DIR=/app
ENV ARTIFACTS_DIR=/app/artifacts
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY config/ /app/config/
COPY pipelines/ /app/pipelines/
COPY feature_store/ /app/feature_store/
COPY models/ /app/models/
