FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt /app/requirements/base.txt
RUN pip install --no-cache-dir -r /app/requirements/base.txt

ENV APP_DIR=/app
ENV ARTIFACTS_DIR=/app/artifacts
ENV PYTHONUNBUFFERED=1

COPY config/ /app/config/
COPY pipelines/ /app/pipelines/
COPY feature_store/ /app/feature_store/

CMD ["sh", "-c", "python pipelines/feature_engineering.py && python feature_store/materialize_features.py"]
