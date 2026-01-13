from datetime import datetime, timedelta, timezone
from feast import FeatureStore
import os
from storage import MinioArtifactStore
from config import (
    FEATURE_PARQUET_KEY, REGISTRY_KEY, ONLINE_STORE_KEY, 
    PROCESSED_DATA_KEY
)

repo_path = "feature_repo"

if not os.path.exists(repo_path):
    raise Exception(f"Feature repo path '{repo_path}' does not exist.")

store = FeatureStore(repo_path=repo_path)
minio_store = MinioArtifactStore()

print("Downloading offline store data...")
minio_store.download_file(FEATURE_PARQUET_KEY, FEATURE_PARQUET_KEY)

print("Updating Feast registry...")
store.apply()

print("Materializing features to online store (last 30 days)...")
store.materialize(
    end_date=datetime.now(timezone.utc),
    start_date=datetime.now(timezone.utc) - timedelta(days=30)
)
print("Materialization complete.")

print("Uploading Feast stores to MinIO...")
minio_store.upload_file("data/registry.db", REGISTRY_KEY)
minio_store.upload_file("data/online_store.db", ONLINE_STORE_KEY)
