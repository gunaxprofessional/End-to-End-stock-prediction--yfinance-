from datetime import datetime, timedelta, timezone
from feast import FeatureStore
import os

repo_path = "feature_repo"

if not os.path.exists(repo_path):
    raise Exception(f"Feature repo path '{repo_path}' does not exist.")

store = FeatureStore(repo_path=repo_path)

print("Materializing features...")
# Materialize from 10 years ago to now
store.materialize(
    end_date=datetime.now(timezone.utc),
    start_date=datetime.now(timezone.utc) - timedelta(days=3650)
)
print("Materialization complete.")
