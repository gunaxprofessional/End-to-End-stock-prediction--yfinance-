"""Materialize features from parquet to Feast online store."""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta, timezone
from feast import FeatureStore

from config.global_config import FEAST_REPO_PATH, FEATURE_PARQUET_PATH
from feature_store.feature_repo.definitions import ticker, stock_features_view


def materialize_features():
    """Materialize features to online store."""
    if not FEATURE_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {FEATURE_PARQUET_PATH}")

    store = FeatureStore(repo_path=str(FEAST_REPO_PATH))

    print("Applying feature definitions...")
    store.apply([ticker, stock_features_view])

    print("Materializing features to online store...")
    store.materialize(
        start_date=datetime.now(timezone.utc) - timedelta(days=365),
        end_date=datetime.now(timezone.utc)
    )

    print("Materialization complete.")


if __name__ == "__main__":
    materialize_features()
