from pathlib import Path
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32

ticker = Entity(
    name="ticker",
    join_keys=["ticker"],
    value_type=ValueType.STRING,
    description="Stock Ticker Symbol"
)

# Fixed local path for parquet file
_parquet_path = str(Path(__file__).parent.parent.parent / "artifacts" / "data" / "processed" / "stock_features.parquet")

stock_features_source = FileSource(
    name="stock_features_source",
    path=_parquet_path,
    timestamp_field="Date",
    created_timestamp_column="created_timestamp",
)

stock_features_view = FeatureView(
    name="stock_features",
    entities=[ticker],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="Close", dtype=Float32),
        Field(name="High", dtype=Float32),
        Field(name="Low", dtype=Float32),
        Field(name="Open", dtype=Float32),
        Field(name="Volume", dtype=Float32),
        Field(name="Returns", dtype=Float32),
        Field(name="High_Low_Pct", dtype=Float32),
        Field(name="Close_Open_Pct", dtype=Float32),
        Field(name="MA_3", dtype=Float32),
        Field(name="MA_6", dtype=Float32),
        Field(name="MA_8", dtype=Float32),
        Field(name="Volatility_3", dtype=Float32),
        Field(name="Volatility_6", dtype=Float32),
        Field(name="Volume_MA_3", dtype=Float32),
        Field(name="Volume_Ratio", dtype=Float32)
    ],
    online=True,
    source=stock_features_source,
    tags={"team": "alpha-quant"},
)
