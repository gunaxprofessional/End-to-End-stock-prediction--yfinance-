from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float32, String, Int64

# Define an entity for the stock ticker
ticker = Entity(
    name="ticker", 
    join_keys=["ticker"],
    value_type=ValueType.STRING, 
    description="Stock Ticker Symbol"
)

# Define the source of the data
stock_features_source = FileSource(
    name="stock_features_source",
    path="../data/processed/stock_features.parquet",
    timestamp_field="Date",
    created_timestamp_column="created_timestamp",
)

# Define the feature view
stock_features_view = FeatureView(
    name="stock_features",
    entities=[ticker],
ttl=timedelta(days=3650), # Long TTL since we have daily data
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