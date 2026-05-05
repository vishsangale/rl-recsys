from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "interactions": {"user_id", "item_id", "timestamp"},
    "sessions": {"session_id", "user_id", "item_id", "timestamp"},
    "rl_sessions": {"session_id", "step", "user_state", "slate", "item_features", "clicks"},
    "slates": {"request_id", "user_id"},
    "social_edges": {"user_a", "user_b"},
    "items": {"item_id"},
}


def validate_parquet_schema(path: Path, schema_type: str) -> None:
    if schema_type not in REQUIRED_COLUMNS:
        raise ValueError(
            f"Unknown schema type {schema_type!r}. Known: {sorted(REQUIRED_COLUMNS)}"
        )
    schema = pq.read_schema(path)
    missing = REQUIRED_COLUMNS[schema_type] - set(schema.names)
    if missing:
        raise ValueError(
            f"{path.name} missing columns for schema {schema_type!r}: {missing}"
        )
