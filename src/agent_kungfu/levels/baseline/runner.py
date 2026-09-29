from __future__ import annotations

import polars as pl

from ...config import Settings
from ...database import SakilaDB
from ..common import load_baseline_assets, resolved_settings


def run_baseline(settings: Settings | None = None) -> pl.DataFrame:
    """Execute the YAML-located reference SQL through Polars/ConnectorX."""

    resolved = resolved_settings(settings)
    assets = load_baseline_assets(resolved)
    return SakilaDB(resolved).query(assets.sql)
