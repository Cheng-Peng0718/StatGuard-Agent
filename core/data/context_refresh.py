from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from core.dataset_intelligence.capability_map import build_capability_map
from core.dataset_intelligence.profiler import profile_dataframe, summarize_profile
from core.dataset_intelligence.schemas import (
    CapabilityMap,
    DatasetProfileV2,
    DatasetSummary,
)
from core.domain.dataset_context import DatasetContext


@dataclass
class DatasetContextRefresh:
    data_version_id: str
    dataset_profile_v2: DatasetProfileV2
    dataset_summary: DatasetSummary
    capability_map: CapabilityMap
    data_path: Optional[str] = None

    def dataset_summary_dict(self) -> Dict[str, Any]:
        summary = self.dataset_summary.model_dump()
        missingness = summary.get("missingness_summary")

        if isinstance(missingness, dict):
            columns = missingness.get("columns")
            if columns is not None and "missing_by_column" not in missingness:
                missingness["missing_by_column"] = columns

        return summary

    def to_state_updates(
        self,
        *,
        include_dataset_context: bool = False,
        dataset_name: str = "uploaded_dataset",
        state_dataset_profile: Optional[Dict[str, Any]] = None,
        source: str = "unknown",
    ) -> Dict[str, Any]:
        updates = {
            "dataset_profile_v2": self.dataset_profile_v2.model_dump(),
            "dataset_summary": self.dataset_summary_dict(),
            "capability_map": self.capability_map.model_dump(),
        }

        if include_dataset_context:
            updates["dataset_context"] = self.to_domain_context(
                dataset_name=dataset_name,
                state_dataset_profile=state_dataset_profile,
                source=source,
            ).model_dump()

        return updates

    def to_domain_context(
        self,
        *,
        dataset_name: str = "uploaded_dataset",
        state_dataset_profile: Optional[Dict[str, Any]] = None,
        source: str = "unknown",
    ) -> DatasetContext:
        return DatasetContext(
            data_version_id=self.data_version_id,
            dataset_name=dataset_name,
            data_path=self.data_path,
            dataset_profile_v2=self.dataset_profile_v2,
            dataset_summary=self.dataset_summary,
            capability_map=self.capability_map,
            state_dataset_profile=state_dataset_profile,
            source=source,
        )


def load_dataframe_for_dataset_context(path: str) -> pd.DataFrame:
    lower_path = str(path).lower()

    if lower_path.endswith(".parquet"):
        return pd.read_parquet(path)

    if lower_path.endswith(".csv"):
        return pd.read_csv(path)

    if lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
        return pd.read_excel(path)

    raise ValueError(f"Unsupported active data file type for profiling: {path}")


def refresh_dataset_context_from_df(
    df: pd.DataFrame,
    *,
    dataset_name: str = "uploaded_dataset",
    data_version_id: str = "unknown",
    data_path: Optional[str] = None,
) -> DatasetContextRefresh:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("refresh_dataset_context_from_df requires a pandas DataFrame.")

    dataset_profile_v2 = profile_dataframe(
        df,
        dataset_name=dataset_name,
        data_version_id=data_version_id,
    )
    dataset_summary = summarize_profile(dataset_profile_v2)
    capability_map = build_capability_map(dataset_profile_v2)

    return DatasetContextRefresh(
        data_version_id=data_version_id,
        dataset_profile_v2=dataset_profile_v2,
        dataset_summary=dataset_summary,
        capability_map=capability_map,
        data_path=data_path,
    )


def refresh_dataset_context_from_path(
    path: str,
    *,
    dataset_name: str = "uploaded_dataset",
    data_version_id: str = "unknown",
) -> DatasetContextRefresh:
    df = load_dataframe_for_dataset_context(path)

    return refresh_dataset_context_from_df(
        df,
        dataset_name=dataset_name,
        data_version_id=data_version_id,
        data_path=path,
    )
