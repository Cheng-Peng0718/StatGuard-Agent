from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ColumnProfileV2(BaseModel):
    name: str
    raw_dtype: str

    semantic_type: str
    measurement_scale: str

    n_missing: int
    missing_rate: float

    n_unique: int
    unique_rate: float

    examples: List[Any] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    numeric_summary: Optional[Dict[str, Any]] = None
    categorical_summary: Optional[Dict[str, Any]] = None


class DatasetProfileV2(BaseModel):
    dataset_name: str = "uploaded_dataset"
    data_version_id: str = "unknown"

    n_rows: int
    n_cols: int

    columns: Dict[str, ColumnProfileV2]

    warnings: List[str] = Field(default_factory=list)


class DatasetSummary(BaseModel):
    dataset_name: str = "uploaded_dataset"
    data_version_id: str = "unknown"

    n_rows: int
    n_cols: int

    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    binary_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    text_columns: List[str] = Field(default_factory=list)
    id_like_columns: List[str] = Field(default_factory=list)

    missingness_summary: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class AnalysisCapability(BaseModel):
    tool_name: str
    display_name: str
    method_family: str = "general"

    status: str
    reason: str

    candidate_variables: Dict[str, List[str]] = Field(default_factory=dict)
    required_user_choices: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggested_alternatives: List[str] = Field(default_factory=list)

    requires_confirmation: bool = False
    mutates_data: bool = False


class CapabilityMap(BaseModel):
    data_version_id: str
    capabilities: List[AnalysisCapability] = Field(default_factory=list)

    warnings: List[str] = Field(default_factory=list)