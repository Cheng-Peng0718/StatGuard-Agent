from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.dataset_intelligence.schemas import (
    CapabilityMap,
    DatasetProfileV2,
    DatasetSummary,
)


class DatasetContext(BaseModel):
    data_version_id: str
    dataset_name: str = "uploaded_dataset"
    data_path: Optional[str] = None

    dataset_profile_v2: DatasetProfileV2
    dataset_summary: DatasetSummary
    capability_map: CapabilityMap

    state_dataset_profile: Optional[Dict[str, Any]] = None
    created_by: str = "context_refresh"
    source: Literal["upload", "build_context", "mutation_refresh", "unknown"] = "unknown"
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
