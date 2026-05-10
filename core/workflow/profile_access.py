from __future__ import annotations

from typing import Any, Dict, Optional

from core.dataset_intelligence.schemas import DatasetProfileV2


def get_context_profile(state: Dict[str, Any]) -> Any:
    """
    Return the workflow context profile.

    This is the profile produced by core.context_builder.generate_profile()
    and stored in graph state as ``dataset_profile``. It is still used by
    build_context(), the supervisor prompt path, and the execution verifier.

    Do not confuse this with dataset_profile_v2.
    """
    return state.get("dataset_profile")


def require_context_profile(state: Dict[str, Any]) -> Any:
    profile = get_context_profile(state)

    if profile is None:
        raise KeyError("dataset_profile")

    return profile


def get_dataset_profile_v2(state: Dict[str, Any]) -> Optional[DatasetProfileV2]:
    """
    Return the dataset intelligence profile, if available.

    This profile is used by LLM planning, plan verification, capability maps,
    and semantic-type-aware logic.
    """
    profile = state.get("dataset_profile_v2")

    if profile is None:
        return None

    if isinstance(profile, DatasetProfileV2):
        return profile

    return DatasetProfileV2.model_validate(profile)


def require_dataset_profile_v2(state: Dict[str, Any]) -> DatasetProfileV2:
    profile = get_dataset_profile_v2(state)

    if profile is None:
        raise KeyError("dataset_profile_v2")

    return profile