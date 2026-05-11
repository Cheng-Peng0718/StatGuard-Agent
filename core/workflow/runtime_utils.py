from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def sanitize_results(obj: Any):
    """
    Recursively convert numpy scalars/arrays to native Python for serialization.
    """
    if isinstance(obj, dict):
        return {k: sanitize_results(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_results(v) for v in obj]

    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)

    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    return str(obj)


def get_action_hash(tool_name: str, arguments: dict):
    """
    Stable MD5 fingerprint from tool name + canonical JSON arguments.
    """
    if not arguments:
        arguments = {}

    arg_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(f"{tool_name}_{arg_str}".encode("utf-8")).hexdigest()