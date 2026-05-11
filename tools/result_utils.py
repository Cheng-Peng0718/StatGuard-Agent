from typing import Any, Dict

# English prefixes plus legacy tool output (Unicode escapes avoid CJK in source)
_LEGACY_ERR = ("\u9519\u8bef", "\u5931\u8d25", "\u5d29\u6e83")

ERROR_PREFIXES = (
    "error",
    "failed",
    "failure",
    "exception",
) + _LEGACY_ERR


def _looks_like_error_string(s: str) -> bool:
    text = s.strip().lower()
    return any(text.startswith(p.lower()) for p in ERROR_PREFIXES) or any(
        m in s for m in _LEGACY_ERR
    )


def normalize_tool_payload(raw: Any) -> Dict[str, Any]:
    """
    Normalize any tool return value into:
    {
        status, success, error_code, message, recoverable,
        payload, artifacts
    }
    """
    if isinstance(raw, dict) and "status" in raw:
        status = raw.get("status", "ok")
        if status not in {"ok", "warning", "blocked", "failed"}:
            status = "warning"

        return {
            "status": status,
            "success": status == "ok",
            "error_code": raw.get("error_code"),
            "message": raw.get("message", ""),
            "recoverable": raw.get("recoverable", False),
            "payload": raw.get("details", raw),
            "artifacts": raw.get("artifacts", []),
        }

    if isinstance(raw, str) and _looks_like_error_string(raw):
        return {
            "status": "failed",
            "success": False,
            "error_code": "tool_string_error",
            "message": raw,
            "recoverable": True,
            "payload": {},
            "artifacts": [],
        }

    return {
        "status": "ok",
        "success": True,
        "error_code": None,
        "message": str(raw) if not isinstance(raw, str) else raw,
        "recoverable": False,
        "payload": raw if isinstance(raw, (dict, list)) else {"value": raw},
        "artifacts": [],
    }
