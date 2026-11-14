"""Conservative, pre-dispatch request-size estimates for bounded LLM campaigns.

The estimate is deliberately an upper-oriented proxy, not tokenizer truth or
provider usage. It uses the complete serialized request, one token per encoded
byte, and additional framing headroom. Provider-reported usage remains the
source of truth after completion.
"""

from __future__ import annotations

import json
import math


class RequestBudgetError(ValueError):
    """The request could not be safely serialized or estimated."""


def _json_value(value):
    """Convert known request values and omit OpenAI's `NOT_GIVEN` sentinels."""
    if type(value).__name__ == "NotGiven":
        return _OMIT
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RequestBudgetError("request contains a non-string object key")
            item = _json_value(item)
            if item is not _OMIT:
                result[key] = item
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            item = _json_value(item)
            if item is not _OMIT:
                result.append(item)
        return result
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json"))
    raise RequestBudgetError("request contains an unsupported value")


_OMIT = object()


def serialized_request(params: dict) -> bytes:
    """Return bounded deterministic JSON bytes for the actual provider params."""
    if not isinstance(params, dict):
        raise RequestBudgetError("request parameters are unavailable")
    try:
        value = _json_value(params)
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise RequestBudgetError("request parameters cannot be safely serialized") from None
    if not encoded or len(encoded) > 4_000_000:
        raise RequestBudgetError("serialized request is outside estimation limits")
    return encoded


def conservative_input_token_estimate(params: dict) -> tuple[int, int]:
    """Estimate from the full serialized request plus 25% and 128 tokens headroom.

    A byte-level token upper bound is conservative for the GPT byte-BPE family;
    extra headroom covers chat framing and adapter/provider wrappers. We do not
    claim this equals Groq's tokenizer accounting.
    """
    size = len(serialized_request(params))
    estimate = size + max(128, math.ceil(size * 0.25))
    return estimate, size
