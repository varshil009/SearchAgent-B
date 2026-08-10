"""Compact structural descriptions of tool responses for the decision node."""

from typing import Any


def describe_tool_result(value: Any, depth: int = 0, max_depth: int = 3) -> str:
    """Describe a result without copying its potentially large contents."""
    if depth >= max_depth:
        return type(value).__name__

    if isinstance(value, dict):
        if not value:
            return "dict (empty)"
        fields = ", ".join(
            f"{key}: {describe_tool_result(item, depth + 1, max_depth)}"
            for key, item in list(value.items())[:20]
        )
        suffix = ", ..." if len(value) > 20 else ""
        return f"dict {{{fields}{suffix}}}"

    if isinstance(value, list):
        if not value:
            return "list (empty)"
        return f"list[{len(value)}] of {describe_tool_result(value[0], depth + 1, max_depth)}"

    if isinstance(value, tuple):
        return f"tuple[{len(value)}]"

    return type(value).__name__
