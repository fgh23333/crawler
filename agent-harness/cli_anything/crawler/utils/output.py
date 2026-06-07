"""Output formatting utilities for JSON and human-readable output."""

import json
import sys
from typing import Any, Optional


def format_output(data: dict, json_mode: bool = False) -> str:
    """Format output data as JSON or human-readable text.

    Args:
        data: The data dict to format
        json_mode: If True, output raw JSON

    Returns:
        Formatted string
    """
    if json_mode:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)

    status = data.get("status", "unknown")
    if status == "error":
        msg = data.get("message", "Unknown error")
        if "stdout" in data:
            msg += f"\nstdout: {data['stdout']}"
        if "stderr" in data:
            msg += f"\nstderr: {data['stderr']}"
        return f"Error: {msg}"

    # Human-readable formatting based on data structure
    lines = []
    _format_dict(data, lines, indent=0)
    return "\n".join(lines)


def _format_dict(data: Any, lines: list, indent: int = 0):
    """Recursively format a dict for human-readable output."""
    prefix = "  " * indent

    if isinstance(data, dict):
        # Skip internal keys
        skip_keys = {"status"}
        for key, value in data.items():
            if key in skip_keys:
                continue
            if isinstance(value, dict) and "status" in value:
                lines.append(f"{prefix}{key}: {value.get('status', '')}")
                if value.get("error"):
                    lines.append(f"{prefix}  error: {value['error']}")
            elif isinstance(value, (dict, list)) and value:
                lines.append(f"{prefix}{key}:")
                _format_dict(value, lines, indent + 1)
            elif value is None:
                lines.append(f"{prefix}{key}: null")
            else:
                lines.append(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _format_dict(item, lines, indent)
            elif isinstance(item, list):
                _format_dict(item, lines, indent)
            else:
                lines.append(f"{prefix}- {item}")
