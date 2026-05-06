from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    raw = value.strip()
    if raw in {"null", "None"}:
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        pass

    return raw


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load a flat key-value YAML file without external dependencies."""
    config: dict[str, Any] = {}
    file_path = Path(path)
    current_list_key: str | None = None

    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("-"):
            if current_list_key is None:
                raise ValueError(f"List item without a list key at line {line_number}: {line}")
            item = stripped[1:].strip()
            if not item:
                raise ValueError(f"Empty list item at line {line_number}: {line}")
            config[current_list_key].append(_parse_scalar(item))
            continue

        current_list_key = None
        if ":" not in stripped:
            raise ValueError(f"Invalid config line {line_number}: {line}")

        key, value = stripped.split(":", maxsplit=1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty config key at line {line_number}.")
        if value.strip() == "":
            config[key] = []
            current_list_key = key
        else:
            config[key] = _parse_scalar(value)

    return config
