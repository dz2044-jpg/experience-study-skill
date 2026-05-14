"""Append-only methodology event log for deterministic workflow auditability."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


METHODOLOGY_LOG_SCHEMA_VERSION = "methodology_log.v1"


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with a compact Z suffix."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class MethodologyEvent:
    """Single deterministic methodology step recorded for audit handoff."""

    step_name: str
    tool_name: str
    input_path: str | None
    output_path: str | None
    parameters: dict[str, Any]
    timestamp: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _empty_log() -> dict[str, Any]:
    return {"schema_version": METHODOLOGY_LOG_SCHEMA_VERSION, "events": []}


def read_methodology_log(path: str | Path) -> dict[str, Any]:
    """Read a methodology log, returning an empty v1 log when it is absent."""
    log_path = Path(path)
    if not log_path.exists():
        return _empty_log()

    payload = json.loads(log_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != METHODOLOGY_LOG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported methodology log schema: {payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("events"), list):
        raise ValueError("Methodology log must contain an events list.")
    return payload


def append_methodology_event(path: str | Path, event: MethodologyEvent) -> Path:
    """Append one methodology event to the JSON log and return the log path."""
    log_path = Path(path)
    payload = read_methodology_log(log_path)
    payload["events"].append(event.to_dict())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return log_path
