"""Content-hash based artifact manifest for deterministic workflow outputs."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any


ARTIFACT_MANIFEST_SCHEMA_VERSION = "artifact_manifest.v1"
AUDIT_FILENAMES = {"methodology_log.json", "artifact_manifest.json"}


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with a compact Z suffix."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 hex digest for a file."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_audit_artifact(path: str | Path) -> bool:
    return Path(path).name in AUDIT_FILENAMES


def normalize_json_value(value: Any) -> Any:
    """Normalize Python values into stable JSON-compatible structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): normalize_json_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, set):
        return [normalize_json_value(item) for item in sorted(value, key=str)]
    return value


def _empty_manifest() -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_timestamp(),
        "entries": [],
    }


def read_artifact_manifest(path: str | Path) -> dict[str, Any]:
    """Read an artifact manifest, returning an empty v1 manifest when absent."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        return _empty_manifest()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact manifest schema: {payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("entries"), list):
        raise ValueError("Artifact manifest must contain an entries list.")
    return payload


def write_artifact_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload["generated_at"] = utc_timestamp()
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def upsert_artifact_entry(
    manifest_path: str | Path,
    *,
    artifact_type: str,
    path: str | Path,
    generating_tool: str,
    parameters: dict[str, Any],
    source_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Upsert one deterministic artifact entry, returning the entry or None for audit files."""
    artifact_path = Path(path)
    if is_audit_artifact(artifact_path):
        return None

    payload = read_artifact_manifest(manifest_path)
    normalized_path = str(artifact_path)
    existing_entry = next(
        (
            entry
            for entry in payload["entries"]
            if entry.get("artifact_type") == artifact_type and entry.get("path") == normalized_path
        ),
        None,
    )
    now = utc_timestamp()
    entry = {
        "artifact_type": artifact_type,
        "path": normalized_path,
        "created_timestamp": (
            existing_entry.get("created_timestamp")
            if existing_entry
            else now
        ),
        "updated_timestamp": now,
        "content_hash": file_sha256(artifact_path),
        "source_artifacts": normalize_json_value(source_artifacts or []),
        "generating_tool": generating_tool,
        "parameters": normalize_json_value(parameters),
    }
    if existing_entry:
        payload["entries"][payload["entries"].index(existing_entry)] = entry
    else:
        payload["entries"].append(entry)
    payload["entries"] = sorted(
        payload["entries"],
        key=lambda item: (str(item.get("artifact_type")), str(item.get("path"))),
    )
    write_artifact_manifest(manifest_path, payload)
    return entry


def upsert_artifact_entries(
    manifest_path: str | Path,
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Upsert many deterministic artifact entries."""
    upserted: list[dict[str, Any]] = []
    for entry in entries:
        upserted_entry = upsert_artifact_entry(manifest_path, **entry)
        if upserted_entry is not None:
            upserted.append(upserted_entry)
    return upserted


def build_state_fingerprint_payload(
    *,
    source_hashes: dict[str, str | None],
    selected_columns: list[str] | None,
    filters: list[dict[str, Any]] | None,
    depth: int | None,
    sort_by: str | None,
    min_mac: int | None,
    packet_schema_version: str,
    skill_name: str | None = None,
    skill_version: str | None = None,
) -> dict[str, Any]:
    return normalize_json_value(
        {
            "source_hashes": source_hashes,
            "selected_columns": selected_columns,
            "filters": filters or [],
            "depth": depth,
            "sort_by": sort_by,
            "min_mac": min_mac,
            "packet_schema_version": packet_schema_version,
            "skill_name": skill_name,
            "skill_version": skill_version,
        }
    )


def build_state_fingerprint(
    *,
    source_hashes: dict[str, str | None],
    selected_columns: list[str] | None,
    filters: list[dict[str, Any]] | None,
    depth: int | None,
    sort_by: str | None,
    min_mac: int | None,
    packet_schema_version: str = "pre_ai_v0",
    skill_name: str | None = None,
    skill_version: str | None = None,
) -> str:
    """Return a deterministic SHA256 fingerprint for stale-state checks."""
    payload = build_state_fingerprint_payload(
        source_hashes=source_hashes,
        selected_columns=selected_columns,
        filters=filters,
        depth=depth,
        sort_by=sort_by,
        min_mac=min_mac,
        packet_schema_version=packet_schema_version,
        skill_name=skill_name,
        skill_version=skill_version,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def update_manifest_fingerprint(
    manifest_path: str | Path,
    *,
    fingerprint: str,
    fingerprint_inputs: dict[str, Any],
) -> Path:
    """Persist the latest state fingerprint into an existing or new manifest."""
    payload = read_artifact_manifest(manifest_path)
    payload["state_fingerprint"] = fingerprint
    payload["fingerprint_inputs"] = normalize_json_value(fingerprint_inputs)
    return write_artifact_manifest(manifest_path, payload)
