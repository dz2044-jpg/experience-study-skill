"""Shared artifact readiness and freshness helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.artifact_manifest import file_sha256, read_artifact_manifest


FRESHNESS_LABELS: Mapping[str, str] = {
    "state_fingerprint": "state fingerprint",
    "packet_fingerprint": "packet fingerprint",
    "sweep_content_hash": "sweep content hash",
}


@dataclass(frozen=True, slots=True)
class PathState:
    """Existence state for a recorded artifact path."""

    path: Path | None
    exists: bool

    @property
    def is_recorded(self) -> bool:
        return self.path is not None


@dataclass(slots=True)
class AIArtifactReadiness:
    """Readiness facts required to build and interpret the latest AI packet."""

    checks: dict[str, bool | None]
    sweep_path: Path | None = None
    artifact_manifest_path: Path | None = None
    state_fingerprint: str | None = None
    sweep_content_hash: str | None = None
    actual_sweep_content_hash: str | None = None

    @property
    def ready(self) -> bool:
        return all(check is not False for check in self.checks.values())

    @property
    def sweep_hash_matches_file(self) -> bool | None:
        if not self.sweep_content_hash or not self.actual_sweep_content_hash:
            return None
        return self.sweep_content_hash == self.actual_sweep_content_hash


@dataclass(frozen=True, slots=True)
class AIResponseFreshness:
    """Comparison result for a stored AI response against the current sweep state."""

    state_fingerprint_matches: bool
    packet_fingerprint_matches: bool
    sweep_content_hash_matches: bool
    reason: str | None = None

    @property
    def is_fresh(self) -> bool:
        return (
            self.reason is None
            and self.state_fingerprint_matches
            and self.packet_fingerprint_matches
            and self.sweep_content_hash_matches
        )

    @property
    def mismatches(self) -> tuple[str, ...]:
        fields = (
            ("state_fingerprint", self.state_fingerprint_matches),
            ("packet_fingerprint", self.packet_fingerprint_matches),
            ("sweep_content_hash", self.sweep_content_hash_matches),
        )
        return tuple(
            FRESHNESS_LABELS[field_name]
            for field_name, matches in fields
            if not matches
        )


def coerce_path(value: Any) -> Path | None:
    """Return a Path for path-like values, or None when coercion is not possible."""

    if value is None:
        return None
    try:
        return Path(value)
    except TypeError:
        return None


def path_exists(path: Path | None) -> bool:
    """Return whether a path exists without surfacing filesystem errors."""

    if path is None:
        return False
    try:
        return path.exists()
    except OSError:
        return False


def path_state(value: Any) -> PathState:
    """Return recorded/existence facts for a path-like value."""

    path = coerce_path(value)
    return PathState(path=path, exists=path_exists(path))


def paths_match(candidate: Any, target: Any) -> bool:
    """Return whether two path-like values identify the same artifact."""

    candidate_path = coerce_path(candidate)
    target_path = coerce_path(target)
    if candidate_path is None or target_path is None:
        return False
    if candidate_path == target_path or str(candidate_path) == str(target_path):
        return True
    try:
        return (
            candidate_path.exists()
            and target_path.exists()
            and candidate_path.resolve() == target_path.resolve()
        )
    except OSError:
        return False


def safe_read_artifact_manifest(path: Any) -> dict[str, Any]:
    """Read a manifest, returning an empty entries payload when unavailable."""

    manifest_path = coerce_path(path)
    if manifest_path is None or not path_exists(manifest_path):
        return {"entries": []}
    try:
        payload = read_artifact_manifest(manifest_path)
    except (OSError, TypeError, ValueError):
        return {"entries": []}
    if not isinstance(payload.get("entries"), list):
        return {"entries": []}
    return payload


def manifest_entry_for_path(
    manifest: dict[str, Any],
    *,
    path: Any,
    artifact_type: str | None = None,
) -> dict[str, Any] | None:
    """Find the manifest entry for a path and optional artifact type."""

    artifact_path = coerce_path(path)
    if artifact_path is None:
        return None
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        if artifact_type and entry.get("artifact_type") != artifact_type:
            continue
        if paths_match(entry.get("path"), artifact_path):
            return entry
    return None


def manifest_content_hash_for_path(
    manifest_path: Any,
    artifact_path: Any,
    *,
    artifact_type: str | None = None,
) -> str | None:
    """Return the manifest content hash for an artifact path."""

    manifest = safe_read_artifact_manifest(manifest_path)
    entry = manifest_entry_for_path(
        manifest,
        path=artifact_path,
        artifact_type=artifact_type,
    )
    content_hash = entry.get("content_hash") if entry else None
    return str(content_hash) if content_hash else None


def entry_source_matches(
    entry: dict[str, Any] | None,
    source_path: Path | None,
    source_entry: dict[str, Any] | None = None,
) -> bool | None:
    """Return whether an entry records the current source artifact relationship."""

    if entry is None or source_path is None:
        return None
    sources = entry.get("source_artifacts", [])
    if not isinstance(sources, list) or not sources:
        return None
    for source in sources:
        if not isinstance(source, dict) or not paths_match(source.get("path"), source_path):
            continue
        source_hash = source.get("content_hash")
        current_hash = source_entry.get("content_hash") if source_entry else None
        if source_hash and current_hash:
            return source_hash == current_hash
        return True
    return False


def get_ai_artifact_readiness(
    state: Any,
    *,
    include_file_hash: bool = True,
    refresh_state: bool = True,
) -> AIArtifactReadiness:
    """Return artifact readiness facts for the AI interpretation workflow."""

    if refresh_state and hasattr(state, "refresh"):
        state.refresh()

    sweep_path = coerce_path(getattr(state, "latest_sweep_path", None))
    manifest_path = coerce_path(getattr(state, "artifact_manifest_path", None))
    state_fingerprint = getattr(state, "latest_state_fingerprint", None)

    latest_sweep_ready = path_exists(sweep_path)
    artifact_manifest_ready = path_exists(manifest_path)

    actual_sweep_content_hash = None
    if include_file_hash and latest_sweep_ready and sweep_path:
        try:
            actual_sweep_content_hash = file_sha256(sweep_path)
        except OSError:
            actual_sweep_content_hash = None

    sweep_content_hash = None
    if latest_sweep_ready and artifact_manifest_ready:
        sweep_content_hash = manifest_content_hash_for_path(
            manifest_path,
            sweep_path,
            artifact_type="sweep_summary",
        )
    sweep_hash_matches_file = (
        bool(
            sweep_content_hash
            and actual_sweep_content_hash
            and sweep_content_hash == actual_sweep_content_hash
        )
        if include_file_hash
        else None
    )

    return AIArtifactReadiness(
        checks={
            "latest_sweep": latest_sweep_ready,
            "artifact_manifest": artifact_manifest_ready,
            "state_fingerprint": bool(state_fingerprint),
            "sweep_manifest_hash": bool(sweep_content_hash),
            "sweep_hash_matches_file": sweep_hash_matches_file,
        },
        sweep_path=sweep_path,
        artifact_manifest_path=manifest_path,
        state_fingerprint=str(state_fingerprint) if state_fingerprint else None,
        sweep_content_hash=sweep_content_hash,
        actual_sweep_content_hash=actual_sweep_content_hash,
    )


def compare_ai_response_freshness(
    response_record: Mapping[str, Any],
    *,
    readiness: AIArtifactReadiness,
    packet_fingerprint: str | None,
    reason: str | None = None,
) -> AIResponseFreshness:
    """Compare stored AI response metadata to the current artifact state."""

    stored_state_fingerprint = response_record.get("response_state_fingerprint")
    stored_packet_fingerprint = response_record.get("response_packet_fingerprint")
    stored_sweep_content_hash = response_record.get("response_sweep_content_hash")
    return AIResponseFreshness(
        state_fingerprint_matches=bool(
            stored_state_fingerprint
            and readiness.state_fingerprint
            and stored_state_fingerprint == readiness.state_fingerprint
        ),
        packet_fingerprint_matches=bool(
            stored_packet_fingerprint
            and packet_fingerprint
            and stored_packet_fingerprint == packet_fingerprint
        ),
        sweep_content_hash_matches=bool(
            stored_sweep_content_hash
            and readiness.sweep_content_hash
            and stored_sweep_content_hash == readiness.sweep_content_hash
        ),
        reason=reason,
    )
