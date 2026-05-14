from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.artifact_manifest import file_sha256, upsert_artifact_entry
from core.artifact_readiness import (
    compare_ai_response_freshness,
    get_ai_artifact_readiness,
    manifest_content_hash_for_path,
    paths_match,
)


def _refresh_noop() -> None:
    return None


def test_ai_artifact_readiness_uses_manifest_hash_and_file_hash(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep_summary.csv"
    sweep_path.write_text("Dimensions,Sum_MAC\nGender=M,2\n", encoding="utf-8")
    manifest_path = tmp_path / "audit" / "artifact_manifest.json"
    upsert_artifact_entry(
        manifest_path,
        artifact_type="sweep_summary",
        path=sweep_path,
        generating_tool="run_dimensional_sweep",
        parameters={"depth": 1},
        source_artifacts=[],
    )

    readiness = get_ai_artifact_readiness(
        SimpleNamespace(
            latest_sweep_path=sweep_path,
            artifact_manifest_path=manifest_path,
            latest_state_fingerprint="state-a",
            refresh=_refresh_noop,
        )
    )

    assert readiness.ready is True
    assert readiness.checks == {
        "latest_sweep": True,
        "artifact_manifest": True,
        "state_fingerprint": True,
        "sweep_manifest_hash": True,
        "sweep_hash_matches_file": True,
    }
    assert readiness.sweep_content_hash == file_sha256(sweep_path)
    assert readiness.actual_sweep_content_hash == file_sha256(sweep_path)
    assert readiness.sweep_hash_matches_file is True
    assert manifest_content_hash_for_path(manifest_path, sweep_path) == file_sha256(
        sweep_path
    )


def test_ai_artifact_readiness_skips_file_hash_when_requested(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep_summary.csv"
    sweep_path.write_text("Dimensions,Sum_MAC\nGender=M,2\n", encoding="utf-8")
    manifest_path = tmp_path / "audit" / "artifact_manifest.json"
    upsert_artifact_entry(
        manifest_path,
        artifact_type="sweep_summary",
        path=sweep_path,
        generating_tool="run_dimensional_sweep",
        parameters={"depth": 1},
        source_artifacts=[],
    )

    readiness = get_ai_artifact_readiness(
        SimpleNamespace(
            latest_sweep_path=sweep_path,
            artifact_manifest_path=manifest_path,
            latest_state_fingerprint="state-a",
            refresh=_refresh_noop,
        ),
        include_file_hash=False,
    )

    assert readiness.ready is True
    assert readiness.actual_sweep_content_hash is None
    assert readiness.sweep_hash_matches_file is None
    assert readiness.checks == {
        "latest_sweep": True,
        "artifact_manifest": True,
        "state_fingerprint": True,
        "sweep_manifest_hash": True,
        "sweep_hash_matches_file": None,
    }


def test_ai_artifact_readiness_blocks_manifest_file_hash_mismatch(
    tmp_path: Path,
) -> None:
    sweep_path = tmp_path / "sweep_summary.csv"
    sweep_path.write_text("Dimensions,Sum_MAC\nGender=M,2\n", encoding="utf-8")
    manifest_path = tmp_path / "audit" / "artifact_manifest.json"
    upsert_artifact_entry(
        manifest_path,
        artifact_type="sweep_summary",
        path=sweep_path,
        generating_tool="run_dimensional_sweep",
        parameters={"depth": 1},
        source_artifacts=[],
    )
    sweep_path.write_text("Dimensions,Sum_MAC\nGender=F,4\n", encoding="utf-8")

    readiness = get_ai_artifact_readiness(
        SimpleNamespace(
            latest_sweep_path=sweep_path,
            artifact_manifest_path=manifest_path,
            latest_state_fingerprint="state-a",
            refresh=_refresh_noop,
        )
    )

    assert readiness.ready is False
    assert readiness.checks["sweep_manifest_hash"] is True
    assert readiness.checks["sweep_hash_matches_file"] is False
    assert readiness.sweep_hash_matches_file is False
    assert readiness.sweep_content_hash != readiness.actual_sweep_content_hash


def test_paths_match_exact_strings_and_resolved_existing_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("artifact", encoding="utf-8")

    assert paths_match(str(artifact), artifact)
    assert paths_match(artifact, artifact.resolve())
    assert not paths_match(artifact, tmp_path / "missing.csv")


def test_compare_ai_response_freshness_reports_explicit_mismatches() -> None:
    readiness = get_ai_artifact_readiness(
        SimpleNamespace(
            latest_sweep_path=None,
            artifact_manifest_path=None,
            latest_state_fingerprint="state-b",
            refresh=_refresh_noop,
        )
    )
    readiness.sweep_content_hash = "hash-b"

    freshness = compare_ai_response_freshness(
        {
            "response_state_fingerprint": "state-a",
            "response_packet_fingerprint": "packet-a",
            "response_sweep_content_hash": "hash-b",
        },
        readiness=readiness,
        packet_fingerprint="packet-b",
    )

    assert freshness.is_fresh is False
    assert freshness.state_fingerprint_matches is False
    assert freshness.packet_fingerprint_matches is False
    assert freshness.sweep_content_hash_matches is True
    assert freshness.mismatches == ("state fingerprint", "packet fingerprint")
