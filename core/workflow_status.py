"""Read-only guided workflow status derivation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from core.artifact_readiness import (
    PathState,
    coerce_path,
    entry_source_matches,
    manifest_entry_for_path,
    path_exists,
    path_state,
    safe_read_artifact_manifest,
)
from core.methodology_log import read_methodology_log


WorkflowStepStatus = Literal["not_started", "ready", "completed", "blocked", "stale"]

_PROFILE_TOOL = "profile_dataset"
_SCHEMA_TOOL = "inspect_dataset_schema"
_VALIDATION_TOOL = "run_actuarial_data_checks"
_FEATURE_TOOLS = {"create_categorical_bands", "regroup_categorical_features"}
_SWEEP_TOOL = "run_dimensional_sweep"
_VISUALIZATION_TOOL = "generate_combined_report"


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """One read-only guided workflow status row."""

    id: str
    label: str
    status: WorkflowStepStatus
    detail: str
    prerequisite_message: str | None = None
    artifact_path: Path | None = None
    basis: str | None = None


@dataclass(frozen=True, slots=True)
class AIWorkflowSnapshot:
    """Cheap mirror of AI panel readiness and stored-response freshness."""

    ready: bool = False
    readiness_checks: Mapping[str, bool] = field(default_factory=dict)
    has_response: bool = False
    response_is_fresh: bool | None = None
    freshness_mismatches: tuple[str, ...] = ()
    detail: str | None = None
    basis: str | None = None


def _path_detail(prefix: str, path: Path | None) -> str:
    if path is None:
        return prefix
    return f"{prefix}: {path}"


def _default_path(state: Any, attribute_name: str) -> Path | None:
    try:
        return coerce_path(getattr(state, attribute_name))
    except (AttributeError, OSError, TypeError):
        return None


def _safe_methodology_events(state: Any) -> list[dict[str, Any]]:
    path = coerce_path(getattr(state, "methodology_log_path", None))
    if path is None:
        path = _default_path(state, "default_methodology_log_path")
    if path is None or not path_exists(path):
        return []
    try:
        payload = read_methodology_log(path)
    except (OSError, TypeError, ValueError):
        return []
    events = payload.get("events", [])
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _safe_artifact_manifest(state: Any) -> dict[str, Any]:
    path = coerce_path(getattr(state, "artifact_manifest_path", None))
    if path is None:
        path = _default_path(state, "default_artifact_manifest_path")
    return safe_read_artifact_manifest(path)


def _latest_event_index(events: list[dict[str, Any]], tool_names: set[str]) -> int | None:
    for index in range(len(events) - 1, -1, -1):
        if events[index].get("tool_name") in tool_names:
            return index
    return None


def _latest_index(*indexes: int | None) -> int | None:
    present_indexes = [index for index in indexes if index is not None]
    return max(present_indexes) if present_indexes else None


def _schema_or_validation_step(
    *,
    step_id: str,
    label: str,
    tool_name: str,
    active_dataset: PathState,
    events: list[dict[str, Any]],
    latest_dataset_mutation_index: int | None,
) -> WorkflowStep:
    if not active_dataset.exists:
        return WorkflowStep(
            id=step_id,
            label=label,
            status="blocked",
            detail="No active analysis dataset is available.",
            prerequisite_message="Profile a dataset before this step.",
            basis="current active analysis dataset",
        )

    step_index = _latest_event_index(events, {tool_name})
    if step_index is None:
        return WorkflowStep(
            id=step_id,
            label=label,
            status="ready",
            detail=_path_detail("Ready for current active analysis dataset", active_dataset.path),
            prerequisite_message=None,
            artifact_path=active_dataset.path,
            basis="current active analysis dataset",
        )

    if latest_dataset_mutation_index is not None and latest_dataset_mutation_index > step_index:
        return WorkflowStep(
            id=step_id,
            label=label,
            status="stale",
            detail="The active prepared dataset changed after this step last ran.",
            prerequisite_message="Rerun this step for the current active analysis dataset.",
            artifact_path=active_dataset.path,
            basis="methodology event order",
        )

    return WorkflowStep(
        id=step_id,
        label=label,
        status="completed",
        detail=_path_detail("Completed for the current active analysis dataset", active_dataset.path),
        artifact_path=active_dataset.path,
        basis="latest methodology event",
    )


def _dataset_step(raw: PathState, prepared: PathState) -> WorkflowStep:
    if prepared.exists:
        return WorkflowStep(
            id="dataset",
            label="Dataset",
            status="completed",
            detail=_path_detail("Prepared dataset is available", prepared.path),
            artifact_path=prepared.path,
            basis="latest prepared dataset",
        )
    if raw.exists:
        return WorkflowStep(
            id="dataset",
            label="Dataset",
            status="completed",
            detail=_path_detail("Source dataset is available", raw.path),
            artifact_path=raw.path,
            basis="source dataset",
        )
    if prepared.is_recorded or raw.is_recorded:
        recorded_path = prepared.path if prepared.is_recorded else raw.path
        return WorkflowStep(
            id="dataset",
            label="Dataset",
            status="stale",
            detail=_path_detail("Recorded dataset path is missing", recorded_path),
            prerequisite_message="Restore the dataset artifact or profile the source dataset again.",
            artifact_path=recorded_path,
            basis="recorded dataset path",
        )
    return WorkflowStep(
        id="dataset",
        label="Dataset",
        status="not_started",
        detail="No dataset has been selected or profiled.",
        prerequisite_message="Profile a dataset to start the workflow.",
        basis="session state",
    )


def _feature_step(
    *,
    active_dataset: PathState,
    prepared: PathState,
    events: list[dict[str, Any]],
    latest_profile_index: int | None,
) -> WorkflowStep:
    if not active_dataset.exists:
        return WorkflowStep(
            id="feature_engineering",
            label="Feature Engineering",
            status="blocked",
            detail="No active analysis dataset is available.",
            prerequisite_message="Profile a dataset before feature engineering.",
            basis="current active analysis dataset",
        )

    feature_index = _latest_event_index(events, _FEATURE_TOOLS)
    if feature_index is None:
        return WorkflowStep(
            id="feature_engineering",
            label="Feature Engineering",
            status="ready",
            detail=_path_detail("Ready for the active analysis dataset", active_dataset.path),
            artifact_path=active_dataset.path,
            basis="current active analysis dataset",
        )

    if prepared.is_recorded and not prepared.exists:
        return WorkflowStep(
            id="feature_engineering",
            label="Feature Engineering",
            status="stale",
            detail=_path_detail("Prepared dataset artifact is missing", prepared.path),
            prerequisite_message="Restore or recreate the prepared dataset.",
            artifact_path=prepared.path,
            basis="latest prepared dataset",
        )

    if latest_profile_index is not None and latest_profile_index > feature_index:
        return WorkflowStep(
            id="feature_engineering",
            label="Feature Engineering",
            status="stale",
            detail="A later profile step superseded the last feature engineering step.",
            prerequisite_message="Rerun feature engineering if derived fields are needed.",
            artifact_path=prepared.path or active_dataset.path,
            basis="methodology event order",
        )

    return WorkflowStep(
        id="feature_engineering",
        label="Feature Engineering",
        status="completed",
        detail=_path_detail("Feature engineering has been applied", prepared.path or active_dataset.path),
        artifact_path=prepared.path or active_dataset.path,
        basis="latest feature engineering event",
    )


def _sweep_step(
    *,
    prepared: PathState,
    sweep: PathState,
    events: list[dict[str, Any]],
    manifest: dict[str, Any],
    latest_dataset_mutation_index: int | None,
) -> WorkflowStep:
    if prepared.is_recorded and not prepared.exists:
        return WorkflowStep(
            id="sweep",
            label="Sweep",
            status="stale",
            detail=_path_detail("Prepared dataset artifact is missing", prepared.path),
            prerequisite_message="Restore or recreate the prepared dataset before sweeping.",
            artifact_path=prepared.path,
            basis="latest prepared dataset",
        )

    if not prepared.exists:
        return WorkflowStep(
            id="sweep",
            label="Sweep",
            status="blocked",
            detail="No prepared dataset is available for A/E analysis.",
            prerequisite_message="Profile a dataset before running a sweep.",
            basis="latest prepared dataset",
        )

    if not sweep.is_recorded:
        return WorkflowStep(
            id="sweep",
            label="Sweep",
            status="ready",
            detail=_path_detail("Ready to run against prepared dataset", prepared.path),
            artifact_path=prepared.path,
            basis="latest prepared dataset",
        )

    if not sweep.exists:
        return WorkflowStep(
            id="sweep",
            label="Sweep",
            status="stale",
            detail=_path_detail("Latest sweep artifact is missing", sweep.path),
            prerequisite_message="Rerun the dimensional sweep.",
            artifact_path=sweep.path,
            basis="latest sweep artifact",
        )

    sweep_entry = manifest_entry_for_path(
        manifest,
        artifact_type="sweep_summary",
        path=sweep.path,
    )
    prepared_entry = manifest_entry_for_path(
        manifest,
        artifact_type="prepared_dataset",
        path=prepared.path,
    )
    source_matches = entry_source_matches(sweep_entry, prepared.path, prepared_entry)
    if source_matches is False:
        return WorkflowStep(
            id="sweep",
            label="Sweep",
            status="stale",
            detail="The sweep manifest points to a different prepared dataset.",
            prerequisite_message="Rerun the dimensional sweep for the active prepared dataset.",
            artifact_path=sweep.path,
            basis="artifact manifest source artifact",
        )

    sweep_index = _latest_event_index(events, {_SWEEP_TOOL})
    if (
        latest_dataset_mutation_index is not None
        and sweep_index is not None
        and latest_dataset_mutation_index > sweep_index
    ):
        return WorkflowStep(
            id="sweep",
            label="Sweep",
            status="stale",
            detail="The prepared dataset changed after the latest sweep.",
            prerequisite_message="Rerun the dimensional sweep.",
            artifact_path=sweep.path,
            basis="methodology event order",
        )

    return WorkflowStep(
        id="sweep",
        label="Sweep",
        status="completed",
        detail=_path_detail("Latest sweep artifact is available", sweep.path),
        artifact_path=sweep.path,
        basis="latest sweep artifact",
    )


def _visualization_step(
    *,
    sweep: PathState,
    visualization: PathState,
    events: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> WorkflowStep:
    if visualization.is_recorded and not visualization.exists:
        return WorkflowStep(
            id="visualization",
            label="Visualization",
            status="stale",
            detail=_path_detail("Visualization artifact is missing", visualization.path),
            prerequisite_message="Regenerate the combined report.",
            artifact_path=visualization.path,
            basis="latest visualization artifact",
        )

    if not sweep.exists:
        return WorkflowStep(
            id="visualization",
            label="Visualization",
            status="blocked",
            detail="No current sweep artifact is available.",
            prerequisite_message="Run a dimensional sweep before generating the report.",
            basis="latest sweep artifact",
        )

    if not visualization.is_recorded:
        return WorkflowStep(
            id="visualization",
            label="Visualization",
            status="ready",
            detail=_path_detail("Ready to generate from latest sweep", sweep.path),
            artifact_path=sweep.path,
            basis="latest sweep artifact",
        )

    visualization_entry = manifest_entry_for_path(
        manifest,
        artifact_type="visualization_report",
        path=visualization.path,
    )
    sweep_entry = manifest_entry_for_path(
        manifest,
        artifact_type="sweep_summary",
        path=sweep.path,
    )
    source_matches = entry_source_matches(visualization_entry, sweep.path, sweep_entry)
    if source_matches is False:
        return WorkflowStep(
            id="visualization",
            label="Visualization",
            status="stale",
            detail="The visualization manifest points to a different sweep artifact.",
            prerequisite_message="Regenerate the combined report for the latest sweep.",
            artifact_path=visualization.path,
            basis="artifact manifest source artifact",
        )

    sweep_index = _latest_event_index(events, {_SWEEP_TOOL})
    visualization_index = _latest_event_index(events, {_VISUALIZATION_TOOL})
    if (
        sweep_index is not None
        and visualization_index is not None
        and sweep_index > visualization_index
    ):
        return WorkflowStep(
            id="visualization",
            label="Visualization",
            status="stale",
            detail="A newer sweep exists after the latest visualization.",
            prerequisite_message="Regenerate the combined report.",
            artifact_path=visualization.path,
            basis="methodology event order",
        )

    return WorkflowStep(
        id="visualization",
        label="Visualization",
        status="completed",
        detail=_path_detail("Visualization artifact is available", visualization.path),
        artifact_path=visualization.path,
        basis="latest visualization artifact",
    )


def _ai_step(ai_snapshot: AIWorkflowSnapshot | None) -> WorkflowStep:
    snapshot = ai_snapshot or AIWorkflowSnapshot()
    basis = snapshot.basis or "AI panel readiness"

    if not snapshot.ready:
        if snapshot.has_response:
            return WorkflowStep(
                id="ai_interpretation",
                label="AI Interpretation",
                status="stale",
                detail=snapshot.detail or "A stored AI response no longer has ready artifacts.",
                prerequisite_message="Restore the required artifacts or run a new sweep.",
                basis=basis,
            )
        missing_checks = [
            check_name.replace("_", " ")
            for check_name, is_ready in snapshot.readiness_checks.items()
            if not is_ready
        ]
        missing_detail = (
            "Missing: " + ", ".join(missing_checks)
            if missing_checks
            else "AI interpretation prerequisites are not ready."
        )
        return WorkflowStep(
            id="ai_interpretation",
            label="AI Interpretation",
            status="blocked",
            detail=snapshot.detail or missing_detail,
            prerequisite_message="Run a dimensional sweep with artifact tracking first.",
            basis=basis,
        )

    if snapshot.has_response:
        if snapshot.response_is_fresh is False:
            mismatch_detail = (
                "Stale fields: " + ", ".join(snapshot.freshness_mismatches)
                if snapshot.freshness_mismatches
                else "Stored AI response does not match the latest sweep state."
            )
            return WorkflowStep(
                id="ai_interpretation",
                label="AI Interpretation",
                status="stale",
                detail=snapshot.detail or mismatch_detail,
                prerequisite_message="Run AI interpretation again for the latest sweep.",
                basis=basis,
            )
        return WorkflowStep(
            id="ai_interpretation",
            label="AI Interpretation",
            status="completed",
            detail=snapshot.detail or "A fresh AI interpretation response is available.",
            basis=basis,
        )

    return WorkflowStep(
        id="ai_interpretation",
        label="AI Interpretation",
        status="ready",
        detail=snapshot.detail or "AI interpretation can run for the latest sweep.",
        prerequisite_message=None,
        basis=basis,
    )


def derive_workflow_steps(
    state: Any,
    ai_snapshot: AIWorkflowSnapshot | None = None,
) -> list[WorkflowStep]:
    """Derive read-only guided workflow status from current session artifacts.

    Schema and validation statuses refer to the current active analysis dataset.
    If feature engineering creates or updates the prepared dataset afterward,
    those review steps become stale until rerun for the active prepared artifact.
    """
    raw = path_state(getattr(state, "raw_input_path", None))
    prepared = path_state(getattr(state, "prepared_dataset_path", None))
    sweep = path_state(getattr(state, "latest_sweep_path", None))
    visualization = path_state(getattr(state, "latest_visualization_path", None))

    events = _safe_methodology_events(state)
    manifest = _safe_artifact_manifest(state)

    latest_profile_index = _latest_event_index(events, {_PROFILE_TOOL})
    latest_feature_index = _latest_event_index(events, _FEATURE_TOOLS)
    latest_dataset_mutation_index = _latest_index(latest_profile_index, latest_feature_index)

    active_dataset = prepared if prepared.exists else raw

    return [
        _dataset_step(raw, prepared),
        _schema_or_validation_step(
            step_id="schema",
            label="Schema",
            tool_name=_SCHEMA_TOOL,
            active_dataset=active_dataset,
            events=events,
            latest_dataset_mutation_index=latest_dataset_mutation_index,
        ),
        _schema_or_validation_step(
            step_id="validation",
            label="Validation",
            tool_name=_VALIDATION_TOOL,
            active_dataset=active_dataset,
            events=events,
            latest_dataset_mutation_index=latest_dataset_mutation_index,
        ),
        _feature_step(
            active_dataset=active_dataset,
            prepared=prepared,
            events=events,
            latest_profile_index=latest_profile_index,
        ),
        _sweep_step(
            prepared=prepared,
            sweep=sweep,
            events=events,
            manifest=manifest,
            latest_dataset_mutation_index=latest_dataset_mutation_index,
        ),
        _visualization_step(
            sweep=sweep,
            visualization=visualization,
            events=events,
            manifest=manifest,
        ),
        _ai_step(ai_snapshot),
    ]
