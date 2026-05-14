"""Streamlit entry point for the unified skill-based experience study copilot."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Iterable

from core.artifact_readiness import (
    AIArtifactReadiness,
    compare_ai_response_freshness,
    get_ai_artifact_readiness,
    manifest_content_hash_for_path,
    paths_match,
)
from core.copilot_agent import CopilotEvent, UnifiedCopilot
from core.workflow_status import (
    AIWorkflowSnapshot,
    WorkflowStep,
    derive_workflow_steps,
)
import pandas as pd
from skills.experience_study_skill.ai_models import (
    AIActionName,
    AIActionResponse,
    AISweepPacket,
)
from skills.experience_study_skill.ai_orchestrator import run_ai_action
from skills.experience_study_skill.ai_packets import build_latest_sweep_packet

try:
    import streamlit as st
except ImportError:  # pragma: no cover - depends on environment
    st = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

EMPTY_STATE_SUGGESTIONS = (
    (
        "info",
        "**Profile Data**\n\n`Profile data/input/synthetic_inforce.csv and tell me the columns.`",
    ),
    (
        "success",
        "**Run a Sweep**\n\n`Run a 1-way sweep on Gender after profiling my dataset.`",
    ),
    (
        "warning",
        "**Visualize**\n\n`Generate the combined report for the latest sweep.`",
    ),
)

_SWEEP_EXPLORER_COLUMN_ORDER = [
    "Dimensions",
    "Sum_MAC",
    "Sum_MOC",
    "Sum_MEC",
    "Sum_MAF",
    "Sum_MEF",
    "AE_Ratio_Count",
    "AE_Ratio_Amount",
    "AE_Count_CI_Lower",
    "AE_Count_CI_Upper",
    "AE_Amount_CI_Lower",
    "AE_Amount_CI_Upper",
]

_SWEEP_EXPLORER_COLUMN_LABELS = {
    "Dimensions": "Cohort Dimension",
    "Sum_MAC": "Actual Deaths (MAC)",
    "Sum_MOC": "Exposure (MOC)",
    "Sum_MEC": "Expected (MEC)",
    "Sum_MAF": "Actual Amount (MAF)",
    "Sum_MEF": "Expected Amount (MEF)",
    "AE_Ratio_Count": "A/E Ratio (Count)",
    "AE_Ratio_Amount": "A/E Ratio (Amount)",
    "AE_Count_CI_Lower": "Count CI Lower",
    "AE_Count_CI_Upper": "Count CI Upper",
    "AE_Amount_CI_Lower": "Amount CI Lower",
    "AE_Amount_CI_Upper": "Amount CI Upper",
}

_AI_READINESS_LABELS = {
    "latest_sweep": "Latest sweep artifact",
    "artifact_manifest": "Artifact manifest",
    "state_fingerprint": "State fingerprint",
    "sweep_manifest_hash": "Sweep manifest content hash",
    "sweep_hash_matches_file": "Manifest/file hash match",
}

_WORKFLOW_STATUS_LABELS = {
    "not_started": "Not started",
    "ready": "Ready",
    "completed": "Completed",
    "blocked": "Blocked",
    "stale": "Stale",
}


_AIArtifactReadiness = AIArtifactReadiness
_paths_match = paths_match
_manifest_content_hash_for_path = manifest_content_hash_for_path


def _require_streamlit() -> None:
    if st is None:  # pragma: no cover - depends on environment
        raise RuntimeError("Streamlit is required to run the web app.")


def _render_empty_state() -> None:
    _require_streamlit()
    st.markdown("### Experience Study Copilot")
    st.markdown(
        "This session uses a unified skill package with session-scoped deterministic "
        "artifacts for profiling, feature engineering, dimensional sweeps, and visualization."
    )
    columns = st.columns(3)
    for column, (variant, message) in zip(columns, EMPTY_STATE_SUGGESTIONS):
        with column:
            getattr(st, variant)(message)
    st.markdown("---")


def _render_visualization_card(visualization_path: str | None, widget_key_prefix: str) -> None:
    _require_streamlit()
    if not visualization_path:
        return
    html_path = Path(visualization_path)
    if not html_path.exists():
        return

    resolved_path = html_path.resolve()
    html_bytes = resolved_path.read_bytes()
    with st.container(border=True):
        st.markdown("##### Generated Visualization Artifact")
        st.caption(f"`{resolved_path.name}`")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                "Download HTML",
                data=html_bytes,
                file_name=resolved_path.name,
                mime="text/html",
                key=f"{widget_key_prefix}-download-visualization",
            )
        with col2:
            st.caption("Download the standalone report, or preview it inline below.")
        with st.expander("Preview Visualization Inline", expanded=False):
            st.components.v1.html(
                resolved_path.read_text(encoding="utf-8"),
                height=600,
                scrolling=True,
            )


def _build_sweep_display_frame(sweep_results: list[dict[str, Any]] | None) -> pd.DataFrame:
    if not sweep_results:
        return pd.DataFrame()

    display_df = pd.DataFrame(sweep_results)
    available_columns = [
        column for column in _SWEEP_EXPLORER_COLUMN_ORDER if column in display_df.columns
    ]
    if not available_columns:
        return pd.DataFrame()

    display_df = display_df.loc[:, available_columns].rename(columns=_SWEEP_EXPLORER_COLUMN_LABELS)
    for column in display_df.columns:
        if column == _SWEEP_EXPLORER_COLUMN_LABELS["Dimensions"]:
            continue
        display_df[column] = display_df[column].map(_format_display_number)
    return display_df


def _render_sweep_explorer(sweep_results: list[dict[str, Any]] | None) -> None:
    _require_streamlit()
    display_df = _build_sweep_display_frame(sweep_results)
    if display_df.empty:
        return

    with st.container(border=True):
        st.markdown("##### Sweep Explorer")
        st.caption("Sort the table columns to compare cohorts across sweep metrics.")
        st.dataframe(display_df, hide_index=True, use_container_width=True)


def _get_ai_panel_readiness(
    state: Any,
    *,
    include_file_hash: bool = True,
    refresh_state: bool = True,
) -> _AIArtifactReadiness:
    return get_ai_artifact_readiness(
        state,
        include_file_hash=include_file_hash,
        refresh_state=refresh_state,
    )


def _coerce_float(value: Any) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric_value) or not math.isfinite(numeric_value):
        return None
    return numeric_value


def _format_display_number(value: Any) -> float | str:
    numeric_value = _coerce_float(value)
    if numeric_value is None:
        return "n/a"
    return round(numeric_value, 2)


def _format_ai_number(value: Any) -> str:
    numeric_value = _coerce_float(value)
    if numeric_value is None:
        return "n/a"
    return f"{numeric_value:.2f}"


def _select_top_cohort_evidence_ref(rows: list[Any]) -> str | None:
    top_ref: str | None = None
    top_value: float | None = None
    for row in rows:
        ae_amount = _coerce_float(getattr(row, "AE_Ratio_Amount", None))
        if ae_amount is None:
            continue
        if top_value is None or ae_amount > top_value:
            top_value = ae_amount
            top_ref = str(getattr(row, "evidence_ref", ""))
    return top_ref or None


def _format_ai_cohort_label(row: Any) -> str:
    dimensions = str(getattr(row, "Dimensions", "Unknown cohort"))
    ae_amount = _format_ai_number(getattr(row, "AE_Ratio_Amount", None))
    mac = _format_ai_number(getattr(row, "Sum_MAC", None))
    evidence_ref = str(getattr(row, "evidence_ref", "unknown"))
    return f"{dimensions} | A/E Amount={ae_amount} | MAC={mac} | {evidence_ref}"


def _build_ai_packet_for_panel(readiness: _AIArtifactReadiness) -> AISweepPacket:
    if not readiness.ready:
        missing = [
            _AI_READINESS_LABELS.get(check_name, check_name)
            for check_name, is_ready in readiness.checks.items()
            if is_ready is False
        ]
        detail = ", ".join(missing) if missing else "unknown prerequisite"
        raise ValueError(f"AI interpretation prerequisites are not ready: {detail}.")
    if not readiness.sweep_path:
        raise ValueError("Latest sweep artifact is required for AI interpretation.")
    return build_latest_sweep_packet(
        sweep_path=readiness.sweep_path,
        artifact_manifest_path=readiness.artifact_manifest_path,
    )


def _build_ai_response_record(
    *,
    action_name: AIActionName,
    selected_evidence_ref: str | None,
    response: AIActionResponse,
    packet: AISweepPacket,
    readiness: _AIArtifactReadiness,
) -> dict[str, Any]:
    return {
        "action_name": action_name,
        "selected_evidence_ref": selected_evidence_ref,
        "response": response,
        "response_state_fingerprint": (
            response.state_fingerprint or readiness.state_fingerprint
        ),
        "response_packet_fingerprint": (
            response.packet_fingerprint or packet.packet_fingerprint
        ),
        "response_sweep_content_hash": readiness.sweep_content_hash,
    }


def _run_ai_interpretation_action(
    *,
    copilot: Any,
    action_name: AIActionName,
    packet: AISweepPacket,
    readiness: _AIArtifactReadiness,
    selected_evidence_ref: str | None = None,
) -> dict[str, Any]:
    action_context = (
        {"evidence_ref": selected_evidence_ref}
        if selected_evidence_ref
        else None
    )
    response = run_ai_action(
        action_name=action_name,
        packet=packet,
        client=getattr(copilot, "client", None),
        model=getattr(copilot, "model", None),
        action_context=action_context,
    )
    return _build_ai_response_record(
        action_name=action_name,
        selected_evidence_ref=selected_evidence_ref,
        response=response,
        packet=packet,
        readiness=readiness,
    )


def _deterministic_key_findings(
    response: AIActionResponse,
    packet: AISweepPacket,
) -> list[str]:
    rows_by_ref = {row.evidence_ref: row for row in packet.rows}
    findings: list[str] = []
    for evidence_ref in response.evidence_refs:
        row = rows_by_ref.get(evidence_ref)
        if row is None:
            continue
        if row.low_credibility:
            findings.append(
                f"{row.evidence_ref}: masked or low-credibility cohort; review cautions."
            )
            continue
        findings.append(
            f"{row.evidence_ref}: {row.Dimensions}; "
            f"MAC={_format_ai_number(row.Sum_MAC)}, "
            f"MEC={_format_ai_number(row.Sum_MEC)}, "
            f"A/E count={_format_ai_number(row.AE_Ratio_Count)}, "
            f"A/E amount={_format_ai_number(row.AE_Ratio_Amount)}."
        )
    return findings


def _collect_ai_caution_flags(response: AIActionResponse) -> list[str]:
    flags = list(response.caution_flags)
    flags.extend(issue.code for issue in response.validation.warnings)
    flags.extend(issue.code for issue in response.validation.blocked_issues)
    return list(dict.fromkeys(flags))


def _validation_issue_messages(response: AIActionResponse) -> list[str]:
    issues = [
        *response.validation.warnings,
        *response.validation.blocked_issues,
    ]
    return [f"{issue.code}: {issue.message}" for issue in issues]


def _ai_response_freshness(
    response_record: dict[str, Any],
    packet: AISweepPacket,
    readiness: _AIArtifactReadiness,
) -> dict[str, Any]:
    freshness = compare_ai_response_freshness(
        response_record,
        readiness=readiness,
        packet_fingerprint=packet.packet_fingerprint,
    )
    return {
        "is_fresh": freshness.is_fresh,
        "mismatches": list(freshness.mismatches),
        "reason": freshness.reason,
    }


def _build_ai_response_sections(
    response_record: dict[str, Any],
    packet: AISweepPacket,
    readiness: _AIArtifactReadiness,
) -> dict[str, Any]:
    response = response_record["response"]
    freshness = _ai_response_freshness(response_record, packet, readiness)
    sweep_hash_matches_file = readiness.sweep_hash_matches_file
    if sweep_hash_matches_file is True:
        sweep_hash_status = "Manifest hash matches latest sweep file."
    elif sweep_hash_matches_file is False:
        sweep_hash_status = "Manifest hash differs from latest sweep file."
    else:
        sweep_hash_status = "Manifest/file hash comparison unavailable."

    return {
        "summary_text": response.response_text,
        "key_findings": _deterministic_key_findings(response, packet),
        "source_mode": response.source_mode,
        "evidence_refs": list(response.evidence_refs),
        "caution_flags": _collect_ai_caution_flags(response),
        "validation_issues": _validation_issue_messages(response),
        "next_review_steps": list(response.next_review_steps),
        "freshness_status": "Fresh" if freshness["is_fresh"] else "Stale",
        "freshness_mismatches": freshness["mismatches"],
        "state_fingerprint": readiness.state_fingerprint,
        "packet_fingerprint": packet.packet_fingerprint,
        "sweep_content_hash": readiness.sweep_content_hash,
        "response_state_fingerprint": response_record.get("response_state_fingerprint"),
        "response_packet_fingerprint": response_record.get("response_packet_fingerprint"),
        "response_sweep_content_hash": response_record.get("response_sweep_content_hash"),
        "sweep_hash_status": sweep_hash_status,
    }


def _render_ai_status(
    readiness: _AIArtifactReadiness,
    sections: dict[str, Any] | None = None,
) -> None:
    _require_streamlit()
    st.caption("Artifact readiness")
    readiness_items = [
        (check_name, _AI_READINESS_LABELS.get(check_name, check_name))
        for check_name in readiness.checks
    ]
    status_columns = st.columns(len(readiness_items))
    for column, (check_name, label) in zip(status_columns, readiness_items):
        with column:
            check_value = readiness.checks.get(check_name, False)
            if check_value is True:
                st.success(label)
            elif check_value is None:
                st.info(f"{label} not run")
            else:
                st.error(label)

    if sections is None:
        return

    if sections["freshness_status"] == "Fresh":
        st.success("AI response is fresh for the current sweep state.")
    else:
        st.warning(
            "AI response is stale: "
            + ", ".join(sections["freshness_mismatches"])
        )
    st.caption(sections["sweep_hash_status"])
    st.caption(f"Current state fingerprint: `{sections['state_fingerprint'] or 'unavailable'}`")
    st.caption(f"Response state fingerprint: `{sections['response_state_fingerprint'] or 'unavailable'}`")
    st.caption(f"Current packet fingerprint: `{sections['packet_fingerprint'] or 'unavailable'}`")
    st.caption(f"Response packet fingerprint: `{sections['response_packet_fingerprint'] or 'unavailable'}`")
    st.caption(f"Current sweep content hash: `{sections['sweep_content_hash'] or 'unavailable'}`")
    st.caption(f"Response sweep content hash: `{sections['response_sweep_content_hash'] or 'unavailable'}`")


def _render_ai_response(
    response_record: dict[str, Any],
    packet: AISweepPacket,
    readiness: _AIArtifactReadiness,
) -> None:
    _require_streamlit()
    sections = _build_ai_response_sections(response_record, packet, readiness)

    st.markdown("##### AI Response")
    st.caption(f"Source mode: `{sections['source_mode']}`")
    st.markdown(sections["summary_text"])

    st.markdown("###### Deterministic Key Findings")
    if sections["key_findings"]:
        for finding in sections["key_findings"]:
            st.markdown(f"- {finding}")
    else:
        st.caption("No deterministic key findings are available for the evidence refs.")

    st.markdown("###### Evidence References")
    st.markdown(
        ", ".join(f"`{ref}`" for ref in sections["evidence_refs"])
        if sections["evidence_refs"]
        else "None"
    )

    st.markdown("###### Caution Flags")
    st.markdown(
        ", ".join(f"`{flag}`" for flag in sections["caution_flags"])
        if sections["caution_flags"]
        else "None"
    )

    if sections["validation_issues"]:
        st.markdown("###### Validation Notes")
        for issue in sections["validation_issues"]:
            st.markdown(f"- {issue}")

    st.markdown("###### Next Review Steps")
    for step in sections["next_review_steps"]:
        st.markdown(f"- {step}")

    _render_ai_status(readiness, sections)


def _render_ai_interpretation_panel(copilot: UnifiedCopilot) -> None:
    _require_streamlit()
    readiness = _get_ai_panel_readiness(copilot.state)

    with st.container(border=True):
        st.markdown("##### AI Interpretation Panel")
        st.caption(
            "Interprets the latest sanitized sweep packet with evidence and freshness metadata."
        )
        _render_ai_status(readiness)

        if not readiness.ready:
            columns = st.columns(3)
            columns[0].button("Summarize Latest Sweep", disabled=True, use_container_width=True)
            columns[1].button("Explain Top Cohort", disabled=True, use_container_width=True)
            columns[2].button("Explain Selected Cohort", disabled=True, use_container_width=True)
            st.caption("Run a dimensional sweep with artifact tracking before using AI interpretation.")
            return

        try:
            packet = _build_ai_packet_for_panel(readiness)
        except (OSError, ValueError) as exc:
            st.error(f"AI packet construction failed: {exc}")
            return

        rows_by_ref = {row.evidence_ref: row for row in packet.rows}
        evidence_refs = list(rows_by_ref)
        top_evidence_ref = _select_top_cohort_evidence_ref(list(rows_by_ref.values()))
        selected_evidence_ref = None
        if evidence_refs:
            selected_evidence_ref = st.selectbox(
                "Selected Cohort",
                options=evidence_refs,
                format_func=lambda ref: _format_ai_cohort_label(rows_by_ref[ref]),
                key="ai-selected-cohort",
            )
        else:
            st.caption("No cohort rows are available in the latest sanitized AI packet.")

        columns = st.columns(3)
        response_record: dict[str, Any] | None = None
        if columns[0].button(
            "Summarize Latest Sweep",
            key="ai-summarize-sweep",
            use_container_width=True,
        ):
            response_record = _run_ai_interpretation_action(
                copilot=copilot,
                action_name="summarize_sweep",
                packet=packet,
                readiness=readiness,
            )
        if columns[1].button(
            "Explain Top Cohort",
            key="ai-explain-top-cohort",
            disabled=top_evidence_ref is None,
            use_container_width=True,
        ):
            response_record = _run_ai_interpretation_action(
                copilot=copilot,
                action_name="explain_cohort",
                packet=packet,
                readiness=readiness,
                selected_evidence_ref=top_evidence_ref,
            )
        if columns[2].button(
            "Explain Selected Cohort",
            key="ai-explain-selected-cohort",
            disabled=selected_evidence_ref is None,
            use_container_width=True,
        ):
            response_record = _run_ai_interpretation_action(
                copilot=copilot,
                action_name="explain_cohort",
                packet=packet,
                readiness=readiness,
                selected_evidence_ref=selected_evidence_ref,
            )

        if response_record:
            st.session_state["ai_interpretation_response"] = response_record

        stored_response_record = st.session_state.get("ai_interpretation_response")
        if stored_response_record:
            _render_ai_response(stored_response_record, packet, readiness)


def _ai_workflow_freshness_mismatches(
    response_record: dict[str, Any],
    readiness: _AIArtifactReadiness,
    packet_fingerprint: str | None = None,
) -> tuple[str, ...]:
    return compare_ai_response_freshness(
        response_record,
        readiness=readiness,
        packet_fingerprint=packet_fingerprint,
    ).mismatches


def _build_ai_workflow_snapshot(copilot: UnifiedCopilot) -> AIWorkflowSnapshot:
    _require_streamlit()
    try:
        readiness = _get_ai_panel_readiness(
            copilot.state,
            include_file_hash=True,
            refresh_state=False,
        )
    except (OSError, ValueError) as exc:
        return AIWorkflowSnapshot(
            ready=False,
            detail=f"AI readiness unavailable ({type(exc).__name__}).",
            basis="existing AI panel readiness",
        )

    stored_response_record = st.session_state.get("ai_interpretation_response")
    has_response = isinstance(stored_response_record, dict)
    freshness_mismatches: tuple[str, ...] = ()
    response_is_fresh: bool | None = None
    packet_build_error: Exception | None = None
    if has_response:
        if readiness.ready:
            try:
                packet = _build_ai_packet_for_panel(readiness)
            except (OSError, ValueError) as exc:
                LOGGER.warning(
                    "AI workflow packet construction failed during sidebar freshness check: %s",
                    exc,
                )
                packet_build_error = exc
                response_is_fresh = False
            else:
                freshness = compare_ai_response_freshness(
                    stored_response_record,
                    readiness=readiness,
                    packet_fingerprint=packet.packet_fingerprint,
                )
                freshness_mismatches = freshness.mismatches
                response_is_fresh = freshness.is_fresh
        else:
            response_is_fresh = False

    if readiness.ready:
        if packet_build_error is not None:
            detail = "Stored AI response freshness could not be checked against the current sweep packet."
        elif has_response and response_is_fresh:
            detail = "Stored AI response matches current sweep fingerprints."
        elif has_response and response_is_fresh is False:
            detail = None
        else:
            detail = "Existing AI panel readiness checks pass."
    else:
        missing = [
            _AI_READINESS_LABELS.get(check_name, check_name)
            for check_name, is_ready in readiness.checks.items()
            if is_ready is False
        ]
        detail = "Missing: " + ", ".join(missing) if missing else "AI panel is not ready."

    return AIWorkflowSnapshot(
        ready=readiness.ready,
        readiness_checks=dict(readiness.checks),
        has_response=has_response,
        response_is_fresh=response_is_fresh,
        freshness_mismatches=freshness_mismatches,
        detail=detail,
        basis="existing AI panel readiness",
    )


def _workflow_step_detail(step: WorkflowStep) -> str | None:
    if step.status == "completed":
        return None
    if step.status in {"blocked", "stale"} and step.prerequisite_message:
        return step.prerequisite_message
    return step.detail or step.prerequisite_message


def _render_workflow_panel(copilot: UnifiedCopilot) -> None:
    _require_streamlit()
    ai_snapshot = _build_ai_workflow_snapshot(copilot)
    steps = derive_workflow_steps(copilot.state, ai_snapshot)

    st.markdown("#### Workflow")
    for step in steps:
        status_label = _WORKFLOW_STATUS_LABELS[step.status]
        st.markdown(f"**{step.label}** · `{status_label}`")
        detail = _workflow_step_detail(step)
        if detail:
            st.caption(detail)


def _render_sidebar() -> bool:
    _require_streamlit()
    with st.sidebar:
        _render_workflow_panel(st.session_state["copilot"])
        st.markdown("---")
        st.title("Copilot Controls")
        st.caption(f"Session: `{st.session_state['session_id']}`")
        if st.button("Clear Conversation", type="primary", use_container_width=True):
            st.session_state["copilot"].reset_session()
            st.session_state["session_id"] = st.session_state["copilot"].state.session_id
            st.session_state["history"] = []
            st.session_state["ai_interpretation_response"] = None
            st.rerun()
            return True
    return False


def _consume_copilot_events(
    events: Iterable[CopilotEvent],
    *,
    status_panel,
    response_placeholder,
) -> tuple[str, str | None, list[dict[str, Any]] | None]:
    _require_streamlit()
    rendered_response = ""
    latest_visualization_path: str | None = None
    latest_sweep_results: list[dict[str, Any]] | None = None
    latest_status = "Processing request..."

    for event in events:
        if event.type == "status":
            latest_status = event.message
            status_panel.write(event.message)
            status_panel.update(label=event.message, state="running", expanded=True)
        elif event.type == "tool_start":
            status_panel.write(event.message)
        elif event.type == "tool_result":
            result = event.data.get("result", {})
            if result.get("ok", False) and result.get("kind") == "analysis":
                sweep_results = result.get("data", {}).get("results", [])
                latest_sweep_results = sweep_results or None
            if not result.get("ok", False):
                status_panel.write(event.message)
        elif event.type == "artifact_update":
            latest_visualization_path = event.data.get("latest_visualization_path") or latest_visualization_path
        elif event.type == "text_delta":
            rendered_response += event.message
            response_placeholder.markdown(rendered_response)
        elif event.type == "final":
            latest_status = "Completed request."
            payload = event.data.get("artifact_state", {})
            latest_visualization_path = (
                payload.get("latest_visualization_path") or latest_visualization_path
            )
            if not rendered_response:
                rendered_response = event.message
                response_placeholder.markdown(rendered_response)

    status_panel.update(label=latest_status, state="complete", expanded=False)
    return rendered_response, latest_visualization_path, latest_sweep_results


def render_app() -> None:
    _require_streamlit()
    st.set_page_config(
        page_title="Experience Study Copilot",
        page_icon="📊",
        layout="wide",
    )

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = UnifiedCopilot.new_session_id()
    if "copilot" not in st.session_state:
        st.session_state["copilot"] = UnifiedCopilot(
            session_id=st.session_state["session_id"]
        )
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "ai_interpretation_response" not in st.session_state:
        st.session_state["ai_interpretation_response"] = None

    if _render_sidebar():
        return

    st.title("Digital Process: Experience Study")
    st.caption(
        "Unified skill-based copilot with session-scoped deterministic artifacts and "
        "dynamic tool gating."
    )
    _render_empty_state()

    for index, item in enumerate(st.session_state["history"]):
        with st.chat_message("user"):
            st.markdown(item["prompt"])
        with st.chat_message("assistant"):
            st.markdown(item["response"])
            _render_sweep_explorer(item.get("sweep_results"))
            _render_visualization_card(
                item.get("visualization_path"),
                widget_key_prefix=f"history-{index}",
            )

    prompt = st.chat_input(
        "Ask the copilot to profile data, engineer features, run a sweep, or generate the combined report."
    )
    if prompt and prompt.strip():
        cleaned_prompt = prompt.strip()
        with st.chat_message("user"):
            st.markdown(cleaned_prompt)
        with st.chat_message("assistant"):
            status_panel = st.status("Starting copilot...", expanded=True)
            response_placeholder = st.empty()
            response, visualization_path, sweep_results = _consume_copilot_events(
                st.session_state["copilot"].process_message(cleaned_prompt),
                status_panel=status_panel,
                response_placeholder=response_placeholder,
            )
            _render_sweep_explorer(sweep_results)
            _render_visualization_card(visualization_path, widget_key_prefix="current")

        st.session_state["history"].append(
            {
                "prompt": cleaned_prompt,
                "response": response,
                "visualization_path": visualization_path,
                "sweep_results": sweep_results,
            }
        )

    _render_ai_interpretation_panel(st.session_state["copilot"])


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
