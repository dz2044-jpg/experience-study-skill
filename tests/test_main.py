from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from core.artifact_manifest import file_sha256, upsert_artifact_entry
from core.copilot_agent import CopilotEvent
import main
from skills.experience_study_skill.ai_models import (
    AIActionResponse,
    AICohortRow,
    AISweepPacket,
    AIValidationIssue,
    AIValidationResult,
)


class _FakeStatusPanel:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.updates: list[dict[str, object]] = []

    def write(self, message: str) -> None:
        self.writes.append(message)

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class _FakeResponsePlaceholder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def markdown(self, message: str) -> None:
        self.messages.append(message)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


class _TrackingContext(_NullContext):
    def __init__(self, parent: "_FakeAIPanelStreamlit", label: str) -> None:
        self.parent = parent
        self.label = label

    def __enter__(self):
        self.parent.context_stack.append(self.label)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.parent.context_stack.pop()
        return False


class _FakeColumn(_NullContext):
    def __init__(self, parent: "_FakeAIPanelStreamlit") -> None:
        self.parent = parent

    def button(self, label: str, **kwargs: object) -> bool:
        return self.parent.button(label, **kwargs)


class _FakeStreamlitApp:
    def __init__(self, *, prompt: str | None, copilot: object) -> None:
        self.prompt = prompt
        self.chat_message_calls: list[dict[str, object]] = []
        self.session_state: dict[str, object] = {
            "session_id": "session-a",
            "copilot": copilot,
            "history": [],
            "ai_interpretation_response": None,
        }

    def set_page_config(self, **kwargs) -> None:
        return None

    def title(self, message: str) -> None:
        return None

    def caption(self, message: str) -> None:
        return None

    def markdown(self, message: str) -> None:
        return None

    def chat_input(self, message: str) -> str | None:
        return self.prompt

    def chat_message(self, role: str, **kwargs: object) -> _NullContext:
        self.chat_message_calls.append({"role": role, **kwargs})
        return _NullContext()

    def status(self, message: str, *, expanded: bool) -> _FakeStatusPanel:
        return _FakeStatusPanel()

    def empty(self) -> _FakeResponsePlaceholder:
        return _FakeResponsePlaceholder()


class _FakeVisualizationStreamlit:
    def __init__(self) -> None:
        self.button_calls: list[dict[str, object]] = []
        self.download_calls: list[dict[str, object]] = []
        self.column_specs: list[list[int]] = []
        self.preview_html: str | None = None
        self.components = SimpleNamespace(
            v1=SimpleNamespace(html=self._render_component_html)
        )

    def container(self, *, border: bool) -> _NullContext:
        return _NullContext()

    def markdown(self, message: str) -> None:
        return None

    def caption(self, message: str) -> None:
        return None

    def columns(self, spec: list[int]) -> list[_NullContext]:
        self.column_specs.append(spec)
        return [_NullContext() for _ in spec]

    def button(self, label: str, **kwargs: object) -> bool:
        self.button_calls.append({"label": label, **kwargs})
        return False

    def download_button(self, label: str, **kwargs: object) -> None:
        self.download_calls.append({"label": label, **kwargs})

    def expander(self, label: str, *, expanded: bool) -> _NullContext:
        return _NullContext()

    def _render_component_html(
        self,
        html: str,
        *,
        height: int,
        scrolling: bool,
    ) -> None:
        self.preview_html = html


class _FakeAIStatusStreamlit:
    def __init__(self) -> None:
        self.success_calls: list[str] = []
        self.info_calls: list[str] = []
        self.error_calls: list[str] = []
        self.caption_calls: list[str] = []

    def caption(self, message: str) -> None:
        self.caption_calls.append(message)

    def columns(self, count: int) -> list[_NullContext]:
        return [_NullContext() for _ in range(count)]

    def success(self, message: str) -> None:
        self.success_calls.append(message)

    def info(self, message: str) -> None:
        self.info_calls.append(message)

    def error(self, message: str) -> None:
        self.error_calls.append(message)


class _FakeAIPanelStreamlit:
    def __init__(
        self,
        *,
        copilot: object | None = None,
        button_results: dict[str, bool] | None = None,
    ) -> None:
        self.button_results = button_results or {}
        self.session_state: dict[str, object] = {
            "session_id": "session-a",
            "copilot": copilot or SimpleNamespace(),
            "history": [],
            "ai_interpretation_response": None,
        }
        self.sidebar = _TrackingContext(self, "sidebar")
        self.context_stack: list[str] = []
        self.button_calls: list[dict[str, object]] = []
        self.caption_calls: list[dict[str, object]] = []
        self.dialog_calls: list[dict[str, object]] = []
        self.expander_calls: list[dict[str, object]] = []
        self.markdown_calls: list[dict[str, object]] = []
        self.selectbox_calls: list[dict[str, object]] = []
        self.status_calls: list[dict[str, object]] = []

    def _context(self) -> tuple[str, ...]:
        return tuple(self.context_stack)

    def container(self, *, border: bool) -> _TrackingContext:
        return _TrackingContext(self, "container")

    def expander(self, label: str, *, expanded: bool) -> _TrackingContext:
        self.expander_calls.append(
            {"label": label, "expanded": expanded, "context": self._context()}
        )
        return _TrackingContext(self, label)

    def columns(self, spec: int | list[int]) -> list[_FakeColumn]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_FakeColumn(self) for _ in range(count)]

    def button(self, label: str, **kwargs: object) -> bool:
        self.button_calls.append({"label": label, "context": self._context(), **kwargs})
        return self.button_results.get(label, False)

    def dialog(self, title: str, *, width: str):
        def decorator(func):
            def wrapped(*args, **kwargs):
                self.dialog_calls.append({"title": title, "width": width})
                return func(*args, **kwargs)

            return wrapped

        return decorator

    def selectbox(
        self,
        label: str,
        *,
        options: list[str],
        format_func,
        key: str,
    ) -> str | None:
        self.selectbox_calls.append(
            {
                "label": label,
                "options": options,
                "key": key,
                "context": self._context(),
            }
        )
        return options[0] if options else None

    def caption(self, message: str) -> None:
        self.caption_calls.append({"message": message, "context": self._context()})

    def error(self, message: str) -> None:
        self.status_calls.append(
            {"kind": "error", "message": message, "context": self._context()}
        )

    def info(self, message: str) -> None:
        self.status_calls.append(
            {"kind": "info", "message": message, "context": self._context()}
        )

    def markdown(self, message: str) -> None:
        self.markdown_calls.append({"message": message, "context": self._context()})

    def success(self, message: str) -> None:
        self.status_calls.append(
            {"kind": "success", "message": message, "context": self._context()}
        )

    def title(self, message: str) -> None:
        self.markdown_calls.append({"message": message, "context": self._context()})

    def warning(self, message: str) -> None:
        self.status_calls.append(
            {"kind": "warning", "message": message, "context": self._context()}
        )

    def rerun(self) -> None:
        self.status_calls.append({"kind": "rerun", "message": "", "context": self._context()})


class _FakeRenderCopilot:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.state = SimpleNamespace(panel_ready=False)

    def process_message(self, user_input: str) -> list[CopilotEvent]:
        self.calls.append(f"process:{user_input}")
        self.state.panel_ready = True
        return [
            CopilotEvent(
                "final",
                message="done",
                data={"artifact_state": {}},
            )
        ]


class _GeneratorRenderCopilot(_FakeRenderCopilot):
    def process_message(self, user_input: str):
        self.calls.append(f"process:{user_input}")

        def events():
            self.state.panel_ready = True
            yield CopilotEvent(
                "final",
                message="done",
                data={"artifact_state": {}},
            )

        return events()


def _write_ai_sweep(path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "Dimensions": "Gender=M",
                "Sum_MAC": 2,
                "Sum_MOC": 10.0,
                "Sum_MEC": 1.0,
                "Sum_MAF": 100000.0,
                "Sum_MEF": 80000.0,
                "AE_Ratio_Count": 2.0,
                "AE_Ratio_Amount": 1.25,
                "AE_Count_CI_Lower": 0.25,
                "AE_Count_CI_Upper": 3.5,
                "AE_Amount_CI_Lower": 0.30,
                "AE_Amount_CI_Upper": 3.75,
            },
            {
                "Dimensions": "Region=West",
                "Sum_MAC": 4,
                "Sum_MOC": 7.0,
                "Sum_MEC": 1.4,
                "Sum_MAF": 120000.0,
                "Sum_MEF": 40000.0,
                "AE_Ratio_Count": 0.8,
                "AE_Ratio_Amount": 3.0,
                "AE_Count_CI_Lower": 0.15,
                "AE_Count_CI_Upper": 2.0,
                "AE_Amount_CI_Lower": 0.55,
                "AE_Amount_CI_Upper": 4.5,
            },
        ]
    ).to_csv(path, index=False)
    return path


def _write_sweep_manifest(manifest_path: Path, sweep_path: Path) -> Path:
    upsert_artifact_entry(
        manifest_path,
        artifact_type="sweep_summary",
        path=sweep_path,
        generating_tool="run_dimensional_sweep",
        parameters={"depth": 1},
        source_artifacts=[],
    )
    return manifest_path


def _refresh_noop() -> None:
    return None


def test_ai_workflow_snapshot_uses_readiness_without_building_packet(monkeypatch) -> None:
    readiness_calls: dict[str, bool] = {}
    monkeypatch.setattr(
        main,
        "st",
        SimpleNamespace(session_state={"ai_interpretation_response": None}),
    )

    def fake_get_ai_panel_readiness(
        state: object,
        *,
        include_file_hash: bool = True,
        refresh_state: bool = True,
    ) -> main._AIArtifactReadiness:
        readiness_calls["include_file_hash"] = include_file_hash
        readiness_calls["refresh_state"] = refresh_state
        return main._AIArtifactReadiness(
            checks={
                "latest_sweep": True,
                "artifact_manifest": True,
                "state_fingerprint": True,
                "sweep_manifest_hash": True,
                "sweep_hash_matches_file": True,
            },
            state_fingerprint="state-a",
            sweep_content_hash="hash-a",
        )

    monkeypatch.setattr(
        main,
        "_get_ai_panel_readiness",
        fake_get_ai_panel_readiness,
    )

    def fail_if_packet_is_built(*args: object, **kwargs: object) -> None:
        raise AssertionError("Workflow sidebar must not build AI packets.")

    monkeypatch.setattr(main, "_build_ai_packet_for_panel", fail_if_packet_is_built)

    snapshot = main._build_ai_workflow_snapshot(SimpleNamespace(state=object()))

    assert snapshot.ready is True
    assert snapshot.has_response is False
    assert snapshot.basis == "existing AI panel readiness"
    assert readiness_calls == {"include_file_hash": True, "refresh_state": False}


def test_ai_workflow_snapshot_marks_stored_response_stale_on_known_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        main,
        "st",
        SimpleNamespace(
            session_state={
                "ai_interpretation_response": {
                    "response_state_fingerprint": "state-a",
                    "response_packet_fingerprint": "packet-a",
                    "response_sweep_content_hash": "hash-a",
                }
            }
        ),
    )
    monkeypatch.setattr(
        main,
        "_get_ai_panel_readiness",
        lambda state, include_file_hash=True, refresh_state=True: main._AIArtifactReadiness(
            checks={
                "latest_sweep": True,
                "artifact_manifest": True,
                "state_fingerprint": True,
                "sweep_manifest_hash": True,
                "sweep_hash_matches_file": True,
            },
            state_fingerprint="state-b",
            sweep_content_hash="hash-b",
        ),
    )
    monkeypatch.setattr(
        main,
        "_build_ai_packet_for_panel",
        lambda readiness: _example_packet(
            state_fingerprint="state-b",
            packet_fingerprint="packet-b",
        ),
    )

    snapshot = main._build_ai_workflow_snapshot(SimpleNamespace(state=object()))

    assert snapshot.ready is True
    assert snapshot.has_response is True
    assert snapshot.response_is_fresh is False
    assert snapshot.freshness_mismatches == (
        "state fingerprint",
        "packet fingerprint",
        "sweep content hash",
    )


def test_ai_workflow_snapshot_marks_stored_response_stale_when_sweep_state_is_cleared(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        main,
        "st",
        SimpleNamespace(
            session_state={
                "ai_interpretation_response": {
                    "response_state_fingerprint": "state-a",
                    "response_packet_fingerprint": "packet-a",
                    "response_sweep_content_hash": "hash-a",
                }
            }
        ),
    )
    monkeypatch.setattr(
        main,
        "_get_ai_panel_readiness",
        lambda state, include_file_hash=True, refresh_state=True: main._AIArtifactReadiness(
            checks={
                "latest_sweep": False,
                "artifact_manifest": True,
                "state_fingerprint": False,
                "sweep_manifest_hash": False,
                "sweep_hash_matches_file": False,
            },
            state_fingerprint=None,
            sweep_content_hash=None,
        ),
    )

    snapshot = main._build_ai_workflow_snapshot(SimpleNamespace(state=object()))

    assert snapshot.ready is False
    assert snapshot.has_response is True
    assert snapshot.response_is_fresh is False
    assert snapshot.freshness_mismatches == ()


def _example_packet(
    *,
    state_fingerprint: str = "state-a",
    packet_fingerprint: str = "packet-a",
) -> AISweepPacket:
    return AISweepPacket(
        source_artifact_path="sweep_summary.csv",
        source_content_hash="hash-a",
        state_fingerprint=state_fingerprint,
        packet_fingerprint=packet_fingerprint,
        rows=[
            AICohortRow(
                evidence_ref="row_0001",
                Dimensions="Gender=M",
                Dimension_Columns=["Gender"],
                Sum_MAC=2.0,
                Sum_MOC=10.0,
                Sum_MEC=1.0,
                Sum_MAF=100000.0,
                Sum_MEF=80000.0,
                AE_Ratio_Count=2.0,
                AE_Ratio_Amount=1.25,
            ),
            AICohortRow(
                evidence_ref="row_0002",
                Dimensions="[masked cohort label]",
                Dimension_Columns=["Gender"],
                Sum_MAC=0.0,
                Sum_MOC=8.0,
                Sum_MEC=1.2,
                Sum_MAF=0.0,
                Sum_MEF=85000.0,
                AE_Ratio_Count=0.0,
                AE_Ratio_Amount=0.0,
                low_credibility=True,
                masking_reason="low_volume",
                caution_flags=["low_volume"],
            ),
        ],
    )


def test_ai_workflow_snapshot_compares_packet_fingerprint_when_response_exists(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        main,
        "st",
        SimpleNamespace(
            session_state={
                "ai_interpretation_response": {
                    "response_state_fingerprint": "state-a",
                    "response_packet_fingerprint": "packet-a",
                    "response_sweep_content_hash": "hash-a",
                }
            }
        ),
    )
    monkeypatch.setattr(
        main,
        "_get_ai_panel_readiness",
        lambda state, include_file_hash=True, refresh_state=True: main._AIArtifactReadiness(
            checks={
                "latest_sweep": True,
                "artifact_manifest": True,
                "state_fingerprint": True,
                "sweep_manifest_hash": True,
                "sweep_hash_matches_file": True,
            },
            state_fingerprint="state-a",
            sweep_content_hash="hash-a",
        ),
    )
    monkeypatch.setattr(
        main,
        "_build_ai_packet_for_panel",
        lambda readiness: _example_packet(packet_fingerprint="packet-b"),
    )

    snapshot = main._build_ai_workflow_snapshot(SimpleNamespace(state=object()))

    assert snapshot.ready is True
    assert snapshot.has_response is True
    assert snapshot.response_is_fresh is False
    assert snapshot.freshness_mismatches == ("packet fingerprint",)


def test_ai_workflow_snapshot_handles_packet_build_failure(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        main,
        "st",
        SimpleNamespace(
            session_state={
                "ai_interpretation_response": {
                    "response_state_fingerprint": "state-a",
                    "response_packet_fingerprint": "packet-a",
                    "response_sweep_content_hash": "hash-a",
                }
            }
        ),
    )
    monkeypatch.setattr(
        main,
        "_get_ai_panel_readiness",
        lambda state, include_file_hash=True, refresh_state=True: main._AIArtifactReadiness(
            checks={
                "latest_sweep": True,
                "artifact_manifest": True,
                "state_fingerprint": True,
                "sweep_manifest_hash": True,
                "sweep_hash_matches_file": True,
            },
            state_fingerprint="state-a",
            sweep_content_hash="hash-a",
        ),
    )
    monkeypatch.setattr(
        main,
        "_build_ai_packet_for_panel",
        lambda readiness: (_ for _ in ()).throw(ValueError("bad packet")),
    )

    snapshot = main._build_ai_workflow_snapshot(SimpleNamespace(state=object()))

    assert snapshot.ready is True
    assert snapshot.has_response is True
    assert snapshot.response_is_fresh is False
    assert snapshot.freshness_mismatches == ()
    assert snapshot.detail == (
        "Stored AI response freshness could not be checked against the current sweep packet."
    )
    assert "AI workflow packet construction failed" in caplog.text


def test_render_visualization_card_offers_html_download_and_inline_preview(
    tmp_path: Path,
    monkeypatch,
) -> None:
    html_path = tmp_path / "combined_ae_report.html"
    html_path.write_text("<html><body>report</body></html>", encoding="utf-8")
    fake_st = _FakeVisualizationStreamlit()
    monkeypatch.setattr(main, "st", fake_st)

    main._render_visualization_card(str(html_path), widget_key_prefix="unit")

    assert fake_st.column_specs == [[1, 3]]
    assert fake_st.button_calls == []
    assert fake_st.download_calls == [
        {
            "label": "Download HTML",
            "data": html_path.read_bytes(),
            "file_name": html_path.name,
            "mime": "text/html",
            "key": "unit-download-visualization",
        }
    ]
    assert fake_st.preview_html == "<html><body>report</body></html>"


def test_render_ai_status_marks_skipped_hash_check_as_not_run(monkeypatch) -> None:
    fake_st = _FakeAIStatusStreamlit()
    monkeypatch.setattr(main, "st", fake_st)
    readiness = main._AIArtifactReadiness(
        checks={
            "latest_sweep": True,
            "artifact_manifest": True,
            "state_fingerprint": True,
            "sweep_manifest_hash": True,
            "sweep_hash_matches_file": None,
        }
    )

    main._render_ai_status(readiness)

    assert fake_st.info_calls == ["Manifest/file hash match not run"]
    assert "Manifest/file hash match" not in fake_st.error_calls
    assert readiness.ready is True


def test_chat_avatar_constants_use_requested_emoji_pair() -> None:
    assert main.USER_CHAT_AVATAR == "🔎"
    assert main.ASSISTANT_CHAT_AVATAR == "🧑‍💻"


def test_render_app_does_not_render_ai_panel_after_current_chat_processing(monkeypatch) -> None:
    calls: list[str] = []
    copilot = _FakeRenderCopilot(calls)
    fake_st = _FakeStreamlitApp(prompt=" Run sweep now ", copilot=copilot)

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(main, "_render_sidebar", lambda: False)
    monkeypatch.setattr(main, "_render_empty_state", lambda: calls.append("empty_state"))
    monkeypatch.setattr(main, "_render_sweep_explorer", lambda sweep_results: None)
    monkeypatch.setattr(
        main,
        "_render_visualization_card",
        lambda visualization_path, widget_key_prefix: None,
    )

    def fake_render_ai_panel(panel_copilot) -> None:
        calls.append(f"ai_panel:{panel_copilot.state.panel_ready}")

    monkeypatch.setattr(main, "_render_ai_interpretation_panel", fake_render_ai_panel)

    main.render_app()

    assert calls == ["empty_state", "process:Run sweep now"]
    assert fake_st.chat_message_calls == [
        {"role": "user", "avatar": main.USER_CHAT_AVATAR},
        {"role": "assistant", "avatar": main.ASSISTANT_CHAT_AVATAR},
    ]
    assert fake_st.session_state["history"] == [
        {
            "prompt": "Run sweep now",
            "response": "done",
            "visualization_path": None,
            "sweep_results": None,
        }
    ]


def test_render_app_passes_copilot_event_iterable_without_materializing(monkeypatch) -> None:
    calls: list[str] = []
    copilot = _GeneratorRenderCopilot(calls)
    fake_st = _FakeStreamlitApp(prompt=" Run sweep now ", copilot=copilot)

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(main, "_render_sidebar", lambda: False)
    monkeypatch.setattr(main, "_render_empty_state", lambda: None)
    monkeypatch.setattr(main, "_render_sweep_explorer", lambda sweep_results: None)
    monkeypatch.setattr(
        main,
        "_render_visualization_card",
        lambda visualization_path, widget_key_prefix: None,
    )
    monkeypatch.setattr(main, "_render_ai_interpretation_panel", lambda panel_copilot: None)

    def fake_consume(events, *, status_panel, response_placeholder):
        calls.append(f"events_is_list:{isinstance(events, list)}")
        rendered_events = list(events)
        return rendered_events[-1].message, None, None

    monkeypatch.setattr(main, "_consume_copilot_events", fake_consume)

    main.render_app()

    assert calls == ["process:Run sweep now", "events_is_list:False"]


def test_render_app_does_not_render_ai_panel_without_prompt(monkeypatch) -> None:
    calls: list[str] = []
    copilot = _FakeRenderCopilot(calls)
    fake_st = _FakeStreamlitApp(prompt=None, copilot=copilot)

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(main, "_render_sidebar", lambda: False)
    monkeypatch.setattr(main, "_render_empty_state", lambda: calls.append("empty_state"))

    def fake_render_ai_panel(panel_copilot) -> None:
        calls.append(f"ai_panel:{panel_copilot.state.panel_ready}")

    monkeypatch.setattr(main, "_render_ai_interpretation_panel", fake_render_ai_panel)

    main.render_app()

    assert calls == ["empty_state"]
    assert fake_st.session_state["history"] == []


def test_sidebar_ai_launcher_opens_dialog(monkeypatch) -> None:
    calls: list[str] = []
    copilot = SimpleNamespace()
    fake_st = _FakeAIPanelStreamlit(
        copilot=copilot,
        button_results={"Open AI Interpretation": True},
    )

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(
        main,
        "_render_workflow_panel",
        lambda panel_copilot: calls.append(f"workflow:{panel_copilot is copilot}"),
    )
    monkeypatch.setattr(
        main,
        "_render_ai_interpretation_panel",
        lambda panel_copilot: calls.append(f"ai_panel:{panel_copilot is copilot}"),
    )

    should_stop = main._render_sidebar()

    assert should_stop is False
    assert calls == ["workflow:True", "ai_panel:True"]
    assert fake_st.dialog_calls == [{"title": "AI Interpretation", "width": "large"}]
    assert [call["label"] for call in fake_st.button_calls] == [
        "Open AI Interpretation",
        "Clear Conversation",
    ]


def test_ai_panel_renders_only_summary_and_selected_actions(monkeypatch) -> None:
    fake_st = _FakeAIPanelStreamlit()
    readiness = main._AIArtifactReadiness(
        checks={
            "latest_sweep": True,
            "artifact_manifest": True,
            "state_fingerprint": True,
            "sweep_manifest_hash": True,
            "sweep_hash_matches_file": True,
        },
        sweep_path=Path("sweep_summary.csv"),
        artifact_manifest_path=Path("artifact_manifest.json"),
        state_fingerprint="state-a",
        sweep_content_hash="hash-a",
    )

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(main, "_get_ai_panel_readiness", lambda state: readiness)
    monkeypatch.setattr(main, "_build_ai_packet_for_panel", lambda current: _example_packet())

    main._render_ai_interpretation_panel(SimpleNamespace(state=object()))

    button_labels = [call["label"] for call in fake_st.button_calls]
    assert button_labels == ["Summarize Latest Sweep", "Explain Selected Cohort"]
    assert "Explain Top Cohort" not in button_labels
    assert fake_st.selectbox_calls == [
        {
            "label": "Selected Cohort",
            "options": ["row_0001", "row_0002"],
            "key": "ai-selected-cohort",
            "context": ("container",),
        }
    ]
    assert fake_st.expander_calls == [
        {
            "label": "Artifact readiness details",
            "expanded": False,
            "context": ("container",),
        }
    ]
    readiness_status_calls = [
        call
        for call in fake_st.status_calls
        if call["message"] in main._AI_READINESS_LABELS.values()
    ]
    assert readiness_status_calls
    assert all(
        "Artifact readiness details" in call["context"]
        for call in readiness_status_calls
    )


def test_empty_state_suggestion_uses_existing_sample_dataset() -> None:
    suggestions = "\n".join(message for _, message in main.EMPTY_STATE_SUGGESTIONS)

    assert "data/input/synthetic_inforce.csv" in suggestions
    assert "data/input/example.csv" not in suggestions


def test_consume_copilot_events_keeps_fallback_status_only(monkeypatch) -> None:
    monkeypatch.setattr(main, "st", object())
    status_panel = _FakeStatusPanel()
    response_placeholder = _FakeResponsePlaceholder()
    final_message = "Columns in `/tmp/analysis_inforce.parquet` (2):\n- `Policy_Number`: `string`"
    success_tool_message = "Inspected the schema for `/tmp/analysis_inforce.parquet`."

    response, visualization_path, sweep_results = main._consume_copilot_events(
        [
            CopilotEvent("status", message="Copilot received a new request."),
            CopilotEvent(
                "status",
                message="OpenAI is unavailable. Using deterministic local planning.",
            ),
            CopilotEvent("tool_start", message="Executing `inspect_dataset_schema`."),
            CopilotEvent(
                "tool_result",
                message=success_tool_message,
                data={"result": {"ok": True, "kind": "schema"}},
            ),
            CopilotEvent("text_delta", message=final_message),
            CopilotEvent("final", message=final_message, data={"artifact_state": {}}),
        ],
        status_panel=status_panel,
        response_placeholder=response_placeholder,
    )

    assert "OpenAI is unavailable. Using deterministic local planning." in status_panel.writes
    assert success_tool_message not in status_panel.writes
    assert response == final_message
    assert visualization_path is None
    assert sweep_results is None


def test_consume_copilot_events_captures_analysis_rows(monkeypatch) -> None:
    monkeypatch.setattr(main, "st", object())
    status_panel = _FakeStatusPanel()
    response_placeholder = _FakeResponsePlaceholder()
    rows = [
        {
            "Dimensions": "Gender=M",
            "Sum_MAC": 2,
            "Sum_MOC": 3.7,
            "Sum_MEC": 0.789,
            "Sum_MAF": 270000.0,
            "Sum_MEF": 317000.0,
            "AE_Ratio_Count": 2.5316,
            "AE_Ratio_Amount": 0.8517,
            "AE_Count_CI_Lower": 0.8,
            "AE_Count_CI_Upper": 4.2,
            "AE_Amount_CI_Lower": 0.2,
            "AE_Amount_CI_Upper": 1.5,
        }
    ]

    _, _, sweep_results = main._consume_copilot_events(
        [
            CopilotEvent("status", message="Copilot received a new request."),
            CopilotEvent(
                "tool_result",
                message="Completed a 1-way dimensional sweep.",
                data={"result": {"ok": True, "kind": "analysis", "data": {"results": rows}}},
            ),
            CopilotEvent("final", message="done", data={"artifact_state": {}}),
        ],
        status_panel=status_panel,
        response_placeholder=response_placeholder,
    )

    assert sweep_results == rows


def test_consume_copilot_events_accepts_generators(monkeypatch) -> None:
    monkeypatch.setattr(main, "st", object())
    status_panel = _FakeStatusPanel()
    response_placeholder = _FakeResponsePlaceholder()

    def events():
        yield CopilotEvent("status", message="working")
        yield CopilotEvent("text_delta", message="done")
        yield CopilotEvent("final", message="done", data={"artifact_state": {}})

    response, _, _ = main._consume_copilot_events(
        events(),
        status_panel=status_panel,
        response_placeholder=response_placeholder,
    )

    assert response == "done"
    assert response_placeholder.messages == ["done"]


def test_build_sweep_display_frame_renames_and_rounds() -> None:
    display_df = main._build_sweep_display_frame(
        [
            {
                "Dimensions": "Risk_Class=Standard",
                "Sum_MAC": 14,
                "Sum_MOC": 10.135,
                "Sum_MEC": 1.384,
                "Sum_MAF": 120000.123,
                "Sum_MEF": 95000.567,
                "AE_Ratio_Count": 10.115,
                "AE_Ratio_Amount": 1.789,
                "AE_Count_CI_Lower": 0.904,
                "AE_Count_CI_Upper": 2.226,
                "AE_Amount_CI_Lower": 0.377,
                "AE_Amount_CI_Upper": 2.444,
            }
        ]
    )

    assert list(display_df.columns) == [
        "Cohort Dimension",
        "Actual Deaths (MAC)",
        "Exposure (MOC)",
        "Expected (MEC)",
        "Actual Amount (MAF)",
        "Expected Amount (MEF)",
        "A/E Ratio (Count)",
        "A/E Ratio (Amount)",
        "Count CI Lower",
        "Count CI Upper",
        "Amount CI Lower",
        "Amount CI Upper",
    ]
    row = display_df.iloc[0].to_dict()
    assert row["Cohort Dimension"] == "Risk_Class=Standard"
    assert row["Actual Deaths (MAC)"] == 14.0
    assert row["Expected (MEC)"] == 1.38
    assert row["A/E Ratio (Count)"] == 10.12
    assert row["Amount CI Upper"] == 2.44
    assert isinstance(display_df, pd.DataFrame)


def test_build_sweep_display_frame_formats_invalid_values_as_na() -> None:
    display_df = main._build_sweep_display_frame(
        [
            {
                "Dimensions": "Segment=Invalid",
                "Sum_MAC": 1,
                "Sum_MOC": 1.0,
                "Sum_MEC": 0.0,
                "Sum_MAF": 100000.0,
                "Sum_MEF": 0.0,
                "AE_Ratio_Count": None,
                "AE_Ratio_Amount": float("inf"),
                "AE_Count_CI_Lower": "",
                "AE_Count_CI_Upper": float("nan"),
                "AE_Amount_CI_Lower": None,
                "AE_Amount_CI_Upper": None,
            }
        ]
    )

    row = display_df.iloc[0].to_dict()
    assert row["A/E Ratio (Count)"] == "n/a"
    assert row["A/E Ratio (Amount)"] == "n/a"
    assert row["Count CI Lower"] == "n/a"
    assert row["Count CI Upper"] == "n/a"


def test_ai_panel_readiness_reports_missing_artifacts_and_manifest_hash(tmp_path: Path) -> None:
    empty_state = SimpleNamespace(
        latest_sweep_path=None,
        artifact_manifest_path=None,
        latest_state_fingerprint=None,
        refresh=_refresh_noop,
    )

    empty_readiness = main._get_ai_panel_readiness(empty_state)

    assert empty_readiness.checks == {
        "latest_sweep": False,
        "artifact_manifest": False,
        "state_fingerprint": False,
        "sweep_manifest_hash": False,
        "sweep_hash_matches_file": False,
    }
    assert empty_readiness.ready is False

    sweep_path = _write_ai_sweep(tmp_path / "sweep_summary.csv")
    missing_manifest_state = SimpleNamespace(
        latest_sweep_path=sweep_path,
        artifact_manifest_path=None,
        latest_state_fingerprint="state-a",
        refresh=_refresh_noop,
    )

    missing_manifest_readiness = main._get_ai_panel_readiness(missing_manifest_state)

    assert missing_manifest_readiness.checks["latest_sweep"] is True
    assert missing_manifest_readiness.checks["artifact_manifest"] is False
    assert missing_manifest_readiness.checks["state_fingerprint"] is True
    assert missing_manifest_readiness.checks["sweep_manifest_hash"] is False
    assert missing_manifest_readiness.checks["sweep_hash_matches_file"] is False

    manifest_path = _write_sweep_manifest(
        tmp_path / "audit" / "artifact_manifest.json",
        sweep_path,
    )
    ready_state = SimpleNamespace(
        latest_sweep_path=sweep_path,
        artifact_manifest_path=manifest_path,
        latest_state_fingerprint="state-a",
        refresh=_refresh_noop,
    )

    ready = main._get_ai_panel_readiness(ready_state)

    assert ready.ready is True
    assert ready.sweep_content_hash == file_sha256(sweep_path)
    assert ready.sweep_hash_matches_file is True
    assert ready.checks["sweep_hash_matches_file"] is True
    assert main._manifest_content_hash_for_path(manifest_path, sweep_path) == file_sha256(
        sweep_path
    )


def test_ai_packet_build_is_blocked_when_manifest_file_hash_differs(
    tmp_path: Path,
) -> None:
    sweep_path = _write_ai_sweep(tmp_path / "sweep_summary.csv")
    manifest_path = _write_sweep_manifest(
        tmp_path / "audit" / "artifact_manifest.json",
        sweep_path,
    )
    sweep_path.write_text(sweep_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    readiness = main._get_ai_panel_readiness(
        SimpleNamespace(
            latest_sweep_path=sweep_path,
            artifact_manifest_path=manifest_path,
            latest_state_fingerprint="state-a",
            refresh=_refresh_noop,
        )
    )

    with pytest.raises(ValueError, match="Manifest/file hash match"):
        main._build_ai_packet_for_panel(readiness)


def test_ai_workflow_snapshot_blocks_on_manifest_file_hash_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sweep_path = _write_ai_sweep(tmp_path / "sweep_summary.csv")
    manifest_path = _write_sweep_manifest(
        tmp_path / "audit" / "artifact_manifest.json",
        sweep_path,
    )
    sweep_path.write_text(sweep_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(
        main,
        "st",
        SimpleNamespace(session_state={"ai_interpretation_response": None}),
    )

    snapshot = main._build_ai_workflow_snapshot(
        SimpleNamespace(
            state=SimpleNamespace(
                latest_sweep_path=sweep_path,
                artifact_manifest_path=manifest_path,
                latest_state_fingerprint="state-a",
                refresh=_refresh_noop,
            )
        )
    )

    assert snapshot.ready is False
    assert snapshot.readiness_checks["sweep_hash_matches_file"] is False
    assert "Manifest/file hash match" in snapshot.detail


def test_format_ai_cohort_label_is_readable_but_evidence_ref_stable() -> None:
    row = SimpleNamespace(
        evidence_ref="row_0007",
        Dimensions="Gender=M | Risk_Class=Preferred",
        AE_Ratio_Amount=1.421,
        Sum_MAC=18,
    )

    label = main._format_ai_cohort_label(row)

    assert "Gender=M | Risk_Class=Preferred" in label
    assert "A/E Amount=1.42" in label
    assert "MAC=18.00" in label
    assert "row_0007" in label


def test_ai_response_sections_render_metadata_and_stale_status() -> None:
    packet = _example_packet()
    response = AIActionResponse(
        action_name="explain_cohort",
        source_mode="fallback",
        response_text="AI summary text should stay separate from deterministic facts.",
        evidence_refs=["row_0001"],
        caution_flags=["low_volume"],
        next_review_steps=["Review the evidence ref."],
        state_fingerprint="state-a",
        packet_fingerprint="packet-a",
        validation=AIValidationResult(
            warnings=[
                AIValidationIssue(
                    code="review_caution",
                    message="Review credibility before report use.",
                )
            ],
        ),
    )
    readiness = main._AIArtifactReadiness(
        checks={
            "latest_sweep": True,
            "artifact_manifest": True,
            "state_fingerprint": True,
            "sweep_manifest_hash": True,
            "sweep_hash_matches_file": True,
        },
        state_fingerprint="state-a",
        sweep_content_hash="hash-a",
        actual_sweep_content_hash="hash-a",
    )
    record = main._build_ai_response_record(
        action_name="explain_cohort",
        selected_evidence_ref="row_0001",
        response=response,
        packet=packet,
        readiness=readiness,
    )

    sections = main._build_ai_response_sections(record, packet, readiness)

    assert sections["summary_text"] == response.response_text
    assert sections["source_mode"] == "fallback"
    assert sections["evidence_refs"] == ["row_0001"]
    assert sections["key_findings"] == [
        "row_0001: Gender=M; MAC=2.00, MEC=1.00, A/E count=2.00, A/E amount=1.25."
    ]
    assert sections["caution_flags"] == ["low_volume", "review_caution"]
    assert sections["validation_issues"] == [
        "review_caution: Review credibility before report use."
    ]
    assert sections["next_review_steps"] == ["Review the evidence ref."]
    assert sections["freshness_status"] == "Fresh"
    assert sections["state_fingerprint"] == "state-a"
    assert sections["packet_fingerprint"] == "packet-a"
    assert sections["sweep_content_hash"] == "hash-a"
    assert sections["sweep_hash_status"] == "Manifest hash matches latest sweep file."

    stale_readiness = main._AIArtifactReadiness(
        checks=readiness.checks,
        state_fingerprint="state-b",
        sweep_content_hash="hash-b",
        actual_sweep_content_hash="hash-b",
    )
    stale_packet = _example_packet(
        state_fingerprint="state-b",
        packet_fingerprint="packet-b",
    )

    stale_sections = main._build_ai_response_sections(record, stale_packet, stale_readiness)

    assert stale_sections["freshness_status"] == "Stale"
    assert stale_sections["freshness_mismatches"] == [
        "state fingerprint",
        "packet fingerprint",
        "sweep content hash",
    ]


def test_ai_interpretation_action_falls_back_without_configured_client(tmp_path: Path) -> None:
    sweep_path = _write_ai_sweep(tmp_path / "sweep_summary.csv")
    manifest_path = _write_sweep_manifest(
        tmp_path / "audit" / "artifact_manifest.json",
        sweep_path,
    )
    readiness = main._get_ai_panel_readiness(
        SimpleNamespace(
            latest_sweep_path=sweep_path,
            artifact_manifest_path=manifest_path,
            latest_state_fingerprint="state-a",
            refresh=_refresh_noop,
        )
    )
    packet = main._build_ai_packet_for_panel(readiness)

    record = main._run_ai_interpretation_action(
        copilot=SimpleNamespace(client=None, model="test-model"),
        action_name="summarize_sweep",
        packet=packet,
        readiness=readiness,
    )

    assert record["response"].source_mode == "fallback"
    assert record["action_name"] == "summarize_sweep"
    assert record["response_sweep_content_hash"] == file_sha256(sweep_path)


def test_ai_panel_path_uses_packet_builder_and_orchestrator(monkeypatch) -> None:
    packet = _example_packet()
    response = AIActionResponse(
        action_name="explain_cohort",
        source_mode="llm",
        response_text="Cohort explanation.",
        evidence_refs=["row_0001"],
        state_fingerprint="state-a",
        packet_fingerprint="packet-a",
    )
    calls: dict[str, object] = {}

    def fake_build_latest_sweep_packet(**kwargs):
        calls["build_packet"] = kwargs
        return packet

    def fake_run_ai_action(**kwargs):
        calls["run_ai_action"] = kwargs
        return response

    monkeypatch.setattr(main, "build_latest_sweep_packet", fake_build_latest_sweep_packet)
    monkeypatch.setattr(main, "run_ai_action", fake_run_ai_action)
    readiness = main._AIArtifactReadiness(
        checks={
            "latest_sweep": True,
            "artifact_manifest": True,
            "state_fingerprint": True,
            "sweep_manifest_hash": True,
            "sweep_hash_matches_file": True,
        },
        sweep_path=Path("sweep_summary.csv"),
        artifact_manifest_path=Path("artifact_manifest.json"),
        state_fingerprint="state-a",
        sweep_content_hash="hash-a",
    )

    built_packet = main._build_ai_packet_for_panel(readiness)
    record = main._run_ai_interpretation_action(
        copilot=SimpleNamespace(client=object(), model="test-model"),
        action_name="explain_cohort",
        packet=built_packet,
        readiness=readiness,
        selected_evidence_ref="row_0001",
    )

    assert calls["build_packet"] == {
        "sweep_path": readiness.sweep_path,
        "artifact_manifest_path": readiness.artifact_manifest_path,
    }
    run_call = calls["run_ai_action"]
    assert run_call["action_name"] == "explain_cohort"
    assert run_call["packet"] is packet
    assert run_call["client"] is not None
    assert run_call["model"] == "test-model"
    assert run_call["action_context"] == {"evidence_ref": "row_0001"}
    assert record["response"] is response
