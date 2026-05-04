from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from skills.experience_study_skill.ai_fallbacks import build_fallback_response
from skills.experience_study_skill.ai_orchestrator import run_ai_action
from skills.experience_study_skill.ai_packets import build_latest_sweep_packet
from skills.experience_study_skill.ai_skill_renderer import render_action_prompt


def _write_sweep(path: Path) -> Path:
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
                "Dimensions": "Gender=F",
                "Sum_MAC": 0,
                "Sum_MOC": 8.0,
                "Sum_MEC": 1.2,
                "Sum_MAF": 0.0,
                "Sum_MEF": 85000.0,
                "AE_Ratio_Count": 0.0,
                "AE_Ratio_Amount": 0.0,
                "AE_Count_CI_Lower": 0.0,
                "AE_Count_CI_Upper": 1.2,
                "AE_Amount_CI_Lower": 0.0,
                "AE_Amount_CI_Upper": 1.5,
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
            {
                "Dimensions": "Region=East",
                "Sum_MAC": 3,
                "Sum_MOC": 9.0,
                "Sum_MEC": 2.25,
                "Sum_MAF": 30000.0,
                "Sum_MEF": 60000.0,
                "AE_Ratio_Count": 4.0,
                "AE_Ratio_Amount": 0.5,
                "AE_Count_CI_Lower": 0.75,
                "AE_Count_CI_Upper": 6.0,
                "AE_Amount_CI_Lower": 0.10,
                "AE_Amount_CI_Upper": 1.25,
            },
        ]
    ).to_csv(path, index=False)
    return path


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.chat = SimpleNamespace(completions=self)
        self.text = text

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))]
        )


def test_fallback_response_includes_required_review_context(tmp_path: Path):
    packet = build_latest_sweep_packet(sweep_path=_write_sweep(tmp_path / "sweep.csv"))

    response = build_fallback_response(action_name="summarize_sweep", packet=packet)

    assert response.source_mode == "fallback"
    assert response.evidence_refs
    assert "low_volume" in response.caution_flags
    assert response.next_review_steps
    assert response.packet_fingerprint == packet.packet_fingerprint
    assert "Source mode: fallback." in response.response_text
    assert "Evidence refs:" in response.response_text
    assert "Caution flags:" in response.response_text
    assert "Next review steps:" in response.response_text
    assert "Packet fingerprint:" in response.response_text


def test_orchestrator_falls_back_without_client(tmp_path: Path):
    packet = build_latest_sweep_packet(sweep_path=_write_sweep(tmp_path / "sweep.csv"))

    response = run_ai_action(action_name="explain_cohort", packet=packet)

    assert response.source_mode == "fallback"
    assert response.action_name == "explain_cohort"


def test_orchestrator_uses_llm_when_response_has_only_warnings(tmp_path: Path):
    packet = build_latest_sweep_packet(sweep_path=_write_sweep(tmp_path / "sweep.csv"))
    client = _FakeClient(
        "This cohort shows elevated A/E on an amount basis. "
        "The confidence interval is wide, so credibility should be reviewed."
    )

    response = run_ai_action(
        action_name="summarize_sweep",
        packet=packet,
        client=client,
    )

    assert response.source_mode == "llm"
    assert response.validation.is_valid is True
    assert response.validation.warnings


def test_llm_success_evidence_refs_follow_action_context(tmp_path: Path):
    packet = build_latest_sweep_packet(sweep_path=_write_sweep(tmp_path / "sweep.csv"))
    client = _FakeClient("This cohort shows elevated A/E on an amount basis.")

    response = run_ai_action(
        action_name="explain_cohort",
        packet=packet,
        client=client,
        action_context={"evidence_ref": "row_0002"},
    )

    assert response.source_mode == "llm"
    assert response.evidence_refs == ["row_0002"]


def test_llm_success_compare_evidence_refs_are_selected_rows(tmp_path: Path):
    packet = build_latest_sweep_packet(sweep_path=_write_sweep(tmp_path / "sweep.csv"))
    client = _FakeClient("The selected cohorts differ on an amount basis.")

    response = run_ai_action(
        action_name="compare_cohorts",
        packet=packet,
        client=client,
    )

    assert response.source_mode == "llm"
    assert response.evidence_refs == ["row_0003", "row_0001"]
    assert response.evidence_refs != [row.evidence_ref for row in packet.rows]


def test_llm_success_divergence_evidence_refs_are_selected_rows(tmp_path: Path):
    packet = build_latest_sweep_packet(sweep_path=_write_sweep(tmp_path / "sweep.csv"))
    client = _FakeClient("These cohorts show count and amount A/E divergence.")

    response = run_ai_action(
        action_name="analyze_count_amount_divergence",
        packet=packet,
        client=client,
    )

    assert response.source_mode == "llm"
    assert response.evidence_refs == ["row_0004", "row_0003", "row_0001"]
    assert response.evidence_refs != [row.evidence_ref for row in packet.rows]


def test_rendered_prompt_uses_safe_source_artifact_ref(tmp_path: Path):
    sweep_path = _write_sweep(tmp_path / "sweep_summary.csv")
    packet = build_latest_sweep_packet(sweep_path=sweep_path)

    prompt = render_action_prompt(action_name="summarize_sweep", packet=packet)

    assert str(sweep_path) not in prompt
    assert "source_artifact_path" not in prompt
    assert '"source_artifact_ref": "sweep_summary.csv"' in prompt


def test_orchestrator_falls_back_when_llm_response_is_blocked(tmp_path: Path):
    packet = build_latest_sweep_packet(sweep_path=_write_sweep(tmp_path / "sweep.csv"))
    client = _FakeClient("This caused mortality to increase, so we should change pricing.")

    response = run_ai_action(
        action_name="summarize_sweep",
        packet=packet,
        client=client,
    )

    assert response.source_mode == "fallback"
    assert response.validation.is_valid is False
    assert response.validation.blocked_issues
