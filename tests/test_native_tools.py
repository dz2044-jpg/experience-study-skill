from pathlib import Path

from skills.experience_study_skill.native_tools import (
    ToolExecutionContext,
    compute_ae_ci,
    compute_ae_ci_amount,
    generate_combined_report,
)


def test_zero_claim_ci_amount_still_returns_upper_bound():
    lower, upper = compute_ae_ci_amount(
        mac=0,
        moc=1000,
        mec=4,
        actual_amount=0,
        expected_amount=400000,
    )

    assert lower is not None
    assert upper is not None
    assert upper > 0


def test_count_ci_returns_values_for_standard_case():
    lower, upper = compute_ae_ci(mac=5, moc=1000, mec=4)

    assert lower is not None
    assert upper is not None
    assert lower < upper


def test_visualization_tool_enforces_prerequisite(tmp_path: Path):
    context = ToolExecutionContext(
        session_id="session-a",
        output_dir=tmp_path / "sessions" / "session-a",
    )

    result = generate_combined_report(context=context)

    assert result["ok"] is False
    assert result["kind"] == "missing_prerequisite"
    assert "Run dimensional sweep first" in result["message"]

