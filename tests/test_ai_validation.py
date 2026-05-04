from skills.experience_study_skill.ai_validation import validate_ai_response


def test_validation_blocks_unsupported_actuarial_claims():
    result = validate_ai_response(
        "This caused mortality to increase. We should change pricing and underwriting should tighten rules."
    )

    assert result.is_valid is False
    codes = {issue.code for issue in result.blocked_issues}
    assert "unsupported_causal_claim" in codes
    assert "pricing_recommendation" in codes
    assert "underwriting_recommendation" in codes


def test_validation_blocks_common_business_language_variants():
    result = validate_ai_response(
        "We recommend increasing premiums. Rates should be adjusted. "
        "This is a key driver of mortality. "
        "This proves the assumption is inadequate. "
        "Underwriting guidelines should be tightened."
    )

    assert result.is_valid is False
    codes = {issue.code for issue in result.blocked_issues}
    assert "unsupported_causal_claim" in codes
    assert "pricing_recommendation" in codes
    assert "underwriting_recommendation" in codes
    assert "assumption_change_recommendation" in codes


def test_validation_allows_evidence_grounded_observations_with_warnings():
    result = validate_ai_response(
        "This cohort shows elevated A/E on an amount basis. "
        "The confidence interval is wide, so credibility should be reviewed. "
        "Count and amount A/E diverge, which may warrant actuarial review."
    )

    assert result.is_valid is True
    assert result.blocked_issues == []
    assert {warning.code for warning in result.warnings} == {
        "credibility_caution",
        "review_caution",
    }
