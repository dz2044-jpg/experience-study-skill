"""Validation rules for AI-generated actuarial interpretation text."""

from __future__ import annotations

import re

from skills.experience_study_skill.ai_models import (
    AIValidationIssue,
    AIValidationResult,
)


_BLOCKING_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "unsupported_causal_claim",
        "Causal conclusions are not supported by the deterministic packet.",
        re.compile(
            r"\b("
            r"caused|causes|cause of|due to|driven by|driver of|key driver|"
            r"resulted in|proves|conclusive"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "pricing_recommendation",
        "Pricing recommendations are outside the AI interpretation boundary.",
        re.compile(
            r"\b("
            r"(change|raise|lower|increase|decrease|adjust)\s+pricing|"
            r"pricing\s+(should|must|needs?)|"
            r"recommend\w*\s+\w*\s*premiums?|"
            r"increase\w*\s+premiums?|decrease\w*\s+premiums?|"
            r"rates?\s+should|rate action"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "underwriting_recommendation",
        "Underwriting strategy recommendations are outside the AI interpretation boundary.",
        re.compile(
            r"\b("
            r"underwriting\s+(should|must|needs?|tighten|loosen|change|adjust|guidelines)|"
            r"tighten\w*\s+underwriting|loosen\w*\s+underwriting"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "assumption_change_recommendation",
        "Assumption changes require actuarial review and cannot be recommended by AI.",
        re.compile(
            r"\b("
            r"assumptions?\s+(should|must|needs?|be updated|change|changed|update)|"
            r"change\s+assumptions?|assumption is inadequate"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "final_actuarial_conclusion",
        "Final actuarial conclusions and sign-off language are not allowed.",
        re.compile(r"\b(final actuarial conclusion|final conclusion|sign[- ]?off)\b", re.IGNORECASE),
    ),
]

_WARNING_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "credibility_caution",
        "Credibility caution language was detected and should be tied to evidence.",
        re.compile(r"\b(wide confidence interval|low credibility|credibility should be reviewed)\b", re.IGNORECASE),
    ),
    (
        "review_caution",
        "Review-oriented caution language was detected.",
        re.compile(r"\b(warrant actuarial review|should be reviewed|review the cohort)\b", re.IGNORECASE),
    ),
]


def validate_ai_response(text: str) -> AIValidationResult:
    """Validate generated text for blocked actuarial claims and non-blocking cautions."""

    blocked_issues = [
        AIValidationIssue(code=code, message=message)
        for code, message, pattern in _BLOCKING_PATTERNS
        if pattern.search(text)
    ]
    warnings = [
        AIValidationIssue(code=code, message=message)
        for code, message, pattern in _WARNING_PATTERNS
        if pattern.search(text)
    ]
    return AIValidationResult(
        is_valid=not blocked_issues,
        blocked_issues=blocked_issues,
        warnings=warnings,
    )
