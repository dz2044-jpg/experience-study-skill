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
        re.compile(r"\b(caused|causes|cause of|due to|driven by|driver of|resulted in)\b", re.IGNORECASE),
    ),
    (
        "pricing_recommendation",
        "Pricing recommendations are outside the AI interpretation boundary.",
        re.compile(
            r"\b((change|raise|lower|increase|decrease|adjust)\s+pricing|pricing\s+(should|must|needs?))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "underwriting_recommendation",
        "Underwriting strategy recommendations are outside the AI interpretation boundary.",
        re.compile(
            r"\b(underwriting\s+(should|must|needs?|tighten|loosen|change|adjust))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "assumption_change_recommendation",
        "Assumption changes require actuarial review and cannot be recommended by AI.",
        re.compile(
            r"\b(assumptions?\s+(should|must|needs?|be updated|change|changed|update)|change\s+assumptions?)\b",
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
