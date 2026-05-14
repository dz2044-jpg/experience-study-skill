"""Baseline wording and review steps for deterministic AI fallbacks."""

from __future__ import annotations

from skills.experience_study_skill.ai_models import AIActionName


ACTION_TITLES: dict[AIActionName, str] = {
    "summarize_sweep": "Sweep Summary",
    "explain_cohort": "Cohort Explanation",
    "compare_cohorts": "Cohort Comparison",
    "analyze_count_amount_divergence": "Count/Amount Divergence",
}

DEFAULT_NEXT_REVIEW_STEPS = [
    "Review masked or low-credibility cohorts before using them as findings.",
    "Confirm deterministic sweep settings, filters, and selected dimensions.",
    "Have an actuary review interpretation before report use or sign-off.",
]
