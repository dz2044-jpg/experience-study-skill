"""Simple Markdown action loader for internal AI interpretation prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

from skills.experience_study_skill.ai_models import AIActionName


AI_ACTION_NAMES: set[str] = {
    "summarize_sweep",
    "explain_cohort",
    "compare_cohorts",
    "analyze_count_amount_divergence",
}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass(frozen=True, slots=True)
class LoadedAIAction:
    """Loaded Markdown prompt for one internal AI action."""

    name: AIActionName
    metadata: dict[str, Any]
    body: str
    path: Path


def _parse_action_markdown(content: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(content.strip())
    if not match:
        return {}, content.strip()
    metadata = yaml.safe_load(match.group(1)) or {}
    return metadata, match.group(2).strip()


def load_ai_action(action_name: AIActionName) -> LoadedAIAction:
    """Load a known action Markdown file, failing closed for unknown actions."""

    if action_name not in AI_ACTION_NAMES:
        raise ValueError(f"Unknown AI action: {action_name}")
    action_path = Path(__file__).parent / "actions" / f"{action_name}.md"
    if not action_path.exists():
        raise FileNotFoundError(f"AI action prompt not found: {action_path}")
    metadata, body = _parse_action_markdown(action_path.read_text(encoding="utf-8"))
    return LoadedAIAction(
        name=action_name,
        metadata=metadata,
        body=body,
        path=action_path,
    )
