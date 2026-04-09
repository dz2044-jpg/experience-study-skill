"""Load skill packages using standard Python imports."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import re
from typing import Any, Callable

import yaml


ToolSpecFactory = Callable[[set[str] | None], list[dict[str, Any]]]
ToolHandler = Callable[[dict[str, Any], Any], dict[str, Any]]


@dataclass(slots=True)
class LoadedSkill:
    """Runtime representation of a loaded skill package."""

    name: str
    description: str
    version: str
    instructions: str
    tool_spec_factory: ToolSpecFactory
    tool_handlers: dict[str, ToolHandler]
    tool_context_type: type[Any]


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_skill_markdown(content: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from markdown instructions."""
    match = _FRONTMATTER_RE.match(content.strip())
    if not match:
        raise ValueError("skill.md is missing valid YAML frontmatter.")

    metadata = yaml.safe_load(match.group(1)) or {}
    instructions = match.group(2).strip()
    return metadata, instructions


def load_skill(skill_name: str) -> LoadedSkill:
    """Load a skill's markdown metadata, schemas, and native tools."""
    skill_package = f"skills.{skill_name}"
    skill_path = Path("skills") / skill_name / "skill.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill markdown not found: {skill_path}")

    metadata, instructions = _parse_skill_markdown(
        skill_path.read_text(encoding="utf-8")
    )
    schemas_module = importlib.import_module(f"{skill_package}.schemas")
    native_tools_module = importlib.import_module(f"{skill_package}.native_tools")

    if not hasattr(schemas_module, "get_tool_specs"):
        raise AttributeError(f"{skill_package}.schemas must define get_tool_specs().")
    if not hasattr(native_tools_module, "get_tool_handlers"):
        raise AttributeError(
            f"{skill_package}.native_tools must define get_tool_handlers()."
        )
    if not hasattr(native_tools_module, "ToolExecutionContext"):
        raise AttributeError(
            f"{skill_package}.native_tools must define ToolExecutionContext."
        )

    return LoadedSkill(
        name=str(metadata.get("name", skill_name)),
        description=str(metadata.get("description", "")),
        version=str(metadata.get("version", "0.0.0")),
        instructions=instructions,
        tool_spec_factory=schemas_module.get_tool_specs,
        tool_handlers=native_tools_module.get_tool_handlers(),
        tool_context_type=native_tools_module.ToolExecutionContext,
    )
