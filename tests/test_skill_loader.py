from core.skill_loader import load_skill
from core.copilot_agent import UnifiedCopilot


def test_load_skill_returns_metadata_and_registry():
    skill = load_skill("experience-study-skill")

    assert skill.name == "experience-study-skill"
    assert "deterministic" in skill.description.lower()
    assert "actuarial" in skill.description.lower()
    assert callable(skill.tool_spec_factory)
    assert "profile_dataset" in skill.tool_handlers
    assert "run_dimensional_sweep" in skill.tool_handlers


def test_load_skill_accepts_hyphenated_and_underscored_names():
    hyphen_skill = load_skill("experience-study-skill")
    underscore_skill = load_skill("experience_study_skill")

    assert hyphen_skill.name == "experience-study-skill"
    assert underscore_skill.name == "experience-study-skill"
    assert hyphen_skill.description == underscore_skill.description
    assert hyphen_skill.instructions == underscore_skill.instructions
    assert hyphen_skill.tool_handlers.keys() == underscore_skill.tool_handlers.keys()


def test_unified_copilot_accepts_public_and_internal_skill_identifiers():
    hyphen_copilot = UnifiedCopilot(skill_name="experience-study-skill")
    underscore_copilot = UnifiedCopilot(skill_name="experience_study_skill")

    assert hyphen_copilot.active_skill.name == "experience-study-skill"
    assert underscore_copilot.active_skill.name == "experience-study-skill"
