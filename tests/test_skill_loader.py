from core.skill_loader import load_skill


def test_load_skill_returns_metadata_and_registry():
    skill = load_skill("experience_study_skill")

    assert skill.name == "experience_study_skill"
    assert "deterministic" in skill.description.lower()
    assert "actuarial" in skill.description.lower()
    assert callable(skill.tool_spec_factory)
    assert "profile_dataset" in skill.tool_handlers
    assert "run_dimensional_sweep" in skill.tool_handlers
