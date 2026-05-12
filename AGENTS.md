# AGENTS.md

## Project Orientation

Experience Study AI Copilot is a uv-managed Streamlit actuarial copilot for first-draft experience study analysis. Follow the Python version declared by the repository; `pyproject.toml` currently requires `>=3.13`.

Important paths:

- `main.py`: Streamlit app entry point.
- `core/`: shared copilot orchestration, session state, artifacts, model config, and fallback planning.
- `skills/experience_study_skill/`: deterministic actuarial tools, AI packet builders, validation, sweeps, schemas, visualization, and AI orchestration.
- `docs/`: product, trust-boundary, privacy, roadmap, and testing documentation.
- `tests/`: pytest coverage for app behavior, schemas, deterministic tools, AI packets, validation, and session state.
- `data/input/`: sample input data, including `synthetic_inforce.csv`.

## Standard Commands

```bash
uv sync
uv run pytest -q
uv run streamlit run main.py
```

For documentation-only changes, no automated tests are required. Do not run the full test suite unless the change is bundled with code changes or the user requests verification.

## Working Rules

- Before editing, inspect the relevant files and current behavior first. Do not guess architecture, filenames, APIs, or test commands.
- Prefer small, focused, reviewable changes that preserve existing naming, style, folder structure, and project patterns.
- Add or update targeted tests when behavior changes.
- Keep deterministic business logic separate from AI reasoning, prompts, interpretation, and generated text.
- Use standard Python `logging` for diagnostics unless the project already uses another logging pattern. Use Streamlit UI messages for user-facing feedback.
- Do not introduce dependencies unless they clearly simplify the implementation or improve maintainability. If a Python dependency is needed, suggest the appropriate `uv add <package>` command.

## Architecture Guardrails

- Python owns deterministic calculations, A/E metrics, confidence intervals, validation checks, feature engineering, sweeps, and artifact generation.
- AI owns interpretation and drafting only after deterministic outputs have been validated and sanitized.
- Past reports and examples may guide style only; they must not override deterministic outputs.
- Humans own final actuarial judgment, review, sign-off, assumption setting, pricing decisions, and underwriting strategy decisions.

Do not change deterministic outputs unless explicitly requested. Treat these as stable public behavior:

- A/E calculation logic.
- Output CSV columns.
- Artifact names and artifact manifest behavior.
- Public tool names and user-facing workflow commands.
- Confidence interval column names.
- Golden workflow baselines and documented testing expectations.

## Privacy And AI Safety

- Never send raw row-level data, policy-level records, claim-level records, or sensitive identifiers to any LLM-backed layer.
- Sensitive fields include names, DOBs, addresses, emails, phone numbers, SSNs, `Policy_Number`, and any unknown or unapproved source columns.
- AI packets should use allowlist-based sanitization. A denylist alone is not sufficient.
- Low-volume or low-credibility cohorts should be masked, suppressed, or clearly flagged before AI interpretation.
- Privacy boundaries must be enforced by code and tests, not documentation alone.

## Git And File Safety

- Check the current branch and uncommitted changes before editing.
- Do not overwrite user work or revert changes you did not make.
- Do not create commits, switch branches, rebase, reset, force-push, delete files, or run destructive git commands unless explicitly requested.
- Summarize changed files, tests run, and any remaining risks or follow-up items after implementation.
