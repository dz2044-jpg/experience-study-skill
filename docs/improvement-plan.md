# Experience Study AI Copilot Improvement Plan

Last updated: May 12, 2026

This is the canonical repository-tracked improvement plan for Experience Study AI Copilot. It replaces the local planning copy in `Downloads` as the source of truth for implementation sequencing, acceptance gates, and future PR scope.

## Purpose

Experience Study AI Copilot is an actuarial workflow copilot designed to accelerate first-draft experience study analysis.

It helps with:

- Deterministic A/E workflow execution.
- Interpretation of validated outputs.
- Report section drafting.
- Methodology documentation.
- Workflow handoff through editable and exportable artifacts.

It does not replace:

- Actuarial peer review.
- Final actuarial sign-off.
- Assumption setting.
- Pricing decisions.
- Underwriting strategy recommendations.

The actuary remains the ultimate reviewer, owner, and signatory.

## Current Status

Completed:

- The project has been renamed conceptually to Experience Study AI Copilot.
- The product direction has moved from a technical skill demo to an actuarial AI workflow copilot.
- The existing app has deterministic workflow capabilities:
  - Profile dataset.
  - Inspect schema.
  - Validate actuarial data.
  - Create bands and regroups.
  - Run dimensional A/E sweep.
  - Generate combined visualization report.
- README now reflects the Experience Study AI Copilot positioning.
- Governance docs now exist for product vision, trust boundary, data privacy, and roadmap.
- Golden workflow tests have been strengthened around artifacts, output columns, known A/E values, and stable public tool names.
- Methodology logging, artifact manifest tracking, and state fingerprinting now exist for deterministic workflow auditability.
- Internal AI packet construction, allowlist sanitization, validation, and deterministic fallback behavior now exist behind tested internal APIs.
- The Streamlit app now has an AI Interpretation Panel for summarizing the latest sweep and explaining top or selected cohorts with evidence refs, source mode, caution flags, and freshness metadata.

Important product framing:

- Python owns calculations.
- AI owns interpretation and drafting.
- Past reports guide style only.
- Humans own final judgment and sign-off.

## Progress Tracker

| PR | Workstream | Status |
| --- | --- | --- |
| PR 1 | Governance Docs, Privacy, and README Polish | Completed in current working branch |
| PR 2 | Golden Workflow Test Hardening | Completed in current working branch |
| PR 3 | Small Runtime Extraction | Completed in current working branch |
| PR 4 | Deterministic Tool Module Split | Completed in current working branch |
| PR 5 | Methodology Log and Artifact Manifest | Completed in current working branch |
| PR 6 | AI Packet, Sanitization, Validation, and Fallback Layer | Completed in current working branch |
| PR 7 | AI Interpretation UI | Completed in current working branch |
| PR 8 | Report Drafting MVP | Planned |
| PR 9 | Human Editor and Export Package | Planned |
| PR 10 | Prior Report Style Reference Layer | Planned |
| PR 11 | Guided Workflow UI | Completed early as read-only status layer |
| PR 12 | Sweep Builder + Run History | Planned |
| PR 13 | Cohort Review Queue | Planned |
| PR 14 | Artifact Center + Methodology Timeline | Planned |
| PR 15 | Structured AI Findings + Claim-Level Validation | Planned |

## Roadmap Extension: Product Workflow PRs After Report Planning

After PR 8-10, the roadmap adds five product/workflow PRs that make the app feel more like a governed actuarial workflow product rather than only a chat interface. These PRs should preserve deterministic A/E calculations, artifact names, public tool names, privacy boundaries, and existing AI fallback behavior.

The added PRs are:

1. Guided Workflow UI.
2. Sweep Builder + Run History.
3. Cohort Review Queue.
4. Artifact Center + Methodology Timeline.
5. Structured AI Findings + Claim-Level Validation.

These are planned after PR 10 so the existing report drafting, editor/export, and prior-report style reference plan remains intact.

## Guiding Principles

1. Protect the working app first.
   Do not start large refactors or AI expansion without golden workflow tests.

2. Separate refactor work from new behavior.
   Behavior-preserving module splits should not be mixed with methodology logging, AI features, or export features in the same PR.

3. Keep deterministic outputs stable.
   A/E calculations, output columns, artifact names, and public tool names should not change unless intentionally versioned.

4. Make privacy and auditability testable.
   Do not rely only on documentation. Add tests proving that raw row-level data is not sent to the LLM.

5. Add AI only after deterministic artifacts are clean and traceable.
   AI should interpret validated outputs, not create actuarial calculations.

6. Do not over-split modules too early.
   Extract the clearest seams first. Avoid creating tiny abstract modules just to make the repo look organized.

## Phase 1 - Governance Docs, Privacy, and README Polish

Goal: make the product story clear, credible, and enterprise-ready.

Add or update:

- `README.md`.
- `docs/product-vision.md`.
- `docs/trust-boundary.md`.
- `docs/data-privacy.md`.
- `docs/roadmap.md`.

README should explain:

- What Experience Study AI Copilot is.
- What workflow it supports.
- What the deterministic layer does.
- What the AI layer does.
- What the AI layer does not do.
- How humans remain responsible for final judgment.
- How to run tests.
- How to launch the app.

Acceptance gate:

- README updated away from old Experience Study Skill positioning.
- Product boundary is clear.
- Privacy boundary is explicit.
- No code behavior changes required.

## Phase 2 - Golden Workflow Test Hardening

Goal: protect the existing deterministic workflow before refactoring.

Build on the existing full-pipeline test in `tests/test_unified_copilot.py`.

Strengthen tests for:

1. Profile dataset.
2. Inspect schema.
3. Validate actuarial data.
4. Create categorical bands or regroup features.
5. Run dimensional A/E sweep.
6. Generate combined visualization report.

Expected artifact checks:

- Prepared dataset parquet exists.
- Sweep summary CSV exists.
- Combined visualization HTML exists.
- Session state correctly tracks prepared dataset.
- Session state correctly tracks latest sweep.
- Session state correctly tracks latest visualization.

Expected output checks:

- Required A/E columns exist.
- Expected confidence interval columns exist.
- Selected known A/E values match current baseline.
- Selected known row counts match current baseline.
- Public tool names remain unchanged.

Snapshot checks:

- Record expected CSV columns from the golden sweep.
- Record a few known A/E values from the sample dataset.
- Record expected artifact naming conventions.

Acceptance gate:

- All existing tests pass.
- New golden workflow tests pass.
- Deterministic outputs are unchanged.
- No public tool names changed.

## Phase 3 - Small Runtime Extraction

Status: Completed in current working branch.

Goal: reduce the size and complexity of `core/copilot_agent.py` without changing behavior.

Extract only the clearest seams first:

- `core/session_state.py`.
- `core/prerequisite_guard.py`.
- `core/fallback_planner.py`.
- `core/response_formatter.py`.

Possible later extraction targets:

- `core/intent_parser.py`.
- `core/tool_runner.py`.

Acceptance gate:

- All tests pass.
- Existing behavior unchanged.
- Existing user-facing output unchanged unless intentionally improved and approved.
- Public tool names unchanged.
- No deterministic output changes.

## Phase 4 - Deterministic Tool Module Split

Status: Completed in current working branch.

Goal: split `skills/experience_study_skill/native_tools.py` into auditable deterministic modules while preserving behavior.

Proposed structure:

- `skills/experience_study_skill/io.py`.
- `skills/experience_study_skill/validation.py`.
- `skills/experience_study_skill/feature_engineering.py`.
- `skills/experience_study_skill/ae_math.py`.
- `skills/experience_study_skill/sweeps.py`.
- `skills/experience_study_skill/visualization.py`.
- `skills/experience_study_skill/native_tools.py`.

Important:

- `native_tools.py` should remain as a compatibility layer.
- Existing public function names must remain available.
- Artifact names, output columns, and known deterministic values must remain stable.
- Do not add methodology logging in this phase.

Acceptance gate:

- All tests pass.
- Golden workflow outputs unchanged.
- Public tool names unchanged.
- Artifact names unchanged.
- Output CSV columns unchanged.
- Known A/E values unchanged.

## Phase 5 - Methodology Log and Artifact Manifest

Goal: add auditability after deterministic tools are cleanly modularized.

New files:

- `core/methodology_log.py`.
- `core/artifact_manifest.py`.

Methodology event model:

```python
@dataclass
class MethodologyEvent:
    step_name: str
    tool_name: str
    input_path: str | None
    output_path: str | None
    parameters: dict[str, object]
    timestamp: str
```

Track events for:

- Source dataset profiled.
- Schema inspected.
- Validation checks run.
- Age bands created.
- Categorical regrouping applied.
- Dimensional sweep run.
- Filters applied.
- Selected columns used.
- Sweep depth used.
- Ranking metric selected.
- Minimum MAC threshold applied.
- Visualization generated.

Artifact manifest should track:

- Artifact type.
- Path.
- Created timestamp.
- Content hash.
- Source artifact relationship.
- Generating tool.
- Relevant parameters.

Fingerprint inputs should include:

- Source artifact SHA256.
- Selected columns.
- Filters.
- Sweep depth.
- `sort_by`.
- `min_mac`.
- Packet schema version.
- Skill version, when applicable.

Acceptance gate:

- Methodology log is created during deterministic workflow.
- Artifact manifest is created or updated.
- Existing deterministic outputs unchanged.
- Tests verify methodology events are recorded for key workflow steps.
- Tests verify content hash changes when source artifact changes.

## Phase 6 - AI Packet, Sanitization, Validation, and Fallback Layer

Goal: create a safe AI foundation before adding AI buttons to the UI.

New files:

- `skills/experience_study_skill/actions/summarize_sweep.md`.
- `skills/experience_study_skill/actions/explain_cohort.md`.
- `skills/experience_study_skill/actions/compare_cohorts.md`.
- `skills/experience_study_skill/actions/analyze_count_amount_divergence.md`.
- `skills/experience_study_skill/ai_models.py`.
- `skills/experience_study_skill/ai_skill_loader.py`.
- `skills/experience_study_skill/ai_skill_renderer.py`.
- `skills/experience_study_skill/ai_packets.py`.
- `skills/experience_study_skill/ai_sanitization.py`.
- `skills/experience_study_skill/ai_baselines.py`.
- `skills/experience_study_skill/ai_validation.py`.
- `skills/experience_study_skill/ai_fallbacks.py`.
- `skills/experience_study_skill/ai_orchestrator.py`.

Use allowlist-based sanitization. Do not rely only on a denylist of sensitive columns.

Allowed AI packet fields should align with actual app output column names:

- `Dimensions`.
- `Dimension_Columns`.
- `Sum_MAC`.
- `Sum_MEC`.
- `Sum_MOC`.
- `Sum_MAF`.
- `Sum_MEF`.
- `AE_Ratio_Count`.
- `AE_Ratio_Amount`.
- `AE_Count_CI_Lower`.
- `AE_Count_CI_Upper`.
- `AE_Amount_CI_Lower`.
- `AE_Amount_CI_Upper`.
- `depth`.
- `filters`.
- `selected_columns`.
- `sort_by`.
- `min_mac`.

Strong privacy rule:

- The LLM should only receive aggregated cohort-level metrics.
- It should never receive raw row-level data.

Rare-cohort masking:

- AI interpretation uses a positive-claims default and masks cohorts with `Sum_MAC < 1` unless explicitly overridden.
- User requests using `min_mac >= N` map to threshold `N`; requests using `min_mac > N` map to threshold `N + 1` because deterministic sweep filtering uses `Sum_MAC >= min_mac`.
- Manifest `min_mac` can raise the AI masking threshold but does not lower the default below `1` unless the user explicitly requests no volume masking.
- Masking only applies to rows present in the sweep artifact; the packet builder does not invent rows already filtered out by deterministic sweep settings.
- Suppress detailed cohort labels when cohort volume is too small.
- Mark rows as low credibility.
- Prevent AI from over-interpreting low-credibility results.
- Allow only aggregate or cautionary discussion.

Sensitive dimension handling:

- Sensitive/disallowed dimension matching is token-aware, not naive substring matching.
- Dimension parsing splits on exact ` | ` first, then splits each part on the first `=`.
- Sensitive dimension matches mask the full `Dimensions` label and do not preserve detailed sensitive cohort values in the AI packet.
- The AI packet builder is not a general-purpose PII scanner; upstream workflow controls should prevent row-level identifiers from becoming benign sweep dimensions.

AI validation layer should block unsupported claims:

- Causal conclusions.
- Pricing recommendations.
- Underwriting strategy recommendations.
- Assumption changes.
- Final actuarial conclusions.

Acceptance gate:

- AI packet construction works from latest sweep artifact.
- Sanitization allowlist works.
- Rare-cohort masking works.
- No raw row-level data reaches the packet.
- Fallback response works when LLM is unavailable.
- Tests cover privacy behavior.
- Public deterministic tool names remain unchanged.

## Phase 7 - AI Interpretation UI

Goal: add MVP AI interpretation to Streamlit.

First AI actions:

- `summarize_sweep`.
- `explain_cohort`.

UI section:

- AI Interpretation Panel.

Buttons:

- Summarize Latest Sweep.
- Explain Selected Cohort.

Response should show:

- Summary text.
- Key findings.
- Caution flags.
- Next review steps.
- Evidence references.
- Source mode: LLM or fallback.
- Stale/fingerprint status.

Acceptance gate:

- AI panel only enables when required artifacts exist.
- AI response shows source mode.
- AI response shows evidence refs.
- AI response shows stale/fingerprint status.
- Existing deterministic workflow remains unaffected.

## Phase 8 - Report Drafting MVP

Goal: draft first report-ready sections using deterministic outputs and methodology events.

New skill files:

- `skills/experience_study_skill/actions/draft_mortality_overview.md`.
- `skills/experience_study_skill/actions/draft_key_findings.md`.
- `skills/experience_study_skill/actions/draft_methodology_appendix.md`.

Report packet should consume:

- Latest sweep artifact.
- Visualization artifact path.
- Top elevated cohorts.
- Top below-expected cohorts.
- Uncertainty notes.
- Methodology events.
- Artifact manifest.
- Approved style snippets, if available later.

Draft sections:

1. Mortality Overview.
2. Key Findings.
3. Methodology and Audit Trail.

Acceptance gate:

- Draft sections are generated from deterministic packet only.
- Methodology appendix uses actual methodology events.
- Draft does not invent assumptions, causes, or recommendations.
- Draft contains evidence refs or source artifact references.
- Fallback draft works if LLM is unavailable.

## Phase 9 - Human Editor and Export Package

Goal: turn AI-generated sections into an editable workflow handoff.

UI additions:

- Editable text area for mortality overview.
- Editable text area for key findings.
- Editable text area for methodology appendix.
- Session-state persistence for edited text.

Export options:

- Download edited section as Markdown.
- Download full artifact package as ZIP.
- Later: export as Word.

New file:

- `core/export_package.py`.

ZIP contents:

```text
experience_study_report_package.zip
  report_sections/
    mortality_overview.md
    key_findings.md
    methodology_appendix.md

  data/
    sweep_summary.csv

  visuals/
    combined_ae_report.html

  audit/
    methodology_log.json
    artifact_manifest.json
```

Acceptance gate:

- User edits are preserved in session state.
- Markdown export works.
- ZIP export works.
- ZIP includes expected files.
- Audit files are included.
- Export does not include raw sensitive input data unless explicitly designed and approved.

## Phase 10 - Prior Report Style Reference Layer

Goal: use prior reports for style, structure, and tone only.

Important confidentiality rule:

- Use only synthetic examples or explicitly approved internal examples.

New folder:

- `data/reference/report_style/mortality_overview_example.md`.
- `data/reference/report_style/key_findings_example.md`.
- `data/reference/report_style/credibility_commentary_example.md`.

Prompt rule:

- Use prior reports only to understand structure, tone, section ordering, and phrasing style.
- Do not copy prior-year numbers, prior conclusions, prior drivers, or verbatim language unless explicitly approved.
- Prior reports are style references, not factual sources and not text templates.

Acceptance gate:

- Style examples are synthetic or approved.
- Prompt includes no-verbatim-copy rule.
- AI output is based on current deterministic packet, not prior report facts.
- Tests or review checks confirm prior-year numbers are not copied into current report unless present in current packet.

## Phase 11 - Guided Workflow UI

Goal:
Make the app feel like a guided actuarial workflow product instead of only a chat interface.

Implementation note:
If implemented before PR 8-10, keep this phase as a read-only navigation/status layer.
Hide the report draft step until report drafting, editor/export, and prior-report style
contracts exist.

Add a workflow status panel that mirrors the real artifact lifecycle:

1. Dataset selected or profiled.
2. Schema inspected.
3. Validation completed.
4. Feature engineering completed.
5. Latest sweep available.
6. Visualization available.
7. AI interpretation available.
8. Report draft available when PR 8 has been completed.

Suggested UI behavior:

- Show each step as Not started, Ready, Completed, Blocked, or Stale.
- Use existing session state, artifact state, methodology log, artifact manifest, and fingerprint metadata where possible.
- Keep the current chat interface available.
- Add clear prerequisite messages for blocked steps.
- Add a compact sidebar or top-of-page workflow progress component.
- Do not change deterministic calculations, artifact names, public tool names, or output columns.

Acceptance gate:

- Workflow panel renders from current session/artifact state.
- Completed, blocked, and stale states are visible to users.
- Existing chat workflow still works.
- AI Interpretation Panel behavior remains unchanged.
- Tests cover workflow state derivation where feasible without requiring Streamlit browser testing.

## Phase 12 - Sweep Builder + Run History

Goal:
Make dimensional sweep setup reproducible through structured UI controls and preserve sweep scenarios for comparison and reuse.

Add a structured Sweep Builder panel for:

- Sweep depth: 1-way and 2-way.
- Dimension column selection.
- Optional filters.
- Minimum MAC threshold.
- Ranking metric such as AE_Ratio_Amount or AE_Ratio_Count.
- Top-N display setting if useful for UI display only.

Add run history support:

- Store each completed sweep as a session-scoped scenario.
- Track scenario label, timestamp, selected columns, filters, depth, sort_by, min_mac, sweep artifact path, state fingerprint, and content hash.
- Allow the user to mark one scenario as the active/baseline scenario.
- Allow the user to restore a prior sweep scenario as the latest selected result when the artifact still exists.

Important:

- Sweep Builder should call existing deterministic sweep behavior rather than duplicating A/E logic.
- Chat-based sweep requests should continue to work.
- Run history should not store raw source data.

Acceptance gate:

- User can run at least a 1-way and 2-way sweep from structured controls.
- Completed sweep scenarios appear in run history.
- Restoring or selecting a prior scenario updates the UI context without changing the underlying artifact.
- Existing chat-based sweep workflow still works.
- Tests verify scenario metadata is recorded without raw row-level data.

## Phase 13 - Cohort Review Queue

Goal:
Turn sweep results into an explicit actuarial review workflow.

Create a session-scoped review queue from the latest active sweep scenario.

Suggested queue categories:

- Highest A/E Amount.
- Highest A/E Count.
- Largest count-vs-amount divergence.
- High exposure with unusual result.
- Low credibility or masked cohorts.
- Below-expected cohorts.
- Cohorts already used in AI interpretation.

Suggested user actions:

- Mark cohort as reviewed.
- Mark cohort as needs follow-up.
- Add reviewer note.
- Add cohort to report candidate findings.
- Remove cohort from report candidate findings.
- Trigger existing explain_cohort action for a selected cohort.

Important:

- Review queue state should be session-scoped.
- Review notes should not mutate deterministic artifacts.
- Review status should reference evidence refs and artifact fingerprints.
- Review queue should support later report drafting by identifying candidate findings.

Acceptance gate:

- Review queue is generated from the active sweep packet or sweep artifact.
- User can mark review status for evidence refs.
- User can identify candidate report findings.
- Existing AI explanation flow can use the selected review queue cohort.
- Stale queue state is detectable when the active sweep fingerprint changes.

## Phase 14 - Artifact Center + Methodology Timeline

Goal:
Make auditability visible and useful inside the app.

Add an Artifact Center that lists session artifacts:

- Prepared dataset artifact.
- Latest sweep summary CSV.
- Combined visualization HTML.
- Methodology log JSON.
- Artifact manifest JSON.
- Future report draft/export artifacts.

For each artifact, show:

- Artifact type.
- Path or basename.
- Generating tool.
- Created timestamp where available.
- Content hash where available.
- Fresh/stale status when comparable.
- Download or open action where appropriate.

Add a Methodology Timeline:

- Render methodology events in chronological order.
- Show step name, tool name, selected parameters, and output artifact reference.
- Keep technical JSON available in an expander rather than making it the primary UX.
- Use existing methodology_log.json and artifact_manifest.json structures.
- Missing audit files should produce helpful messages rather than errors.

Acceptance gate:

- Artifact Center renders from existing artifact state and manifest data.
- Methodology Timeline renders from the methodology log.
- Missing audit files produce helpful messages rather than errors.
- No raw sensitive source data is exposed by default.
- Tests cover artifact/timeline parsing helpers where feasible.

## Phase 15 - Structured AI Findings + Claim-Level Validation

Goal:
Make AI outputs easier to review, test, and reuse in report drafting.

Extend AI response contracts from mostly free text into structured findings.

Suggested model:

```python
class AIFinding(BaseModel):
    finding_text: str
    evidence_refs: list[str]
    metrics_used: list[str]
    severity: Literal["info", "review", "important"]
    caveats: list[str]
    report_ready: bool
```

Here, claim-level validation means validation of individual AI assertions, not exposure of claim-level source records.

Suggested behavior:

- AI interpretation and report drafting should return structured findings alongside narrative text where useful.
- Each finding should cite evidence refs from sanitized packets or current artifacts.
- Each finding should identify the metrics used to support the statement.
- Unsupported, stale, low-credibility, or heavily caveated findings should not be marked report-ready.
- Fallback AI responses should use the same structured finding contract where feasible.
- Structured findings should not include raw row-level data, claim-level records, policy-level records, or sensitive identifiers.

Acceptance gate:

- Structured findings parse and validate with explicit evidence refs and metrics_used values.
- Invalid or missing evidence refs are rejected or clearly flagged.
- Unsupported AI assertions are flagged before report drafting uses them.
- Report drafting can consume report-ready findings without relying on free-text parsing.
- Fallback responses preserve the structured contract where feasible.
- Privacy tests or review checks confirm structured findings do not expose raw source records or sensitive identifiers.

## Updated PR Plan

1. Governance Docs, Privacy, and README Polish.
   Product story, privacy rules, and human sign-off boundary.

2. Golden Workflow Test Hardening.
   Stronger deterministic workflow tests before refactor work.

3. Small Runtime Extraction.
   Behavior-preserving extraction from the copilot runtime.

4. Deterministic Tool Module Split.
   Behavior-preserving split of deterministic tools into focused modules.

5. Methodology Log and Artifact Manifest.
   Audit events and content-hash based artifact tracking.

6. AI Packet, Sanitization, Validation, and Fallback Layer.
   Allowlisted cohort-level packets, privacy tests, and fallback behavior.

7. AI Interpretation UI.
   Streamlit AI interpretation panel gated by deterministic artifacts.

8. Report Drafting MVP.
   Draft report sections from deterministic packets and methodology events.

9. Human Editor and Export Package.
   Editable text and exportable handoff package.

10. Prior Report Style Reference Layer.
    Style-only prior report examples with no-verbatim-copy rules.

11. Guided Workflow UI.
    Workflow status panel that mirrors artifact lifecycle and freshness.

12. Sweep Builder + Run History.
    Structured sweep setup and session-scoped scenario history.

13. Cohort Review Queue.
    Explicit actuarial review workflow for sweep-derived cohorts.

14. Artifact Center + Methodology Timeline.
    In-app auditability for artifacts and methodology events.

15. Structured AI Findings + Claim-Level Validation.
    Evidence-backed AI finding objects with assertion-level validation.

## Final MVP Definition

Existing deterministic capabilities:

1. Profile dataset.
2. Inspect schema.
3. Validate actuarial data.
4. Engineer bands and regroups.
5. Run dimensional A/E sweep.
6. Generate combined visualization report.

New governance capabilities:

7. Product vision document.
8. Trust boundary document.
9. Data privacy document.
10. Roadmap document.
11. README with product framing and run/test instructions.

New audit capabilities:

12. Methodology log.
13. Artifact manifest.
14. Content-hash based state fingerprint.

New AI capabilities:

15. Summarize latest sweep.
16. Explain selected cohort.
17. Draft mortality overview.
18. Draft key findings.
19. Generate methodology appendix.

New workflow handoff capabilities:

20. Human-edit drafted text.
21. Export report sections.
22. Export full ZIP package.
23. Include methodology log and artifact manifest.

## Immediate Next Move

PR 1, PR 2, PR 3, PR 4, PR 5, PR 6, and PR 7 are complete in the current working branch. The next implementation slice is PR 8: Report Drafting MVP.

Do not start workflow export or prior-report style reference work until the PR 8 report drafting MVP is completed and all tests still pass.

PR 11-15 are planned after PR 10 so the existing report drafting, editor/export, and prior-report style reference sequence remains intact.
