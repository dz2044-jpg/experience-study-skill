# Trust Boundary

Experience Study AI Copilot uses a deliberately separated workflow boundary:

- Python owns the calculations.
- AI owns interpretation and drafting.
- Past reports guide style only.
- Humans own final judgment and sign-off.

## Deterministic Layer

The deterministic Python layer is responsible for data profiling, schema inspection, actuarial validation checks, feature engineering, dimensional A/E sweeps, confidence intervals, cohort rankings, and visualization artifacts.

A/E ratios, exposures, credibility intervals, aggregates, and output artifacts must come from deterministic tools rather than natural-language generation.

## AI Layer

The AI layer may summarize validated outputs, explain cohort-level results, draft report sections, and help describe methodology. It must ground statements in deterministic artifacts and should clearly distinguish observation from judgment.

The AI layer must not:

- Recalculate A/E metrics in free text.
- Invent assumptions, causes, or drivers.
- Recommend pricing action.
- Recommend underwriting strategy.
- Present draft language as final actuarial opinion.

## Human Review

All generated analysis and drafting are intermediate work products. An actuary must review source data quality, methodology choices, deterministic outputs, draft conclusions, and final report language before anything is used for decision-making or sign-off.
