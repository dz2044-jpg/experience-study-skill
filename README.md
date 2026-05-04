# Experience Study AI Copilot

Experience Study AI Copilot is an actuarial workflow copilot for first-draft experience study analysis. It helps actuaries move from source data to deterministic Actual-to-Expected (A/E) outputs, interpretation-ready exhibits, and report-ready draft language while preserving a clear human review boundary.

The app currently supports deterministic data profiling, schema inspection, actuarial data validation, feature engineering, dimensional A/E sweeps, and combined visualization reports. Future phases add methodology logs, artifact manifests, sanitized AI interpretation packets, report drafting, and export packages.

## Trust Boundary

Experience Study AI Copilot separates deterministic calculation, AI drafting, prior-report style reference, and human judgment:

- Python owns calculations.
- AI owns interpretation and drafting.
- Past reports guide style only.
- Humans own final judgment and sign-off.

The AI layer does not replace actuarial peer review, final actuarial sign-off, assumption setting, pricing decisions, or underwriting strategy recommendations. The actuary remains the ultimate reviewer, owner, and signatory.

See [docs/trust-boundary.md](docs/trust-boundary.md) for the full boundary.

## Supported Workflow

The deterministic workflow supports:

- Profiling tabular source data.
- Inspecting dataset schemas.
- Running actuarial data checks.
- Creating categorical bands and regrouped features.
- Running dimensional A/E sweeps.
- Generating combined A/E visualization reports.

The Streamlit app provides a chat-style interface over this workflow and stores generated artifacts in session-scoped output directories.

## AI And Privacy Direction

The planned AI layer should interpret only validated, aggregated, cohort-level outputs. It must not receive raw row-level data, policy-level rows, claim-level rows, names, DOBs, addresses, emails, phone numbers, SSNs, or unapproved source columns.

Future AI packet builders should use allowlist-based sanitization and rare-cohort masking before any LLM-backed interpretation or drafting.

See [docs/data-privacy.md](docs/data-privacy.md) for the privacy rules.

## Project Docs

- [Product vision](docs/product-vision.md)
- [Trust boundary](docs/trust-boundary.md)
- [Data privacy](docs/data-privacy.md)
- [Roadmap](docs/roadmap.md)
- [Improvement plan](docs/improvement-plan.md)

## Setup

This project uses `uv` and Python 3.13.

```bash
uv sync
```

## Run Tests

```bash
uv run pytest -q
```

## Launch The App

```bash
uv run streamlit run main.py
```

Then open the Streamlit URL printed in the terminal. A sample input file is available at `data/input/synthetic_inforce.csv`.

## Repository Notes

Some internal package and skill identifiers still use `experience-study-skill` for compatibility with the existing runtime. The product-facing name is Experience Study AI Copilot.
