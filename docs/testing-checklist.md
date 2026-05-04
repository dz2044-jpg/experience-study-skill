# Testing Checklist

Last updated: May 4, 2026

This checklist uses expected response patterns rather than exact session paths, because generated artifact paths include session-specific folders.

The app's current intended workflow is deterministic first, then AI interpretation only on validated and sanitized outputs. That matches the improvement plan framing: deterministic workflow execution, interpretation of validated outputs, report drafting, methodology documentation, and human final sign-off remain separate.

| # | Test question / action | Expected response |
| ---: | --- | --- |
| 1 | **What can you do?** | Should return a capability summary: it can inspect schemas, profile a dataset, validate actuarial data, engineer bands/regroups, run dimensional sweeps, and generate the combined report. This matches the fallback planner's default capability response. |
| 2 | **Run a 1-way sweep on Gender.** | If no dataset has been profiled yet, expected response should say a prepared dataset is missing and ask you to profile a dataset first. |
| 3 | **Profile data/input/synthetic_inforce.csv and tell me the columns.** | Should create a session-local prepared dataset, usually `analysis_inforce.parquet`, and return a profile summary plus columns. Expected columns should include: `Policy_Number`, `Duration`, `MAC`, `MOC`, `MEC`, `MAF`, `MEF`, `COLA`, `Face_Amount`, `Issue_Age`, `Gender`, `Smoker`, `Risk_Class`. The test fixture uses these columns. |
| 4 | **Show me the schema for the current dataset.** | Should inspect the current prepared dataset first and list ordered columns with data types. The response should start like: `Columns in <prepared dataset path> (...)`. |
| 5 | **Validate the current dataset.** | Should run deterministic actuarial data checks on the prepared dataset. Expected response should confirm validation completed and report any warnings/errors, especially around required actuarial numeric fields such as `MAC`, `MOC`, `MEC`, `MAF`, `MEF`. |
| 6 | **Group Issue_Age into 3 equal-width bands.** | Should create a new derived band column from `Issue_Age`, save/update the prepared dataset, and confirm the feature-engineering step completed. |
| 7 | **Show me the schema for the current dataset.** | Expected follow-up after banding: schema should now include the new Issue Age band column, likely something like `Issue_Age_Band` or a similar derived name depending on the implementation. |
| 8 | **Run a 1-way sweep on Gender.** | Should complete a deterministic 1-way dimensional sweep, save the latest summary to `sweep_summary.csv`, and display a summary table with cohort rows like `Gender=F` and `Gender=M`. Expected columns include actual deaths, expected deaths, A/E count, A/E amount, and CI fields. The sweep supports `AE_Ratio_Count`, `AE_Ratio_Amount`, `Sum_MAC`, `Sum_MOC`, `Sum_MEC`, `Sum_MAF`, and `Sum_MEF` as ranking fields. |
| 9 | **Run a 1-way sweep on Gender where Smoker = Yes.** | Should apply the filter first, then sweep by `Gender`. Expected output should only reflect records where `Smoker` equals `Yes`. |
| 10 | **Run a 1-way sweep on Gender with at least 1 deaths.** | Should treat "at least 1 deaths" as `min_mac = 1`. Expected output should only include cohorts where `Sum_MAC >= 1`. The fallback planner explicitly parses this phrase into `min_mac`. |
| 11 | **Run a 2-way sweep on Gender and Smoker.** | Should complete a deterministic 2-way dimensional sweep and return interaction cohorts like `Gender=F` with `Smoker=No`, `Gender=M` with `Smoker=Yes`, and similar combinations. This is important to test because the sweep tool supports depth from 1 to 3. |
| 12 | **Run a 2-way sweep on Gender and Risk_Class with at least 1 deaths.** | Should run a 2-way sweep filtered by `min_mac = 1`. Expected output should include only two-dimensional cohorts with at least one actual death. |
| 13 | **Run a 2-way sweep on Gender and Smoker where Risk_Class = Standard.** | Should apply `Risk_Class = Standard` as a row filter first, then run a 2-way sweep on `Gender` and `Smoker`. Expected result should contain only Standard risk class records. |
| 14 | **Run a 2-way sweep on Gender and Face_Amount.** | Expected failure. `Face_Amount` should not be eligible as a sweep dimension because semantic numeric non-dimensions are excluded. The response should say the column is not eligible as a sweep dimension. |
| 15 | **Run a 1-way sweep on BadColumn.** | Expected failure. Response should say `BadColumn` was not found in the prepared dataset and may return available columns. |
| 16 | **Generate the combined report.** | Should generate a standalone HTML visualization from the latest sweep artifact. Expected response should confirm the combined report was generated and provide/open a visualization artifact. The Streamlit UI should also show a visualization card with "Open in Browser" and inline preview. |
| 17 | **Generate the combined report for count.** | Should generate the report using the count metric instead of amount. Expected artifact is still an HTML combined A/E report. |
| 18 | **Clear Conversation, then open AI Interpretation Panel.** | Expected: AI buttons should be disabled because there is no latest sweep artifact, artifact manifest, state fingerprint, or sweep manifest hash. The panel should show artifact readiness status. |
| 19 | **After running a sweep, use AI Interpretation Panel -> Summarize Latest Sweep.** | Expected: button should be enabled only after sweep artifact, manifest, and fingerprint are available. Response should show summary text, source mode (`fallback` or `llm`), evidence references, caution flags, next review steps, and freshness status. |
| 20 | **AI Interpretation Panel -> Explain Top Cohort.** | Expected: should select the row with the highest `AE_Ratio_Amount` and explain it using the sanitized sweep packet. Response should include deterministic key findings such as MAC, MEC, A/E count, and A/E amount. |
| 21 | **AI Interpretation Panel -> Explain Selected Cohort.** | Expected: selectbox should show cohorts using evidence refs and labels like `Dimensions`, `A/E Amount=...`, `MAC=...`, and `evidence_ref`. The response should explain only the selected cohort. |
| 22 | **Run a new sweep after an AI response, then look at the prior AI response.** | Expected: prior AI response should be marked stale if the state fingerprint, packet fingerprint, or sweep content hash changed. The AI panel explicitly compares those freshness fields. |

A good end-to-end smoke test sequence is:

```text
1. What can you do?
2. Profile data/input/synthetic_inforce.csv and tell me the columns.
3. Show me the schema for the current dataset.
4. Validate the current dataset.
5. Group Issue_Age into 3 equal-width bands.
6. Run a 1-way sweep on Gender.
7. Run a 1-way sweep on Gender where Smoker = Yes.
8. Run a 1-way sweep on Gender with at least 1 deaths.
9. Run a 2-way sweep on Gender and Smoker.
10. Run a 2-way sweep on Gender and Risk_Class with at least 1 deaths.
11. Generate the combined report.
12. Use AI Interpretation Panel:
    - Summarize Latest Sweep
    - Explain Top Cohort
    - Explain Selected Cohort
```

For PR 7 regression, the most important pass/fail check is: deterministic workflow still works normally, and the AI panel only becomes usable after the latest sweep artifact, artifact manifest, state fingerprint, and sweep hash are available.
