"""Shared fixtures for the unified experience study test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def clear_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force tests through the deterministic fallback path."""
    for env_var in (
        "OPENAI_API_KEY",
        "OPENAI_COPILOT_MODEL",
        "OPENAI_ACTUARY_MODEL",
        "OPENAI_STEWARD_MODEL",
        "OPENAI_ROUTER_MODEL",
    ):
        monkeypatch.delenv(env_var, raising=False)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Synthetic inforce dataset with enough variance for sweeps and validation."""
    return pd.DataFrame(
        [
            {
                "Policy_Number": "P001",
                "Duration": 1,
                "MAC": 1,
                "MOC": 1.0,
                "MEC": 0.20,
                "MAF": 100000.0,
                "MEF": 80000.0,
                "COLA": 100000.0,
                "Face_Amount": 250000,
                "Issue_Age": 42,
                "Gender": "F",
                "Smoker": "No",
                "Risk_Class": "Preferred",
            },
            {
                "Policy_Number": "P002",
                "Duration": 1,
                "MAC": 0,
                "MOC": 0.80,
                "MEC": 0.15,
                "MAF": 0.0,
                "MEF": 70000.0,
                "COLA": None,
                "Face_Amount": 200000,
                "Issue_Age": 51,
                "Gender": "M",
                "Smoker": "Yes",
                "Risk_Class": "Standard",
            },
            {
                "Policy_Number": "P003",
                "Duration": 1,
                "MAC": 0,
                "MOC": 1.0,
                "MEC": 0.18,
                "MAF": 0.0,
                "MEF": 65000.0,
                "COLA": None,
                "Face_Amount": 180000,
                "Issue_Age": 36,
                "Gender": "F",
                "Smoker": "No",
                "Risk_Class": "Preferred Plus",
            },
            {
                "Policy_Number": "P004",
                "Duration": 1,
                "MAC": 1,
                "MOC": 1.0,
                "MEC": 0.22,
                "MAF": 150000.0,
                "MEF": 90000.0,
                "COLA": 150000.0,
                "Face_Amount": 300000,
                "Issue_Age": 47,
                "Gender": "M",
                "Smoker": "Yes",
                "Risk_Class": "Standard Plus",
            },
            {
                "Policy_Number": "P005",
                "Duration": 1,
                "MAC": 0,
                "MOC": 0.60,
                "MEC": 0.11,
                "MAF": 0.0,
                "MEF": 50000.0,
                "COLA": None,
                "Face_Amount": 125000,
                "Issue_Age": 29,
                "Gender": "F",
                "Smoker": "No",
                "Risk_Class": "Preferred",
            },
            {
                "Policy_Number": "P006",
                "Duration": 1,
                "MAC": 1,
                "MOC": 1.0,
                "MEC": 0.25,
                "MAF": 120000.0,
                "MEF": 85000.0,
                "COLA": 120000.0,
                "Face_Amount": 220000,
                "Issue_Age": 57,
                "Gender": "M",
                "Smoker": "No",
                "Risk_Class": "Standard",
            },
            {
                "Policy_Number": "P007",
                "Duration": 1,
                "MAC": 0,
                "MOC": 1.0,
                "MEC": 0.14,
                "MAF": 0.0,
                "MEF": 60000.0,
                "COLA": None,
                "Face_Amount": 210000,
                "Issue_Age": 41,
                "Gender": "F",
                "Smoker": "Yes",
                "Risk_Class": "Standard",
            },
            {
                "Policy_Number": "P008",
                "Duration": 1,
                "MAC": 0,
                "MOC": 0.90,
                "MEC": 0.17,
                "MAF": 0.0,
                "MEF": 72000.0,
                "COLA": None,
                "Face_Amount": 260000,
                "Issue_Age": 63,
                "Gender": "M",
                "Smoker": "No",
                "Risk_Class": "Preferred",
            },
        ]
    )


@pytest.fixture
def sample_csv_path(tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
    path = tmp_path / "synthetic_inforce.csv"
    sample_dataframe.to_csv(path, index=False)
    return path


def final_message(events) -> str:
    finals = [event.message for event in events if event.type == "final"]
    return finals[-1] if finals else ""
