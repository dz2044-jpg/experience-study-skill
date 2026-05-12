import importlib
import inspect
from pathlib import Path

import pandas as pd

import skills.experience_study_skill.native_tools as native_tools
from skills.experience_study_skill.native_tools import (
    ToolExecutionContext,
    compute_ae_ci,
    compute_ae_ci_amount,
    generate_combined_report,
    run_actuarial_data_checks,
    run_dimensional_sweep,
)
from skills.experience_study_skill.visualization import _build_table_figure


EXPECTED_PUBLIC_FUNCTIONS = {
    "profile_dataset",
    "inspect_dataset_schema",
    "run_actuarial_data_checks",
    "create_categorical_bands",
    "regroup_categorical_features",
    "run_dimensional_sweep",
    "generate_combined_report",
    "get_tool_handlers",
}

EXPECTED_PUBLIC_TOOL_KEYS = {
    "profile_dataset",
    "inspect_dataset_schema",
    "run_actuarial_data_checks",
    "create_categorical_bands",
    "regroup_categorical_features",
    "run_dimensional_sweep",
    "generate_combined_report",
}

FOCUSED_MODULE_NAMES = [
    "skills.experience_study_skill.io",
    "skills.experience_study_skill.validation",
    "skills.experience_study_skill.feature_engineering",
    "skills.experience_study_skill.ae_math",
    "skills.experience_study_skill.sweeps",
    "skills.experience_study_skill.visualization",
]


def test_native_tools_public_api_inventory_remains_available():
    for function_name in EXPECTED_PUBLIC_FUNCTIONS:
        assert hasattr(native_tools, function_name)
        assert callable(getattr(native_tools, function_name))

    handlers = native_tools.get_tool_handlers()

    assert set(handlers) == EXPECTED_PUBLIC_TOOL_KEYS
    for tool_name, handler in handlers.items():
        assert callable(handler)
        assert handler.__globals__[tool_name] is getattr(native_tools, tool_name)


def test_focused_deterministic_modules_import_without_native_tools_dependency():
    for module_name in FOCUSED_MODULE_NAMES:
        module = importlib.import_module(module_name)

        assert module is not None
        assert "native_tools" not in inspect.getsource(module)


def test_zero_claim_ci_amount_still_returns_upper_bound():
    lower, upper = compute_ae_ci_amount(
        mac=0,
        moc=1000,
        mec=4,
        actual_amount=0,
        expected_amount=400000,
    )

    assert lower is not None
    assert upper is not None
    assert upper > 0


def test_count_ci_returns_values_for_standard_case():
    lower, upper = compute_ae_ci(mac=5, moc=1000, mec=4)

    assert lower is not None
    assert upper is not None
    assert lower < upper


def test_visualization_tool_enforces_prerequisite(tmp_path: Path):
    context = ToolExecutionContext(
        session_id="session-a",
        output_dir=tmp_path / "sessions" / "session-a",
    )

    result = generate_combined_report(context=context)

    assert result["ok"] is False
    assert result["kind"] == "missing_prerequisite"
    assert "Run dimensional sweep first" in result["message"]


def test_dimensional_sweep_caps_top_n_payload(tmp_path: Path):
    rows = [
        {
            "Policy_Number": f"P{index:03d}",
            "MAC": 1,
            "MOC": 1.0,
            "MEC": 0.25,
            "MAF": 100000.0 + index,
            "MEF": 80000.0,
            "Segment": f"segment_{index:02d}",
        }
        for index in range(25)
    ]
    data_path = tmp_path / "prepared.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    context = ToolExecutionContext(
        session_id="session-a",
        output_dir=tmp_path / "sessions" / "session-a",
    )

    result = run_dimensional_sweep(
        context=context,
        data_path=str(data_path),
        selected_columns=["Segment"],
        top_n=500,
    )

    assert result["ok"] is True
    assert len(result["data"]["results"]) == 20


def test_actuarial_validation_flags_missing_core_actuarial_columns(tmp_path: Path):
    data_path = tmp_path / "missing_core.csv"
    pd.DataFrame(
        [
            {
                "Policy_Number": "P001",
                "Duration": 1,
                "Face_Amount": 100000,
                "Issue_Age": 45,
            }
        ]
    ).to_csv(data_path, index=False)
    context = ToolExecutionContext(
        session_id="session-a",
        output_dir=tmp_path / "sessions" / "session-a",
    )

    result = run_actuarial_data_checks(context=context, data_path=str(data_path))

    assert result["ok"] is True
    assert result["data"]["status"] == "FAIL"
    assert result["data"]["issues"][:5] == [
        "Missing required actuarial column: MAC.",
        "Missing required actuarial column: MEC.",
        "Missing required actuarial column: MAF.",
        "Missing required actuarial column: MEF.",
        "Missing required actuarial column: MOC.",
    ]


def test_dimensional_sweep_blanks_invalid_ratios_and_sorts_them_last(tmp_path: Path):
    data_path = tmp_path / "prepared.csv"
    pd.DataFrame(
        [
            {
                "Policy_Number": "P001",
                "MAC": 1,
                "MOC": 1.0,
                "MEC": 0.5,
                "MAF": 200000.0,
                "MEF": 100000.0,
                "Segment": "valid",
            },
            {
                "Policy_Number": "P002",
                "MAC": 1,
                "MOC": 1.0,
                "MEC": 0.0,
                "MAF": 100000.0,
                "MEF": 100000.0,
                "Segment": "zero_mec",
            },
            {
                "Policy_Number": "P003",
                "MAC": 1,
                "MOC": 1.0,
                "MEC": 1.0,
                "MAF": 100000.0,
                "MEF": 0.0,
                "Segment": "zero_mef",
            },
        ]
    ).to_csv(data_path, index=False)
    context = ToolExecutionContext(
        session_id="session-a",
        output_dir=tmp_path / "sessions" / "session-a",
    )

    result = run_dimensional_sweep(
        context=context,
        data_path=str(data_path),
        selected_columns=["Segment"],
        sort_by="AE_Ratio_Count",
        top_n=10,
    )

    assert result["ok"] is True
    assert [row["Dimensions"] for row in result["data"]["results"]][-1] == "Segment=zero_mec"
    zero_mec_result = next(
        row for row in result["data"]["results"] if row["Dimensions"] == "Segment=zero_mec"
    )
    zero_mef_result = next(
        row for row in result["data"]["results"] if row["Dimensions"] == "Segment=zero_mef"
    )
    assert zero_mec_result["AE_Ratio_Count"] is None
    assert zero_mec_result["AE_Count_CI_Lower"] is None
    assert zero_mec_result["AE_Count_CI_Upper"] is None
    assert zero_mef_result["AE_Ratio_Amount"] is None
    assert zero_mef_result["AE_Amount_CI_Lower"] is None
    assert zero_mef_result["AE_Amount_CI_Upper"] is None

    latest_path = context.canonical_sweep_path()
    csv_df = pd.read_csv(latest_path, keep_default_na=False)
    assert list(csv_df.columns) == [
        "Dimensions",
        "Sum_MAC",
        "Sum_MOC",
        "Sum_MEC",
        "Sum_MAF",
        "Sum_MEF",
        "AE_Ratio_Count",
        "AE_Ratio_Amount",
        "AE_Count_CI_Lower",
        "AE_Count_CI_Upper",
        "AE_Amount_CI_Lower",
        "AE_Amount_CI_Upper",
    ]
    csv_by_dimension = csv_df.set_index("Dimensions")
    assert csv_by_dimension.loc["Segment=zero_mec", "AE_Ratio_Count"] == ""
    assert csv_by_dimension.loc["Segment=zero_mec", "AE_Count_CI_Lower"] == ""
    assert csv_by_dimension.loc["Segment=zero_mec", "AE_Count_CI_Upper"] == ""
    assert csv_by_dimension.loc["Segment=zero_mef", "AE_Ratio_Amount"] == ""
    assert csv_by_dimension.loc["Segment=zero_mef", "AE_Amount_CI_Lower"] == ""
    assert csv_by_dimension.loc["Segment=zero_mef", "AE_Amount_CI_Upper"] == ""

    amount_sorted = run_dimensional_sweep(
        context=context,
        data_path=str(data_path),
        selected_columns=["Segment"],
        sort_by="AE_Ratio_Amount",
        top_n=10,
    )
    assert amount_sorted["ok"] is True
    assert [row["Dimensions"] for row in amount_sorted["data"]["results"]][-1] == "Segment=zero_mef"


def test_visualization_table_formats_invalid_ratios_as_na():
    fig = _build_table_figure(
        pd.DataFrame(
            [
                {
                    "Dimensions": "Segment=Invalid",
                    "Sum_MOC": 1.0,
                    "Sum_MAC": 1,
                    "Sum_MEC": 0.0,
                    "AE_Ratio_Count": None,
                    "AE_Count_CI_Lower": None,
                    "AE_Count_CI_Upper": None,
                }
            ]
        ),
        "count",
        "sweep_summary.csv",
    )

    cell_values = fig.data[0].cells.values
    assert cell_values[4][0] == "n/a"
    assert cell_values[5][0] == "n/a"
