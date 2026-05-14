"""Allowlist sanitization helpers for AI interpretation packets."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from skills.experience_study_skill.ai_models import AIValidationIssue


MASKED_COHORT_LABEL = "[masked cohort label]"
MASKED_DIMENSION_COLUMN = "[masked dimension]"
MASKED_FILTER_COLUMN = "[masked column]"
MASKED_FILTER_VALUE = "[masked value]"

SENSITIVE_DIMENSION_TERMS = {
    "policy",
    "policy_number",
    "name",
    "dob",
    "birth",
    "ssn",
    "email",
    "phone",
    "address",
    "zip",
    "postal",
    "member",
    "applicant",
    "insured",
    "id",
    "number",
    "account",
    "certificate",
}

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")
_SSN_RE = re.compile(r"^\d{3}-?\d{2}-?\d{4}$")


@dataclass(frozen=True, slots=True)
class ParsedDimensionPart:
    """One parsed dimension component from a sweep Dimensions label."""

    column: str
    value: str


def _dimension_tokens(column: str) -> set[str]:
    expanded = _CAMEL_BOUNDARY_RE.sub("_", column)
    tokens = [
        token.lower()
        for token in re.split(r"[^A-Za-z0-9]+", expanded)
        if token
    ]
    token_set = set(tokens)
    if tokens:
        token_set.add("_".join(tokens))
    return token_set


def dimension_column_is_sensitive(column: str) -> bool:
    """Return True when a dimension column token matches the sensitive term set."""

    return bool(_dimension_tokens(column) & SENSITIVE_DIMENSION_TERMS)


def parse_dimension_label(
    dimensions: str,
    *,
    evidence_ref: str,
) -> tuple[list[ParsedDimensionPart], list[AIValidationIssue]]:
    """Parse a sweep Dimensions label conservatively without failing the packet."""

    parsed: list[ParsedDimensionPart] = []
    warnings: list[AIValidationIssue] = []
    for part in str(dimensions).split(" | "):
        if "=" not in part:
            warnings.append(
                AIValidationIssue(
                    code="dimension_parse_warning",
                    message="Unable to fully parse dimension label.",
                    evidence_refs=[evidence_ref],
                )
            )
            continue
        column, value = part.split("=", 1)
        column = column.strip()
        value = value.strip()
        if not column:
            warnings.append(
                AIValidationIssue(
                    code="dimension_parse_warning",
                    message="Unable to fully parse dimension label.",
                    evidence_refs=[evidence_ref],
                )
            )
            continue
        parsed.append(ParsedDimensionPart(column=column, value=value))
    return parsed, warnings


def _value_looks_sensitive(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return bool(
        _EMAIL_RE.match(stripped)
        or _PHONE_RE.match(stripped)
        or _SSN_RE.match(stripped)
    )


def sanitize_selected_columns(
    selected_columns: list[str] | None,
) -> tuple[list[str] | None, list[AIValidationIssue]]:
    """Mask selected column metadata when it names sensitive dimensions."""

    if selected_columns is None:
        return None, []

    sanitized: list[str] = []
    warnings: list[AIValidationIssue] = []
    for column in selected_columns:
        if dimension_column_is_sensitive(str(column)):
            sanitized.append(MASKED_DIMENSION_COLUMN)
            warnings.append(
                AIValidationIssue(
                    code="sensitive_selected_column_masked",
                    message="A sensitive or disallowed selected column was masked.",
                )
            )
        else:
            sanitized.append(str(column))
    return sanitized, warnings


def sanitize_filters(
    filters: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[AIValidationIssue]]:
    """Allow filter metadata while masking sensitive columns or obvious identifiers."""

    sanitized_filters: list[dict[str, Any]] = []
    warnings: list[AIValidationIssue] = []
    for filter_spec in filters or []:
        column = str(filter_spec.get("column", ""))
        value = filter_spec.get("value")
        operator = filter_spec.get("operator")
        if dimension_column_is_sensitive(column) or _value_looks_sensitive(value):
            sanitized_filters.append(
                {
                    "column": MASKED_FILTER_COLUMN,
                    "operator": operator,
                    "value": MASKED_FILTER_VALUE,
                }
            )
            warnings.append(
                AIValidationIssue(
                    code="sensitive_filter_masked",
                    message="A sensitive or disallowed filter was masked.",
                )
            )
        else:
            sanitized_filters.append(
                {
                    "column": column,
                    "operator": operator,
                    "value": value,
                }
            )
    return sanitized_filters, warnings
