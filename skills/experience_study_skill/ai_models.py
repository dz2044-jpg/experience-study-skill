"""Typed contracts for internal AI interpretation packets and responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AI_SWEEP_PACKET_SCHEMA_VERSION = "ai_sweep_packet.v1"
AI_DEFAULT_MASKING_MIN_MAC = 1

AIActionName = Literal[
    "summarize_sweep",
    "explain_cohort",
    "compare_cohorts",
    "analyze_count_amount_divergence",
]
AISourceMode = Literal["llm", "fallback"]
ThresholdOperator = Literal[">=", ">"]


class AIValidationIssue(BaseModel):
    """Validation issue or caution attached to AI text or packet construction."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    evidence_refs: list[str] = Field(default_factory=list)


class AIValidationResult(BaseModel):
    """Result from validating generated AI interpretation text."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool = True
    blocked_issues: list[AIValidationIssue] = Field(default_factory=list)
    warnings: list[AIValidationIssue] = Field(default_factory=list)


class AICohortRow(BaseModel):
    """Sanitized cohort-level row made available for AI interpretation."""

    model_config = ConfigDict(extra="forbid")

    evidence_ref: str
    Dimensions: str
    Dimension_Columns: list[str] = Field(default_factory=list)
    Sum_MAC: float
    Sum_MOC: float
    Sum_MEC: float
    Sum_MAF: float
    Sum_MEF: float
    AE_Ratio_Count: float
    AE_Ratio_Amount: float
    AE_Count_CI_Lower: float | None = None
    AE_Count_CI_Upper: float | None = None
    AE_Amount_CI_Lower: float | None = None
    AE_Amount_CI_Upper: float | None = None
    low_credibility: bool = False
    masking_reason: str | None = None
    caution_flags: list[str] = Field(default_factory=list)


class AISweepPacket(BaseModel):
    """Sanitized packet derived from a deterministic sweep artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = AI_SWEEP_PACKET_SCHEMA_VERSION
    source_artifact_path: str
    source_content_hash: str
    state_fingerprint: str | None = None
    packet_fingerprint: str | None = None
    depth: int | None = None
    filters: list[dict[str, Any]] = Field(default_factory=list)
    selected_columns: list[str] | None = None
    sort_by: str | None = None
    deterministic_min_mac: int | None = None
    ai_masking_min_mac: int = AI_DEFAULT_MASKING_MIN_MAC
    rows: list[AICohortRow] = Field(default_factory=list)
    warnings: list[AIValidationIssue] = Field(default_factory=list)


class AIActionResponse(BaseModel):
    """Internal response from an AI interpretation action."""

    model_config = ConfigDict(extra="forbid")

    action_name: AIActionName
    source_mode: AISourceMode
    response_text: str
    evidence_refs: list[str] = Field(default_factory=list)
    caution_flags: list[str] = Field(default_factory=list)
    next_review_steps: list[str] = Field(default_factory=list)
    state_fingerprint: str | None = None
    packet_fingerprint: str | None = None
    validation: AIValidationResult = Field(default_factory=AIValidationResult)
