"""Closed evidence schemas for the M1 frozen-intent replay repair."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Sha1 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class _ClosedEvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class M1ReplayCandidateAudit(_ClosedEvidenceModel):
    """Explicit audit of raw replay candidate evolution."""

    candidate_count: int = Field(ge=0)
    candidate_order_equal: bool
    candidate_identity_equal: bool
    candidate_payload_equal: bool
    candidate_delta_fields: list[str]
    protected_candidate_snapshot_path: str = Field(min_length=1)
    protected_candidate_snapshot_sha256: Sha256
    replay_candidate_array_sha256: Sha256


class M1ReplayTransportAudit(_ClosedEvidenceModel):
    """Transport and cache counts for the replay."""

    transport_requests: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    query_count: int = Field(ge=0)


class M1ReplayRepairManifest(_ClosedEvidenceModel):
    """Closed manifest binding all three append-only repair artifacts."""

    repair_version: Literal["m1-frozen-intent-replay-repair.v2"]
    source_bundle: str = Field(min_length=1)
    source_bundle_manifest_sha256: Sha256
    original_first_run_path: str = Field(min_length=1)
    original_first_run_sha256: Sha256
    original_replay_path: str = Field(min_length=1)
    original_replay_sha256: Sha256
    drift_fields: list[str]
    corrected_first_run_path: str = Field(min_length=1)
    corrected_first_run_sha256: Sha256
    corrected_replay_path: str = Field(min_length=1)
    corrected_replay_sha256: Sha256
    canonical_intent_sha256: Sha256
    candidate_audit: M1ReplayCandidateAudit
    query_plan_sha256: Sha256
    zero_transport_replay: M1ReplayTransportAudit
    historical_artifacts_modified: Literal[False]


class M1ReplayImplementationCommits(_ClosedEvidenceModel):
    """Implementation commits named by the repair report."""

    intent_core: Sha1
    repair_runner: Sha1
    repair_validator: Sha1
    repair_schema: Sha1


class M1ReplayHistoricalBundle(_ClosedEvidenceModel):
    """Immutable historical bundle bindings."""

    status: Literal["unchanged"]
    source_bundle: str = Field(min_length=1)
    old_first_run_sha256: Sha256
    old_replay_sha256: Sha256
    old_first_run_frozen_at: datetime
    old_replay_frozen_at: datetime
    drift_fields: list[str]


class M1ReplayM2HistoricalEvidence(_ClosedEvidenceModel):
    """Protected M2 evidence status carried by the repair report."""

    status: Literal["unchanged"]
    protected_hashes_unchanged: Literal[True]
    m2_progress: Literal["IN_PROGRESS 2/5"]


class M1ReplaySourceArtifacts(_ClosedEvidenceModel):
    """Source bundle and historical artifact bindings."""

    source_bundle_manifest_sha256: Sha256
    original_first_run_sha256: Sha256
    original_replay_sha256: Sha256
    drift_fields: list[str]


class M1ReplayRepairArtifact(_ClosedEvidenceModel):
    """Bindings from the report to the generated repair manifest and runs."""

    manifest_sha256: Sha256
    corrected_first_run_sha256: Sha256
    corrected_replay_sha256: Sha256
    canonical_intent_sha256: Sha256
    query_plan_sha256: Sha256
    zero_transport_replay: M1ReplayTransportAudit


class M1ReplayAssertions(_ClosedEvidenceModel):
    """Semantic equality and transport assertions for the replay."""

    canonical_intent_equal: Literal[True]
    candidate_identity_equal: bool
    candidate_payload_equal: bool
    query_ids_equal: Literal[True]
    query_text_equal: Literal[True]
    query_plan_equal: Literal[True]
    transport_requests: int = Field(ge=0)
    cache_hits: int = Field(ge=0)


class M1ReplayEvidenceTypes(_ClosedEvidenceModel):
    """Evidence provenance classification."""

    offline_cache_replay: Literal["real_cache_replay"]
    real_arxiv_requests: Literal["not_run"]
    model_runs: Literal["not_run"]
    human_review: Literal["not_started"]


class M1ReplayCommand(_ClosedEvidenceModel):
    """One reproducible validation command."""

    name: str = Field(min_length=1)
    command: str = Field(min_length=1)
    exit_code: Literal[0]


class M1ReplayShortTestTotal(_ClosedEvidenceModel):
    """Test total without a deselection count."""

    passed: int = Field(ge=0)
    failed: int = Field(ge=0)


class M1ReplayFullTestTotal(M1ReplayShortTestTotal):
    """Test total that records deselected tests."""

    deselected: int = Field(ge=0)


class M1ReplayTestTotals(_ClosedEvidenceModel):
    """Closed test-total map used by the repair report."""

    new_repair_contract: M1ReplayShortTestTotal
    m1_focused_tests: M1ReplayShortTestTotal
    focused_repair_and_m2_tests: M1ReplayShortTestTotal
    full_non_packaging_tests: M1ReplayFullTestTotal
    packaging_tests: M1ReplayFullTestTotal


class M1ReplayNotRun(_ClosedEvidenceModel):
    """Explicitly unexecuted external and later-stage gates."""

    real_arxiv_requests: Literal[True]
    model_runs: Literal[True]
    human_review: Literal[True]
    m2_t04: Literal[True]


class M1ReplayRepairReport(_ClosedEvidenceModel):
    """Closed report schema for the v2 append-only repair evidence."""

    report_version: Literal["m1-frozen-intent-replay-repair.v2"]
    task_id: Literal["M1-FROZEN-INTENT-REPLAY-REPAIR"]
    timestamp_utc: datetime
    implementation_commit: Sha1
    validated_commit: Sha1
    remote_ci_run_id: str = Field(pattern=r"^\d+$")
    remote_ci_head_sha: Sha1
    implementation_commits: M1ReplayImplementationCommits
    root_cause: str = Field(min_length=1)
    repair: str = Field(min_length=1)
    historical_bundle: M1ReplayHistoricalBundle
    m2_historical_evidence: M1ReplayM2HistoricalEvidence
    m2_progress: Literal["IN_PROGRESS 2/5"]
    human_review: Literal["not started"]
    m2_t04: Literal["not started"]
    scoring_eligible: Literal[False]
    source_bundle_inventory_verified: Literal[True]
    source_bundle_verified_file_count: Literal[16]
    source_cache_verified_file_count: Literal[12]
    candidate_projection_applied: Literal[False]
    source_artifacts: M1ReplaySourceArtifacts
    repair_artifact: M1ReplayRepairArtifact
    candidate_audit: M1ReplayCandidateAudit
    replay_assertions: M1ReplayAssertions
    protected_hashes: dict[str, Sha256]
    evidence_types: M1ReplayEvidenceTypes
    commands: list[M1ReplayCommand] = Field(min_length=1)
    test_totals: M1ReplayTestTotals
    not_run: M1ReplayNotRun
