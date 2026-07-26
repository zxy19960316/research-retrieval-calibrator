# Research Retrieval Calibrator M0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build executable domain, state-machine, evaluation, and evidence contracts that gate all later RRC implementation.

**Architecture:** M0 is a network-free Python contract layer. Pydantic models encode record-level invariants, a pure transition function encodes workflow invariants, JSON Schema and small metric functions encode evaluation invariants, and a phase validator refuses to unlock M1 without fresh machine-readable evidence.

**Tech Stack:** Python 3.12, Pydantic 2, pytest 8, PyYAML 6, jsonschema 4, Ruff, mypy

## Global Constraints

- Use Python `>=3.12,<3.13`; do not silently use the currently installed Python 3.13 runtime.
- Do not call arXiv, download embedding/reranker models, start FastAPI, or access CNKI.
- Use test-first red/green cycles and preserve the real command outcome in M0 evidence.
- Keep automated, mock, recorded external, real external, human judged, and not-run evidence separate.
- Do not begin M1 until `scripts/validate_phase.py M0` exits 0.

---

## File map

- `pyproject.toml`: Python and tool configuration.
- `app/models/enums.py`: closed vocabulary shared by every later module.
- `app/models/project.py`: project and Research Intent contracts.
- `app/models/paper.py`: source-backed paper contract.
- `app/models/feedback.py`: feedback and partial-aspect invariants.
- `app/models/query.py`: query and revision provenance.
- `app/core/state_machine.py`: pure legal-transition guard.
- `evaluation/datasets/questions.schema.json`: 10-question template contract.
- `evaluation/datasets/questions.v0.1.yaml`: unjudged M0 question template.
- `evaluation/metrics/retrieval.py`: hand-verifiable metrics.
- `evaluation/report.schema.json`: evidence report contract.
- `scripts/validate_phase.py`: M0 gate.
- `tests/contract/`: invalid/valid contract fixtures.
- `tests/unit/`: state machine and metric tests.

### Task 1: Bootstrap and domain contracts

**Files:**

- Create: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/models/__init__.py`
- Create: `app/models/enums.py`
- Create: `app/models/project.py`
- Create: `app/models/paper.py`
- Create: `app/models/feedback.py`
- Create: `app/models/query.py`
- Test: `tests/contract/test_domain_models.py`

**Interfaces:**

- Produces: `ProjectStage`, `QueryBranch`, `QueryBreadth`, `Relevance`, `FeedbackAspect`, `EvidenceSlot`, `SupportLevel`
- Produces: `RetrievalProject`, `ResearchIntent`, `PaperRecord`, `FeedbackRecord`, `QueryRevision`, `Query`
- Consumes: no runtime service or external data

- [ ] **Step 1: Create the Python project configuration**

```toml
[project]
name = "research-retrieval-calibrator"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "jsonschema>=4.23,<5",
  "pydantic>=2.9,<3",
  "PyYAML>=6.0,<7",
]

[project.optional-dependencies]
dev = [
  "mypy>=1.11,<2",
  "pytest>=8.3,<9",
  "ruff>=0.6,<1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
plugins = ["pydantic.mypy"]
```

- [ ] **Step 2: Write failing contract tests**

```python
from pydantic import ValidationError
import pytest

from app.models.enums import FeedbackAspect, ProjectStage, QueryBranch, Relevance
from app.models.feedback import FeedbackRecord
from app.models.paper import PaperRecord
from app.models.query import Query


def test_partial_feedback_requires_aspect() -> None:
    with pytest.raises(ValidationError):
        FeedbackRecord(
            project_id="RRC-2026-0001",
            paper_id="arxiv:2401.00001",
            relevance=Relevance.PARTIAL,
            aspects=[],
            raw_text="方法部分相关",
        )


def test_visible_paper_requires_source_identity_and_url() -> None:
    with pytest.raises(ValidationError):
        PaperRecord(
            paper_id="arxiv:2401.00001",
            source="arxiv",
            source_id="",
            title="A real title",
            url="",
            language="en",
            user_visible=True,
        )


def test_round_two_query_requires_revision_source() -> None:
    with pytest.raises(ValidationError):
        Query(
            query_id="Q2-M-01",
            round_number=2,
            branch=QueryBranch.METHOD_DOMAIN,
            breadth="MEDIUM",
            language="en",
            query_text="graph surrogate",
            weight=0.25,
            revision_sources=[],
        )


def test_closed_enums_reject_unknown_values() -> None:
    with pytest.raises(ValueError):
        ProjectStage("SEARCHING")
    with pytest.raises(ValueError):
        FeedbackAspect("WHOLE_PAPER")
```

- [ ] **Step 3: Run the tests and confirm red**

Run:

```powershell
py -3.12 -m pytest tests/contract/test_domain_models.py -q
```

Expected: collection fails because `app.models` does not exist.

- [ ] **Step 4: Implement closed enums**

```python
from enum import StrEnum


class ProjectStage(StrEnum):
    INIT = "INIT"
    CLARIFYING = "CLARIFYING"
    ROUND1_SEARCHING = "ROUND1_SEARCHING"
    WAITING_FOR_FEEDBACK = "WAITING_FOR_FEEDBACK"
    ROUND2_SEARCHING = "ROUND2_SEARCHING"
    WAITING_FOR_DIRECTION_CONFIRMATION = "WAITING_FOR_DIRECTION_CONFIRMATION"
    FINALIZED = "FINALIZED"


class QueryBranch(StrEnum):
    DIRECT_INTERSECTION = "DIRECT_INTERSECTION"
    PROBLEM_DOMAIN = "PROBLEM_DOMAIN"
    METHOD_DOMAIN = "METHOD_DOMAIN"
    BRIDGE_DOMAIN = "BRIDGE_DOMAIN"


class QueryBreadth(StrEnum):
    NARROW = "NARROW"
    MEDIUM = "MEDIUM"
    WIDE = "WIDE"


class Relevance(StrEnum):
    HIGH = "HIGH"
    PARTIAL = "PARTIAL"
    IRRELEVANT = "IRRELEVANT"


class FeedbackAspect(StrEnum):
    OBJECT = "OBJECT"
    TASK = "TASK"
    METHOD = "METHOD"
    PHYSICS = "PHYSICS"
    BRIDGE = "BRIDGE"
    EVALUATION = "EVALUATION"


class EvidenceSlot(StrEnum):
    PROBLEM_EXISTENCE = "PROBLEM_EXISTENCE"
    CURRENT_METHODS = "CURRENT_METHODS"
    METHOD_TRANSFERABILITY = "METHOD_TRANSFERABILITY"
    IMPLEMENTATION_PATH = "IMPLEMENTATION_PATH"
    EVALUATION_BASIS = "EVALUATION_BASIS"


class SupportLevel(StrEnum):
    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"
    HYPOTHETICAL = "HYPOTHETICAL"
```

- [ ] **Step 5: Implement minimal Pydantic contracts**

```python
# app/models/feedback.py
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.models.enums import FeedbackAspect, Relevance


class FeedbackRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str = Field(pattern=r"^RRC-\d{4}-\d{4}$")
    paper_id: str = Field(min_length=1)
    relevance: Relevance
    aspects: list[FeedbackAspect] = Field(default_factory=list)
    raw_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aspects(self) -> "FeedbackRecord":
        if self.relevance is Relevance.PARTIAL and not self.aspects:
            raise ValueError("PARTIAL feedback requires at least one aspect")
        if self.relevance is not Relevance.PARTIAL and self.aspects:
            raise ValueError("Only PARTIAL feedback accepts aspects")
        return self
```

```python
# app/models/paper.py
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PaperRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_id: str
    title: str = Field(min_length=1)
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1900, le=2100)
    doi: str | None = None
    url: str
    language: str = Field(pattern=r"^(zh|en)$")
    user_visible: bool = False

    @model_validator(mode="after")
    def validate_source_identity(self) -> "PaperRecord":
        if self.user_visible and (not self.source_id.strip() or not self.url.startswith("http")):
            raise ValueError("User-visible papers require source_id and HTTP(S) URL")
        return self
```

```python
# app/models/query.py
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.models.enums import QueryBranch, QueryBreadth


class QueryRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_paper_id: str | None = None
    rule: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_id: str = Field(min_length=1)
    round_number: int = Field(ge=1, le=2)
    branch: QueryBranch
    breadth: QueryBreadth
    language: str = Field(pattern=r"^(zh|en)$")
    query_text: str = Field(min_length=1)
    weight: float = Field(ge=0, le=1)
    revision_sources: list[QueryRevision] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_round_provenance(self) -> "Query":
        if self.round_number == 1 and self.revision_sources:
            raise ValueError("Round-one queries cannot consume feedback revisions")
        if self.round_number == 2 and not self.revision_sources:
            raise ValueError("Round-two queries require revision provenance")
        return self
```

Implement `ResearchIntent` and `RetrievalProject` exactly from `docs/data_model.md`, using `ConfigDict(extra="forbid")`, non-empty term collections, UTC-aware datetimes, and `RRC-YYYY-NNNN` IDs.

- [ ] **Step 6: Run contract, lint, and type checks**

Run:

```powershell
py -3.12 -m pytest tests/contract/test_domain_models.py -q
py -3.12 -m ruff check app tests
py -3.12 -m mypy app
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml app tests/contract/test_domain_models.py
git commit -m "feat: define M0 domain contracts"
```

### Task 2: State machine

**Files:**

- Create: `app/core/__init__.py`
- Create: `app/core/state_machine.py`
- Test: `tests/unit/test_state_machine.py`

**Interfaces:**

- Consumes: `ProjectStage`
- Produces: `transition(current: ProjectStage, target: ProjectStage, valid_feedback_count: int) -> ProjectStage`

- [ ] **Step 1: Write failing state tests**

```python
import pytest
from app.core.state_machine import InvalidTransition, transition
from app.models.enums import ProjectStage


def test_legal_transition() -> None:
    assert transition(ProjectStage.INIT, ProjectStage.CLARIFYING, 0) is ProjectStage.CLARIFYING


def test_cannot_skip_clarification() -> None:
    with pytest.raises(InvalidTransition):
        transition(ProjectStage.INIT, ProjectStage.ROUND1_SEARCHING, 0)


def test_round_two_requires_six_valid_feedback_items() -> None:
    with pytest.raises(InvalidTransition):
        transition(
            ProjectStage.WAITING_FOR_FEEDBACK,
            ProjectStage.ROUND2_SEARCHING,
            5,
        )


def test_finalized_is_terminal() -> None:
    with pytest.raises(InvalidTransition):
        transition(ProjectStage.FINALIZED, ProjectStage.CLARIFYING, 6)
```

- [ ] **Step 2: Run and confirm red**

Run: `py -3.12 -m pytest tests/unit/test_state_machine.py -q`

Expected: FAIL because `app.core.state_machine` is missing.

- [ ] **Step 3: Implement the pure transition guard**

```python
from app.models.enums import ProjectStage


class InvalidTransition(ValueError):
    pass


_NEXT: dict[ProjectStage, ProjectStage] = {
    ProjectStage.INIT: ProjectStage.CLARIFYING,
    ProjectStage.CLARIFYING: ProjectStage.ROUND1_SEARCHING,
    ProjectStage.ROUND1_SEARCHING: ProjectStage.WAITING_FOR_FEEDBACK,
    ProjectStage.WAITING_FOR_FEEDBACK: ProjectStage.ROUND2_SEARCHING,
    ProjectStage.ROUND2_SEARCHING: ProjectStage.WAITING_FOR_DIRECTION_CONFIRMATION,
    ProjectStage.WAITING_FOR_DIRECTION_CONFIRMATION: ProjectStage.FINALIZED,
}


def transition(
    current: ProjectStage,
    target: ProjectStage,
    valid_feedback_count: int,
) -> ProjectStage:
    if _NEXT.get(current) is not target:
        raise InvalidTransition(f"Cannot transition from {current} to {target}")
    if target is ProjectStage.ROUND2_SEARCHING and valid_feedback_count < 6:
        raise InvalidTransition("Round two requires at least six valid feedback items")
    return target
```

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
py -3.12 -m pytest tests/unit/test_state_machine.py -q
py -3.12 -m pytest -q
git add app/core tests/unit/test_state_machine.py
git commit -m "feat: enforce project state transitions"
```

Expected: tests exit 0 and commit succeeds.

### Task 3: Evaluation template and metrics

**Files:**

- Create: `evaluation/__init__.py`
- Create: `evaluation/metrics/__init__.py`
- Create: `evaluation/metrics/retrieval.py`
- Create: `evaluation/datasets/questions.schema.json`
- Create: `evaluation/datasets/questions.v0.1.yaml`
- Test: `tests/contract/test_evaluation_dataset.py`
- Test: `tests/unit/test_retrieval_metrics.py`

**Interfaces:**

- Produces: `precision_at_k`（Strict/Inclusive，固定分母 `k`）、`ndcg_at_k`（支持显式 ideal sequence）、`evidence_coverage`、`negative_suppression`、`new_useful_papers`、`metadata_hallucination_rate`。
- Produces: `JudgedPaper`，仅组合非空论文 ID 与现有 `Relevance`，不依赖 M2 `RankedPaper`。
- Produces: 一个 schema 校验通过的 5+5 `unjudged` 十题模板。
- Metric semantics: the six frozen formulas and all `k > 0` failure behavior are authoritative in `docs/evaluation.md`; Task 3 must not substitute a returned-list-length denominator, implicit-only IDCG, absolute negative suppression, or inferred source confirmation.

- [ ] **Step 1: Write hand-calculated failing metric tests**

```python
from evaluation.metrics.retrieval import (
    evidence_coverage,
    metadata_hallucination_rate,
    ndcg_at_k,
    precision_at_k,
)


def test_precision_variants() -> None:
    labels = ["HIGH", "PARTIAL", "IRRELEVANT", "HIGH"]
    assert precision_at_k(labels, 4, inclusive=False) == 0.5
    assert precision_at_k(labels, 4, inclusive=True) == 0.75


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["HIGH", "PARTIAL", "IRRELEVANT"], 3) == 1.0


def test_evidence_coverage_uses_five_fixed_slots() -> None:
    assert evidence_coverage({"PROBLEM_EXISTENCE", "CURRENT_METHODS"}) == 0.4


def test_metadata_hallucination_rate() -> None:
    assert metadata_hallucination_rate([True, True, False, True]) == 0.25
```

- [ ] **Step 2: Run and confirm red**

Run: `py -3.12 -m pytest tests/unit/test_retrieval_metrics.py -q`

Expected: collection fails because metric functions are absent.

- [ ] **Step 3: Implement the metric functions**

```python
from math import log2

_GAIN = {"HIGH": 2, "PARTIAL": 1, "IRRELEVANT": 0}
_SLOTS = {
    "PROBLEM_EXISTENCE",
    "CURRENT_METHODS",
    "METHOD_TRANSFERABILITY",
    "IMPLEMENTATION_PATH",
    "EVALUATION_BASIS",
}


def precision_at_k(labels: list[str], k: int, *, inclusive: bool) -> float:
    window = labels[:k]
    if not window:
        return 0.0
    relevant = {"HIGH", "PARTIAL"} if inclusive else {"HIGH"}
    return sum(label in relevant for label in window) / len(window)


def ndcg_at_k(labels: list[str], k: int) -> float:
    gains = [_GAIN[label] for label in labels[:k]]
    dcg = sum(gain / log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(gain / log2(index + 2) for index, gain in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def evidence_coverage(slots: set[str]) -> float:
    unknown = slots - _SLOTS
    if unknown:
        raise ValueError(f"Unknown evidence slots: {sorted(unknown)}")
    return len(slots) / len(_SLOTS)


def metadata_hallucination_rate(source_verified: list[bool]) -> float:
    if not source_verified:
        return 0.0
    return sum(not value for value in source_verified) / len(source_verified)
```

- [ ] **Step 4: Create and validate the 10-question template**

The YAML must contain IDs `RRC-Q01` through `RRC-Q10`; `RRC-Q01`-`RRC-Q05` use `nuclear_engineering`, and `RRC-Q06`-`RRC-Q10` use `cross_domain_stem`. Every entry has non-empty `question_zh`, `question_en`, `scope`, `exclusions`, `judging_guide`, and `label_status: unjudged`.

Write `tests/contract/test_evaluation_dataset.py` to load the YAML, validate it against the JSON Schema, assert exactly 10 unique IDs, assert the 5+5 split, and reject any status other than `unjudged` or `frozen`.

- [ ] **Step 5: Run and commit**

Run:

```powershell
py -3.12 -m pytest tests/contract/test_evaluation_dataset.py tests/unit/test_retrieval_metrics.py -q
py -3.12 -m pytest -q
git add evaluation tests/contract/test_evaluation_dataset.py tests/unit/test_retrieval_metrics.py
git commit -m "eval: define M0 dataset and metrics"
```

Expected: all tests pass and commit succeeds.

### Task 4: Evidence report and M0 gate

**Files:**

- Create: `evaluation/report.schema.json`
- Create: `scripts/validate_phase.py`
- Create: `tests/contract/test_evidence_report.py`
- Create: `evaluation/reports/m0-validation.json`
- Modify: `STATUS.md`

**Interfaces:**

- Consumes: test results, input SHA-256, Git commit, runtime versions
- Produces: exit code 0 only for a fresh, complete M0 report

- [ ] **Step 1: Write failing evidence tests**

Test that the schema rejects:

- missing commit
- unknown evidence type
- a successful `real_external` check with no command and no observed timestamp
- an M0 report that omits explicit `not_run` checks for arXiv, models, platform, and CNKI
- an input path whose current SHA-256 differs from the report

- [ ] **Step 2: Run and confirm red**

Run: `py -3.12 -m pytest tests/contract/test_evidence_report.py -q`

Expected: FAIL because the schema and validator are missing.

- [ ] **Step 3: Implement the report schema and validator**

The report schema requires:

```json
{
  "phase": "M0",
  "commit": "40 lowercase hex characters",
  "generated_at": "UTC date-time",
  "python_version": "3.12.x",
  "checks": [
    {
      "name": "domain-contract-tests",
      "evidence_type": "automated",
      "status": "passed",
      "command": "py -3.12 -m pytest tests/contract/test_domain_models.py -q",
      "exit_code": 0
    }
  ],
  "validated_inputs": [
    {
      "path": "PRODUCT_SPEC.md",
      "sha256": "64 lowercase hex characters"
    }
  ]
}
```

`scripts/validate_phase.py M0` must validate JSON Schema, recompute every input hash, require all M0 tests to be passed, and require arXiv/models/platform/CNKI checks to exist with `evidence_type: not_run`.

- [ ] **Step 4: Run full verification**

Run:

```powershell
py -3.12 -m ruff check app evaluation scripts tests
py -3.12 -m mypy app evaluation scripts
py -3.12 -m pytest -q
py -3.12 scripts/validate_phase.py M0
```

Expected: every command exits 0. The report must contain the actual commit that owns all validated files; if generating the report changes the commit, use a two-commit evidence pattern and validate the report's declared code commit rather than self-referencing the report commit.

- [ ] **Step 5: Update status and commit**

Set M0 to `COMPLETE`, M1 to `READY`, current stage to `M1`, and next action to `M1-T01`. Do not change later stages.

```powershell
git add evaluation/report.schema.json scripts/validate_phase.py tests/contract/test_evidence_report.py evaluation/reports/m0-validation.json STATUS.md
git commit -m "chore: close M0 contract gate"
git status --short --branch
```

Expected: commit succeeds and the worktree is clean.

## Self-review checklist

- Every requirement in `docs/phases/M0-product-and-evaluation-contracts.md` maps to one task above.
- No M1 runtime, external source, model download, API, or CNKI behavior is implemented.
- Enum and field names match `docs/data_model.md`.
- Evidence categories remain separate.
- The status transition occurs only after the validator succeeds.
