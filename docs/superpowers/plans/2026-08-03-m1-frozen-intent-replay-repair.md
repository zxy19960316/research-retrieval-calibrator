# M1 Frozen Intent Replay Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans when available. Execute this plan inline because the user explicitly authorized the implementation.

**Goal:** Repair M1 first-round replay so a replay reuses and preserves the original frozen `ResearchIntent`, while producing append-only, zero-network repair evidence without changing historical M1 or protected M2 artifacts.

**Architecture:** Add one canonical `ResearchIntent` byte representation in the core intent module. Extend the first-round core API with an optional validated frozen intent; replay derives the deterministic intent from the same question and compares canonical bytes before using the frozen value. Build the repaired replay with a cache-only adapter and a transport that fails closed. Store the repaired bundle and manifest under a new dated evidence directory, and validate it with a dedicated validator wired into CI.

**Tech Stack:** Python 3.12, Pydantic models, pytest, Ruff, mypy, JSON evidence artifacts, GitHub Actions, PowerShell, `gh`.

## Global Constraints

- Work only on `agent/m1-frozen-intent-replay-repair`, based on the latest `origin/main`.
- Preserve all existing M1 rebaseline files and reports byte-for-byte.
- Preserve the M2 candidate snapshot, manifest, M2-T01/T02/T03 formal artifacts, and `STATUS.md` M2 state (`IN_PROGRESS 2/5`).
- Do not run models, make real arXiv requests, create a human review bundle, start M2-T04, or claim real external/model validation.
- Do not omit `frozen_at` from canonical identity; no semantic identity bypass, hand-edited historical JSON, rebase, squash, amend, force-push, or destructive cleanup.
- Use explicit file paths for staging and retain independent commits.

## Step 1: Capture baseline and expose the contract failure

1. Verify branch, `HEAD`, remote tracking state, clean worktree, protected hashes, and the unchanged historical first-run/replay hashes.
2. Add focused tests in `tests/contract/test_m1_frozen_intent_replay_repair.py` covering normal timestamp behavior, replay preservation, canonical identity, stable role ordering, every intent-field mutation, non-UTC rejection, and cache-miss fail-closed behavior.
3. Run the focused tests before implementation and record the expected red failures.
4. Commit only the tests as `test: expose frozen intent replay drift`.

## Step 2: Implement canonical identity and frozen-intent replay

1. Add `canonical_research_intent_bytes()` to `app/core/intent.py`; revalidate the model, sort roles, retain every identity field including `frozen_at`, normalize UTC to microsecond `Z`, and serialize sorted compact UTF-8 JSON.
2. Extend `run_first_round()` in `app/core/first_round.py` with `frozen_intent: ResearchIntent | None = None`.
3. On replay, revalidate the supplied intent, derive the question intent using its `frozen_at`, compare canonical bytes, return `INTENT_REPLAY_MISMATCH` on any mismatch, and use the validated frozen object.
4. Use the intent timestamp for query-plan generation so a repaired replay has the same plan identity as the original run; keep normal-run behavior unchanged.
5. Run focused intent/replay tests, existing first-round unit/integration tests, Ruff import ordering, and mypy.
6. Commit the core changes as `fix: preserve frozen intent during first-round replay`.

## Step 3: Build the offline append-only repair runner

1. Add `scripts/repair_m1_frozen_intent_replay.py` with a refusal-by-default CLI requiring `--execute-offline-repair`.
2. Verify fixed source hashes and exact historical drift before writing anything.
3. Load and validate the historical first-run/replay `FirstRoundRun` objects; byte-copy only the historical first-run output into the new dated bundle.
4. Replay from the historical question and frozen first-run intent with the existing cache and a forbidden transport; require zero transport requests, 12 cache hits, equal candidates, equal query IDs/text, and equal canonical intent/query-plan identity.
5. Write only the new replay output and `repair-manifest.json`; make reruns idempotent and refuse conflicting target bytes.
6. Run the runner refusal test, execute the authorized offline repair, inspect generated hashes and bundle inventory, and commit the runner as `feat: add offline M1 intent replay repair runner`.

## Step 4: Add append-only validation and repair report

1. Add `scripts/validate_m1_frozen_intent_replay_repair.py` to validate source hashes, Pydantic contracts, exact historical drift, first-run byte identity, corrected replay semantic equality, canonical hashes, zero transport/12 cache hits, clean bundle contents, no embedded cache or absolute-path/network evidence, old-file immutability, protected M2 hashes, M2 `2/5`, and M2-T04 not started.
2. Run the new validator against the generated bundle and fix only implementation or generated new evidence when it reports a genuine defect.
3. Add `evaluation/reports/m1-frozen-intent-replay-repair-2026-08-03.json` with the root cause, repair, old/new hashes, manifest/canonical/candidate/query-plan bindings, exact validation commands/results, unchanged historical/protected evidence, M2 state, human-review state, and scoring disabled.
4. Commit the generated append-only evidence and report as `chore: record append-only intent replay repair evidence`.

## Step 5: Integrate CI and run release gates

1. Add the dedicated validator step to `.github/workflows/docs-validation.yml` immediately after the existing M1 validation and before M2 validators.
2. Run the requested focused commands, all source tests, packaging checks, Ruff (including `I001`), mypy, existing M0/M1/M2-T01/T02/T03 validators, and the new validator.
3. Recheck old M1 files, reports, M2 artifacts, and protected hashes byte-for-byte; verify no model/network activity occurred.
4. Commit CI as `ci: validate M1 frozen intent replay repair`.
5. Run final status/diff/commit verification, stage explicit paths, push the branch, and create Draft PR #14 titled `fix: preserve frozen intent across M1 replay` with the required provenance and gate status. Do not merge it or begin PR #15.

## Verification and handoff

- The repaired replay must be a new append-only artifact; historical first-run/replay files and all protected M2 artifacts remain unchanged.
- The final response must report exact commit/PR/CI outcomes, distinguish offline cache replay from real external validation, and include only successful Git action directives.
