# M2-T03 Blind Human-Adjudication Protocol

Version: `m2-t03-human-adjudication-protocol.v1`

This protocol is frozen before any human review begins. Its UTF-8 bytes are
SHA-256 bound in the pending M2-T03 bundle and in the blind review template.
Changing the protocol after review starts invalidates the associated review
artifacts.

## Review purpose

The review records whether a paper's title and abstract support one evidence
slot for the frozen ResearchIntent. It does not score papers, select papers,
or produce the M2-T04 six-item score.

## Allowed sources

The reviewer may use only these fields:

- `paper_id`
- `title`
- `abstract`
- `source`
- `source_id`
- `url`
- `ResearchIntent`

The source identity fields identify the record; they are not evidence beyond
the title and abstract. ResearchIntent supplies the target problem and the
context in which transferability must still be judged.

## Disallowed sources

The reviewer must not use:

- full paper text
- citation counts
- author reputation
- journal rank or venue prestige
- reranker or dense scores
- final selection or rank results
- network supplementation
- user feedback
- machine advisory labels, reasons, or excerpts as a decision aid

## EvidenceSlot vocabulary

`PROBLEM_EXISTENCE` identifies a title/abstract statement of the problem,
gap, limitation, or need that the target work must address.

`CURRENT_METHODS` identifies existing methods, systems, algorithms, or already
adopted solution paths.

`METHOD_TRANSFERABILITY` identifies evidence that a method may transfer to the
target problem or a similar domain. The review reason must state that the
transfer still requires validation on the target problem.

`IMPLEMENTATION_PATH` identifies components, workflows, architecture,
engineering steps, or an executable implementation route.

`EVALUATION_BASIS` identifies datasets, benchmarks, metrics, experimental
designs, or evaluation methods.

## SupportLevel vocabulary

`DIRECT` means the title or abstract explicitly and directly supports the
slot.

`INDIRECT` means a similar task, adjacent domain, or transferable method gives
indirect support. The reason must explicitly state that validation on the
target problem is still required.

`HYPOTHETICAL` means the title or abstract only proposes, expects, or
speculates that the evidence may work; no validated result is asserted.

## Human fields

The reviewer fills only the following fields:

```text
verdict: SUPPORTED | REJECTED
evidence_slot
support_level
grounded_reason
supporting_excerpt
reviewer_id
reviewed_at_utc
notes
```

For `SUPPORTED`, the slot, support level, non-blank grounded reason, reviewer
identity, and a non-blank exact substring from the title or abstract are
required. For `REJECTED`, slot and support level are null; the reason and
reviewer identity remain required, and the excerpt may be empty.

## Derived comparison

The system computes the comparison only after all human judgments are sealed.
The reviewer never enters this value.

- `CONFIRM`: human verdict is `SUPPORTED` and human slot/support level equal
  the machine advisory slot/support level.
- `REVISE`: human verdict is `SUPPORTED` but either the slot or support level
  differs from the machine advisory.
- `REJECT`: human verdict is `REJECTED`.

These comparison labels are reproducible diagnostics. They must not be
converted into an M2-T04 score or used to claim real classification quality
before the separate human-result gate is complete.
