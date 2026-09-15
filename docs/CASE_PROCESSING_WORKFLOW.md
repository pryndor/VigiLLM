# Real-Life Case Processing Workflow

The four tasks in `ARCHITECTURE.md` are the reasoning-heavy pieces of a bigger real pharmacovigilance case lifecycle. This doc maps the full real-world flow so VigiLLM's scope is honest about what it automates vs what stays a rule-engine step vs what must stay human — not a toy narrative-in/JSON-out demo.

## Standard ICSR lifecycle (ICH E2B-aligned)

```
1. Case Receipt / Intake
   source: spontaneous report, literature, clinical trial, solicited (patient support program), regulatory authority
   captured: reporter type, country, receipt date, raw narrative/document

2. Minimum Criteria Validation
   ICH's 4 criteria — a case is NOT a valid ICSR unless all 4 present:
     identifiable patient, identifiable reporter, suspect drug, adverse event
   invalid case -> triggers follow-up request, does not proceed to full processing

3. Duplicate Search
   check against existing case database (same patient/drug/event/reporter/date pattern)
   confirmed duplicate -> merge/close, do not double-count in aggregate reporting

4. Case Data Entry / Structuring          <-- VigiLLM Task 1 (extraction)
   turn raw narrative/documents into structured fields

5. MedDRA Coding                          <-- VigiLLM Task 2 (retrieval-based coding)
   code each adverse event and relevant medical history term

6. Medical/Scientific Assessment          <-- VigiLLM Task 3 (hybrid rule engine + LLM)
   seriousness criteria (rule engine)
   expectedness / listedness vs product label (rule engine, needs DailyMed data)
   causality assessment (LLM, WHO-UMC-trained)

7. Narrative Writing                      <-- VigiLLM Task 4
   generate case narrative for the safety report / submission

8. Quality Review (QC)
   second human reviewer sign-off — NOT automated, see Human-in-the-Loop below

9. Follow-Up Information Request
   if case incomplete (fails step 2, or medical reviewer needs more info) —
   generate a follow-up query back to the original reporter

10. Regulatory Reportability Determination & Submission
    expedited vs periodic, timeline calculation (rule engine)
    E2B(R2)/E2B(R3) transmission — this is where MedExtract's mapper/ picks up
    (see ARCHITECTURE.md integration note, Phase 7)

11. Case Closure / Aggregate Reporting
    closed case feeds into periodic aggregate reports (PSUR/PBRER) —
    this is VigiDraft's scope (separate project), not VigiLLM's
```

## What VigiLLM currently covers vs what's a gap

| Lifecycle step | VigiLLM coverage | Status |
|---|---|---|
| 1. Intake | `case_metadata` schema fields (source_type, reporter_type, receipt_date, country) | Schema field added; not yet a trained task — currently just captured data, no model reasoning needed here |
| 2. Minimum criteria validation | `case_metadata.minimum_criteria_met` — deterministic rule engine, not a model task | Schema field added; rule engine implementation still Phase 5 (`ARCHITECTURE.md` rule engine section) |
| 3. Duplicate search | `case_metadata.duplicate_check_status` field present | **Gap** — no duplicate-detection logic planned yet. Real implementation would need a similarity-search step (embedding-based, same technique as MedDRA retrieval) over existing case records. Not scoped into a phase yet — add as Phase 5.5 if the team wants it before deployment |
| 4. Extraction | Task 1 | Phase 3 |
| 5. MedDRA coding | Task 2 | Phase 4 |
| 6. Assessment | Task 3 | Phase 5 |
| 7. Narrative writing | Task 4 | Phase 3 (paired with extraction) |
| 8. QC review | — | **Deliberately not automated** — human sign-off required, see ARCHITECTURE.md's Human-in-the-Loop note |
| 9. Follow-up requests | `case_metadata.follow_up_status` / `follow_up_requested_info` fields present | **Gap** — no follow-up-letter generation task planned. Could be a light addition to Task 4's narrative-writing adapter (same backbone, new task prefix `draft_followup_request`) — cheap to add once Task 4 is working, since it reuses the same fine-tuned writing capability |
| 10. Reportability/submission | `reportability` schema object, rule engine (Phase 5) | Actual E2B XML generation stays MedExtract's job, not VigiLLM's — VigiLLM only determines *whether/when* to report, not the XML transmission itself |
| 11. Aggregate reporting | out of scope | Belongs to VigiDraft (separate project) |

## Why this matters for training data realism

`scripts/generate_scenarios.py` now generates `case_metadata` with a realistic mix of complete and incomplete cases (~5-10% missing a minimum criterion, mirroring real intake messiness) rather than assuming every synthetic case is a clean, fully-formed report. This matters because:

- Real intake data is often incomplete — a model trained only on clean synthetic cases will not handle the messy real cases it will actually see
- The minimum-criteria and follow-up fields give the rule engine (Phase 5) real signal to test against, not just happy-path data

## Recommended scope decision for the team

Two gaps identified above (duplicate search, follow-up-letter generation) are **not yet in any roadmap phase**. Recommend:
- **Duplicate search**: add as an explicit Phase 5.5 in `ROADMAP.md` if the team wants case-processing realism before Phase 6 deployment — otherwise defer to Phase 7 (post-integration) since it's lower risk than causality/reportability
- **Follow-up letter generation**: cheap addition to Task 4's existing adapter, low priority, can be folded into Phase 3 as an extra task-prefix once extraction/writing gate passes

Ask the team which of these two gaps matters enough to schedule now vs defer — this doc exists so that decision is made explicitly rather than the scope quietly staying narrower than "real case processing" implies.
