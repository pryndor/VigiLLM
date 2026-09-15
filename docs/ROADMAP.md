# Roadmap — Slow Start, Phased

Each phase has a concrete, checkable exit condition. Don't start the next phase until the current one's condition is met — this project fails if it tries to do everything at once with a volunteer team on free compute.

## Phase 0 — Setup (no GPU needed, do this first)

- [ ] Create GitHub private repo, invite team members
- [ ] Create HuggingFace org (free), invite team members as collaborators
- [ ] Create shared Weights & Biases project, invite team members
- [ ] Each member creates their own free Kaggle account + verifies phone (required for GPU access) and free Colab account
- [ ] Set up the training-queue tracking board (GitHub Issue or shared sheet)
- [ ] Review and finalize `schema/case_schema.json` as a team — this is the one thing expensive to change later

**Exit condition**: everyone can log into all four systems, schema is agreed and committed.

## Phase 1 — Data foundation (no GPU needed)

- [ ] Download FAERS/VAERS/PMC-case-reports/ADE-Corpus-V2/CADEC per [`DATA_SOURCES.md`](DATA_SOURCES.md)
- [ ] Hand-curate 20-30 CIOMS/ICH E2B textbook example cases (highest quality, small volume)
- [ ] Write the synthetic scenario generator (`scripts/generate_scenarios.py`) — randomized structured case data sampled from schema fields
- [ ] Build MedDRA PT term embedding index (`scripts/build_meddra_index.py`) using `sentence-transformers/all-MiniLM-L6-v2` — reuse content from [`meddra-learn`](../../meddra-learn) project's curated SOC/terminology data as seed material where useful

**Exit condition**: `data/` has real-data anchors + a working synthetic generator + a queryable MedDRA embedding index.

## Phase 2 — Distillation (light GPU use, inference only)

- [ ] Query OpenBioLLM-8B and BioMistral-7B per [`DISTILLATION_PLAN.md`](DISTILLATION_PLAN.md) on synthetic scenarios
- [ ] Build the WHO-UMC deterministic rule-checker (`scripts/rule_engine/who_umc.py`) — used to quality-filter distilled causality labels
- [ ] Filter and merge into `data/distilled_training_set.jsonl`, push to HF Hub dataset repo
- [ ] Spot-check 5% sample manually for plausibility

**Exit condition**: a reviewed, quality-filtered multi-task training dataset exists on HF Hub.

## Phase 3 — First fine-tune: extraction + narrative writing only

Start with just the two inverse tasks (extraction and narrative writing) before adding MedDRA coding or causality — smallest scope that produces a demoable result.

- [ ] Run first QLoRA training turn per [`TRAINING_PLAN.md`](TRAINING_PLAN.md), extraction + writing tasks only
- [ ] Evaluate against Phase 3 gate: extraction F1 ≥ 0.80 on core fields
- [ ] Iterate (more data, more steps, or schema fixes) until gate clears

**Exit condition**: extraction/writing gate passes. This is the first real milestone — a model that can turn a narrative into structured data and back.

## Phase 4 — Add MedDRA coding

- [ ] Wire the retrieval pipeline (embedding index + LLM reranker) into the same backbone via the `code_meddra_term` task prefix
- [ ] Fine-tune reranking behavior on distilled examples
- [ ] Evaluate: ≥0.85 top-1 exact PT match

**Exit condition**: MedDRA coding gate passes.

## Phase 5 — Add case assessment (causality/seriousness/reportability)

Highest regulatory risk — do this last, with the most scrutiny.

- [ ] Implement deterministic rule engine for seriousness/reportability first (no model needed, pure logic, fast to build and verify)
- [ ] Implement deterministic minimum-criteria validation (`case_metadata.minimum_criteria_met`) — see `CASE_PROCESSING_WORKFLOW.md` step 2
- [ ] Fine-tune causality judgment on distilled + rule-checker-filtered examples
- [ ] Evaluate against WHO-UMC oracle agreement rate ≥ 0.75
- [ ] Have an actual PV professional review a sample of causality outputs before considering this phase done — automated metrics aren't sufficient sign-off for this task

**Exit condition**: causality gate passes AND human PV review sample approved.

## Phase 5.5 — Duplicate search (optional, team decides — see CASE_PROCESSING_WORKFLOW.md)

- [ ] Embedding-based similarity search over existing case records (same technique as MedDRA retrieval index)
- [ ] Flag possible duplicates for human review, never auto-merge

**Exit condition**: team explicitly decided to include or defer this — do not let it stay silently unscoped.

## Phase 6 — Deployment

- [ ] Merge LoRA, convert to GGUF, push to HF Hub per [`DEPLOYMENT_PLAN.md`](DEPLOYMENT_PLAN.md)
- [ ] Each member pulls and runs locally via Ollama
- [ ] Optional: HF Spaces demo for stakeholder review

**Exit condition**: every team member has a working local instance.

## Phase 7 — Integration (future, not scoped yet)

- [ ] Explore feeding VigiLLM's structured extraction output into [`MedExtract`](../../MedExtract)'s mapping pipeline as a second input path alongside its OCR pipeline

## What "slow" means in practice

Phases 0-2 need no training GPU time at all — do them thoroughly before touching Kaggle quota. Phase 3 is the first real training investment and the first point where the team should pause and honestly assess whether results justify continuing to Phase 4/5, rather than assuming the full four-task system automatically.
