# VigiLLM Task Status

Plain checklist of what's actually done vs not, as of 2026-09-15. Cross-reference: `docs/ROADMAP.md` has the full phased plan with exit conditions — this file is the quick "what's done right now" view.

## Phase 0 — Infrastructure setup

| Task | Status |
|---|---|
| GitHub private repo created | ✅ Done — `github.com/pryndor/VigiLLM`, pushed, 5 commits |
| Local git identity / commits working | ✅ Done |
| HuggingFace account authenticated on this machine | ✅ Done — user `Pryndor` |
| HF private repo: adapter checkpoints | ✅ Done — `Pryndor/vigillm-adapter` |
| HF private repo: training dataset | ✅ Done — `Pryndor/vigillm-dataset` |
| Weights & Biases account authenticated | ✅ Done — entity `balathepharmacist-drugvigil` |
| W&B project created | ✅ Done — `vigillm` project live |
| **Kaggle account created + phone-verified** | ✅ Done — user confirmed 2026-09-15 |
| Colab account verified (overflow compute) | ✅ Done — user confirmed 2026-09-15 |
| Local dev environment (venv + dependencies) | ✅ Done — `venv/` has full `requirements.txt` stack installed and tested |
| Ollama installed + running locally | ✅ Done — running as background process |
| Candidate backbone models pulled locally for testing | ✅ Done — `qwen2.5:7b-instruct` and `cniongolo/biomistral`, both tested |
| Schema locked (`schema/case_schema.json`) | ✅ Done — includes `case_metadata` real-life intake fields added after review |
| Team members invited (GitHub, HF, W&B) | ⏸️ **Deliberately deferred** — user starting solo first, will invite later |
| Shared training-queue tracking board (per `TEAM_WORKFLOW.md`) | ❌ Not created — only matters once a second person joins |

## Phase 1 — Data foundation

| Task | Status |
|---|---|
| `scripts/generate_scenarios.py` written | ✅ Done |
| Scenario generator tested at scale | ✅ Done — 500 scenarios, 100% schema-valid, 0 birth-year/date-ordering bugs after fixes |
| `scripts/build_meddra_index.py` written | ✅ Done |
| MedDRA index tested against live openFDA API | ✅ Done — 300-term test index, retrieval quality confirmed on 3 realistic queries |
| Known gap documented: openFDA caps at 1000 terms/request | ✅ Documented in `docs/DATA_SOURCES.md` |
| Real data downloaded: VAERS narratives | ❌ Not done |
| Real data downloaded: FAERS bulk/API extracts | ❌ Not done (only the small 300-term MedDRA test pull, not full FAERS case data) |
| Real data downloaded: PMC open-access case reports | ❌ Not done |
| Real data downloaded: CADEC / ADE-Corpus-V2 | ❌ Not done |
| Hand-curated CIOMS/ICH E2B example cases (20-30) | ❌ Not done |

## Phase 2 — Distillation

| Task | Status |
|---|---|
| Query OpenBioLLM-8B + BioMistral-7B on synthetic scenarios | ❌ Not started |
| WHO-UMC deterministic rule-checker (`scripts/rule_engine/who_umc.py`) | ❌ Not started |
| Filtered distilled training set pushed to HF Hub | ❌ Not started |

## Phase 2.5 — Zero-shot baseline

| Task | Status |
|---|---|
| Formal baseline eval (extraction F1, narrative quality) on untrained backbones | ❌ Not started — only an informal single-prompt sanity check done (BioMistral answered a causality question weakly/rambling vs Qwen2.5's clean response) |

## Phase 3-7 — Fine-tuning, MedDRA coding, case assessment, deployment, integration

All ❌ **not started** — blocked on Phase 0's Kaggle setup and Phase 1's real data pulls.

## Documentation

| Doc | Status |
|---|---|
| `README.md` | ✅ Done |
| `docs/ARCHITECTURE.md` | ✅ Done, updated with tested BioMistral/Qwen2.5 findings |
| `docs/DISTILLATION_PLAN.md` | ✅ Done |
| `docs/TRAINING_PLAN.md` | ✅ Done, updated with real HF repo names |
| `docs/DEPLOYMENT_PLAN.md` | ✅ Done, updated with tested Ollama findings |
| `docs/TEAM_WORKFLOW.md` | ✅ Done, updated with real account/repo names |
| `docs/ROADMAP.md` | ✅ Done, includes Phase 2.5 and Phase 5.5 additions |
| `docs/DATA_SOURCES.md` | ✅ Done |
| `docs/CASE_PROCESSING_WORKFLOW.md` | ✅ Done |
| `schema/case_schema.json` | ✅ Done, locked |
| This file (`TASKS.md`) | ✅ Done |

## Working method

User has chosen to go **one task at a time, in order** — not parallelizing across tasks. Update this file's status after each task closes before starting the next one, so it always reflects exactly where things stand.

## Current task queue (in order)

1. ✅ ~~Kaggle account setup~~ — done 2026-09-15
2. ✅ ~~Colab account verification~~ — done 2026-09-15
3. ⏳ **Real data downloads** (Phase 1): VAERS → FAERS → PMC case reports → CADEC/ADE-Corpus-V2 → hand-curated CIOMS examples — up next
4. Distillation pipeline (Phase 2)
5. Zero-shot baseline eval (Phase 2.5)
6. First fine-tune: extraction + narrative writing (Phase 3)
7. MedDRA coding integration (Phase 4)
8. Case assessment: causality + rule engine (Phase 5)
9. Deployment (Phase 6)
10. Team invites + integration (Phase 7, whenever user is ready to bring others in)
