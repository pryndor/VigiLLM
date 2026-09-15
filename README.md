# VigiLLM

Open-source, zero-cost, team-trainable LLM for pharmacovigilance case processing.

## What it does (target capabilities)

1. **Narrative extraction** — free-text case narrative → structured JSON (medical history, current conditions, concomitant medications, suspect drugs, adverse events, onset/outcome, rechallenge/dechallenge)
2. **MedDRA coding** — extracted event terms → correct PT/LLT/HLT/HLGT/SOC codes
3. **Case assessment** — structured case data → causality (WHO-UMC), seriousness criteria, expectedness, regulatory reportability + timeline
4. **Narrative writing** — structured case data → CIOMS/ICH E2B-style narrative text (inverse of task 1)

## Why this exists

Existing tools in this workspace ([`MedExtract`](../MedExtract), [`Vigi-Vault`](../Vigi-Vault)) do rule-based OCR/parsing/E2B-mapping. VigiLLM is different: it's a **fine-tuned open-weight LLM** meant to handle the reasoning-heavy parts (causality judgment, free-text understanding, narrative generation) that rule-based parsers can't. Long-term, VigiLLM's output can feed into MedExtract's/Vigi-Vault's E2B pipelines rather than replace them.

## Core design decision

**One backbone, multi-teacher distilled, single LoRA adapter, multi-task.** Not four separate models. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for why.

- **Backbone**: BioMistral-7B (Apache 2.0, Mistral-based, PubMed-pretrained)
- **Distillation teachers**: OpenBioLLM-8B (clinical reasoning), BioMistral-7B itself (terminology) — see [`docs/DISTILLATION_PLAN.md`](docs/DISTILLATION_PLAN.md)
- **MedDRA coding**: retrieval-based (embedding + rerank), not free generation — avoids hallucinated codes
- **Causality/reportability**: hybrid — deterministic rule engine (Python) for regulatory logic + LLM for nuanced judgment

## Zero-cost infrastructure

```
GitHub (private repo)          <- code, training scripts, configs, rule engine
        |
Kaggle/Colab (per-member GPU)  <- burst compute, pull code + checkpoint, train, push
        |
HuggingFace Hub (private repo) <- model checkpoints, LoRA adapters, dataset
        |
W&B (free tier)                <- shared experiment tracking
```

Full detail: [`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md)

## Documentation map

| Doc | Contents |
|---|---|
| [`docs/CASE_PROCESSING_WORKFLOW.md`](docs/CASE_PROCESSING_WORKFLOW.md) | Real ICSR lifecycle (intake → validation → duplicate check → coding → assessment → QC → follow-up → submission), mapped to what VigiLLM covers vs gaps |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Model design, why single backbone, task routing, schema-driven design |
| [`docs/DISTILLATION_PLAN.md`](docs/DISTILLATION_PLAN.md) | How BioMistral + OpenBioLLM knowledge gets combined into one model |
| [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) | Step-by-step QLoRA fine-tuning process, session budgeting |
| [`docs/DEPLOYMENT_PLAN.md`](docs/DEPLOYMENT_PLAN.md) | GGUF conversion, Ollama serving, HF Spaces demo, no-server-cost deploy |
| [`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md) | How multiple members share compute/checkpoints without collision |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phased plan — what to build first, slow-start order |
| [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | Real-world free data sources — where case narratives, structured AE data, and drug labels come from |
| [`schema/case_schema.json`](schema/case_schema.json) | Locked structured-data schema all tasks conform to |

## Status

See [`TASKS.md`](TASKS.md) for the current done/not-done checklist. Short version: Phase 0 infra (GitHub/HF/W&B/local env) is complete; Kaggle account setup and Phase 1 real-data pulls are the next blockers before any training can start.

## License note

All chosen models (BioMistral-7B, Qwen2.5-7B fallback) are Apache 2.0 — fully open, no redistribution restriction. OpenBioLLM-8B (Llama-3 based, Llama community license) is used only as a **distillation teacher at inference time** — its weights are never merged or redistributed, only its text outputs are used as training data for the Apache-2.0 backbone. This keeps the final model's license clean.
