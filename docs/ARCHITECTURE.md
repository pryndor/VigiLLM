# Architecture

## Design principle

One backbone model, one multi-task LoRA adapter, task-prefixed instructions. Not four separate fine-tunes. Reasons:

- One model to deploy, one Ollama pull, one GGUF file
- Shared learning across tasks (extraction and narrative-writing are inverses of each other — training both makes each stronger)
- Simpler for a small team to maintain — one training script, one checkpoint lineage

## Backbone

**BioMistral-7B** (Mistral-7B base, continued-pretrained on PubMed Central, Apache 2.0).

Chosen over Qwen2.5-7B-Instruct because it already carries medical vocabulary before any task fine-tuning starts — less training needed to get baseline medical fluency. Qwen2.5-7B-Instruct is the fallback if BioMistral's instruction-following proves too weak after testing (BioMistral is a base/continued-pretrain model, not instruction-tuned — may need an extra instruction-tuning pass first, see [`DISTILLATION_PLAN.md`](DISTILLATION_PLAN.md)).

## The four tasks, and why each is handled differently

### 1. Narrative extraction (text → structured JSON)

Generative task on the fine-tuned backbone. Input: raw narrative. Output: JSON conforming to [`schema/case_schema.json`](../schema/case_schema.json).

Optional accuracy booster (not required for v1): a small encoder model (Bio_ClinicalBERT, ~110M params, near-zero GPU cost) runs first as a token-classification NER pass to pre-tag entities (drug names, conditions, dates). The LLM then assembles the final structured JSON using these tags as hints. Add this only if pure-LLM extraction accuracy plateaus below target.

### 2. MedDRA coding (event term → PT/SOC code)

**Not free generation.** An LLM asked to output a MedDRA code from memory will hallucinate plausible-but-wrong codes — unacceptable for regulatory use.

Instead: retrieval-based pipeline —
1. Pre-compute embeddings for every MedDRA PT term using `sentence-transformers/all-MiniLM-L6-v2` (or `PubMedBERT` embeddings for better medical-domain match) — one-time index build
2. At inference, embed the extracted event text, cosine-similarity search top-5 candidate PT terms from the index
3. Fine-tuned backbone reranks the 5 candidates and picks the best match, also returns SOC/HLT via MedDRA hierarchy lookup (deterministic, not generated)

This guarantees output is always a real MedDRA term that exists in the index — no hallucinated codes possible by construction.

### 3. Case assessment (causality / seriousness / reportability)

Hybrid, not pure LLM:
- **Seriousness criteria** (death, life-threatening, hospitalization, disability, congenital anomaly, other medically important) — deterministic rule engine in plain Python, driven directly by structured fields already extracted in task 1. This is regulatory logic, not reasoning — hard-coding it is more reliable AND easier for any team member to edit without retraining a model.
- **Reportability timeline** (e.g., 15-day expedited vs periodic) — same, deterministic rule engine keyed off jurisdiction + seriousness + expectedness.
- **Causality judgment** (WHO-UMC categories: certain/probable/possible/unlikely/unclassifiable) — this genuinely needs reasoning over temporal relationship, dechallenge/rechallenge, alternative causes. This part uses the fine-tuned backbone, trained on WHO-UMC-algorithm-derived training examples (see [`DISTILLATION_PLAN.md`](DISTILLATION_PLAN.md) for how training labels are generated).

Rule engine lives in `scripts/rule_engine/` — no model, no training required, plain conditional logic team can read and edit directly.

### 4. Narrative writing (structured JSON → text)

Generative task on the same fine-tuned backbone, same LoRA adapter as task 1, just the reverse instruction prefix. No separate model.

## Task routing at inference time

```
### Task: extract_case_data
### Input: <narrative text>
### Output: <json>
```
```
### Task: code_meddra_term
### Input: <event text>
### Output: <candidate PT list + reranked choice>
```
```
### Task: assess_causality
### Input: <structured case JSON>
### Output: <causality category + reasoning>
```
```
### Task: write_narrative
### Input: <structured case JSON>
### Output: <narrative text>
```

Same prompt-prefix convention used consistently in training data and inference calls.

## Why the schema is locked first

Every task's training data must conform to the same JSON structure ([`schema/case_schema.json`](../schema/case_schema.json)), otherwise extraction output won't match what narrative-writing expects as input, and causality assessment won't know which fields to read. Schema changes after training data exists mean regenerating that data — lock it before writing a single training example.

## Human-in-the-loop requirement

This system produces **decision-support drafts only**. Causality judgment and reportability determination directly affect regulatory submission timelines — no output from this model is submission-ready without human PV-professional review and sign-off. This constraint must be visible in any UI built on top of this model, not just in documentation.
