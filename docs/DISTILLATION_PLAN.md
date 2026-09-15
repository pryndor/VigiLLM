# Multi-Teacher Distillation Plan

## Why distillation instead of weight-merging

BioMistral-7B (Mistral architecture), OpenBioLLM-8B (Llama-3 architecture), and Meditron-7B (Llama-2 architecture) cannot be merged at the weight level — different hidden dimensions, tokenizers, and vocab sizes make direct merging (SLERP/TIES/DARE via `mergekit`) impossible across these three. Weight-merging only works between fine-tunes that share the same base architecture.

Instead: **knowledge distillation**. Query each teacher model on the task it's strongest at, capture its output as training data, then fine-tune ONE student backbone on the combined dataset. The student absorbs reasoning patterns from both teachers without any weight-level compatibility requirement — and since only *text outputs* are used (not the teacher's weights), the final model isn't bound by OpenBioLLM's Llama license.

Meditron-7B is dropped from this plan — superseded in quality by the other two, not worth the extra compute budget.

## Teacher assignment

| Teacher | Query for | Why |
|---|---|---|
| **OpenBioLLM-8B** | Clinical reasoning: causality judgment scenarios, differential diagnosis reasoning, seriousness-relevant clinical context | Strongest of the three on clinical reasoning benchmarks |
| **BioMistral-7B** | Medical terminology: symptom-to-term phrasing variety, MedDRA-adjacent vocabulary, drug/condition naming | Already the chosen student backbone — but querying its base (pre-fine-tune) form for terminology diversity still adds value as training signal alongside OpenBioLLM's reasoning |

## Pipeline

```
1. Generate scenario seeds
   → randomized structured case data (drug, event, demographics, history)
     sampled programmatically from schema/case_schema.json fields

2. Query teachers (inference only, free GPU — Kaggle/Colab)
   → OpenBioLLM-8B: "Given this case data, what is the WHO-UMC causality
     category and why?" → capture reasoning + category
   → BioMistral-7B: "Describe this adverse event in clinical narrative
     language" → capture varied phrasing

3. Collect as instruction-tuning pairs
   → format: {"task": "assess_causality", "input": <case json>,
              "output": <OpenBioLLM's reasoning + category>}
   → format: {"task": "extract_case_data" (reverse direction),
              "input": <BioMistral's narrative>, "output": <case json>}

4. Merge with:
   → real narratives from FAERS/VAERS public datasets (ground truth anchor,
     prevents drifting entirely into synthetic-only patterns)
   → hand-curated CIOMS/ICH E2B textbook examples (small, high quality)

5. Fine-tune BioMistral-7B (the student) via QLoRA on the combined dataset
   → see docs/TRAINING_PLAN.md for the mechanics
```

## Quality control on distilled data

Teacher outputs are not ground truth — they're a starting point. Before use in fine-tuning:

- **Causality labels**: cross-check OpenBioLLM's causality category against a deterministic WHO-UMC rule-checker (build this as a small Python function — WHO-UMC's algorithm is publicly documented and rule-based, not proprietary). Discard or flag training examples where the LLM's category disagrees with the rule-based oracle — these are exactly the noisy examples that would teach the model to reason incorrectly.
- **Narrative phrasing**: spot-check a sample of BioMistral-generated narratives for medical plausibility before bulk-including in training set — a team member reviews a random 5% sample each generation batch.

## Output of this phase

A dataset file (`data/distilled_training_set.jsonl`) containing multi-task instruction pairs, pushed to the team's HuggingFace Hub dataset repo, ready for the training script in `docs/TRAINING_PLAN.md`.
