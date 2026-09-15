# Training Plan

## Prerequisites (must exist before training starts)

1. `schema/case_schema.json` locked (already in repo)
2. `data/distilled_training_set.jsonl` built per [`DISTILLATION_PLAN.md`](DISTILLATION_PLAN.md)
3. Real-data anchors (FAERS/VAERS extracts, hand-curated CIOMS examples) merged in
4. HuggingFace Hub private repo created for checkpoints (`Pryndor/vigillm-adapter`)
5. Weights & Biases project created, shared with team

## Method: QLoRA fine-tuning

- Base model loaded in 4-bit (`bitsandbytes`), LoRA adapter rank 16-32 on attention + MLP projection layers
- Library stack: `transformers` + `peft` + `trl` (`SFTTrainer`)
- Fits on free-tier GPU: Kaggle T4/P100 (16GB) or Colab free T4

## Session mechanics (per training turn, per team member)

```bash
# 1. Sync code
git pull

# 2. Pull latest checkpoint (skip on very first run)
hf download Pryndor/vigillm-adapter --local-dir ./checkpoint

# 3. Train for the session's time budget
python scripts/train.py \
  --base_model BioMistral/BioMistral-7B \
  --dataset data/distilled_training_set.jsonl \
  --resume_from_checkpoint ./checkpoint \
  --output_dir ./checkpoint_new \
  --max_steps <budget-dependent> \
  --report_to wandb

# 4. Push updated checkpoint
hf upload Pryndor/vigillm-adapter ./checkpoint_new

# 5. Mark turn complete on the team's shared tracking board (see TEAM_WORKFLOW.md)
```

## Session budgeting

- Kaggle: ~9hr GPU session cap, 30hr/week quota per account
- Colab free: shorter, less predictable session length (can disconnect early) — use for overflow/short runs only, not primary training
- Checkpoint every N steps (not just at session end) — if a Kaggle/Colab session disconnects mid-run, you lose at most N steps, not the whole session. Configure `save_steps` conservatively (e.g., every 100 steps) given free-tier disconnect risk.

## Training data mix (avoid catastrophic forgetting across tasks)

Shuffle all four task types together within each epoch — do not train task-by-task sequentially. Sequential single-task training causes the model to forget earlier tasks as it specializes on the current one. `SFTTrainer` with a pre-shuffled combined dataset handles this by default as long as the dataset file itself is interleaved, not concatenated by task block.

## Evaluation gates (check before advancing to next phase)

| Task | Metric | Target before moving on |
|---|---|---|
| Extraction | Field-level F1 vs held-out labeled set | ≥ 0.80 on core fields (events, drugs, dates) |
| MedDRA coding | Exact PT match rate (retrieval + rerank) | ≥ 0.85 top-1, ≥ 0.95 top-5 |
| Causality | Agreement rate vs WHO-UMC rule-based oracle | ≥ 0.75 (causality is inherently judgment-heavy, perfect agreement not expected) |
| Narrative writing | Human PV-reviewer rating (1-5 scale, sample of 20) | Average ≥ 4.0, zero factual contradictions with input JSON |

Do not deploy or demo a phase that hasn't cleared its gate — regulatory-adjacent output needs the bar met, not "good enough for now."

## Iteration and code changes during training

Because checkpoints live on HF Hub (git-based) and code lives on GitHub separately, any team member can:
- Modify `scripts/train.py` or the LoRA config on their own branch
- Test the change on a short run without touching the shared checkpoint (train from the current checkpoint into a *new* checkpoint name, compare eval metrics, only merge into the shared lineage if it's an improvement)
- Open a PR describing what changed and why, before pushing to the main adapter repo

This is the same discipline as normal software development — training script changes go through review just like any other code change.
