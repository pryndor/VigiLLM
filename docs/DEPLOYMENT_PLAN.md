# Deployment Plan

All steps zero cost. No always-on server required.

## Tested finding (2026-09-15): BioMistral is not on Ollama's public registry

`ollama pull biomistral` (and `biomistral:7b`, `biomistral:latest`, `biomistral/biomistral`) all fail with "pull model manifest: file does not exist" — confirmed directly, not assumed. This doesn't block anything: the merge→GGUF→`ollama create` path below never depended on Ollama's registry having BioMistral pre-listed, since we convert our own trained/merged model ourselves. It just means anyone wanting to test *stock* BioMistral locally via Ollama before fine-tuning needs to download it from HuggingFace and convert to GGUF manually first — there's no one-line `ollama pull` shortcut for the untrained base model. For quick local testing (e.g. the Phase 2.5 zero-shot baseline) without that conversion step, `ollama pull qwen2.5:7b-instruct` works directly — it's the documented fallback backbone and is officially on Ollama's registry.

## Step 1: Merge LoRA into base

```bash
python scripts/merge_lora.py \
  --base_model BioMistral/BioMistral-7B \
  --adapter ./checkpoint_new \
  --output ./merged_model
```

## Step 2: Convert to GGUF (for Ollama)

```bash
# using llama.cpp's convert script
python llama.cpp/convert_hf_to_gguf.py ./merged_model --outfile vigillm.gguf --outtype q4_k_m
```

`q4_k_m` quantization keeps quality reasonable while shrinking file size enough to run comfortably on a normal laptop (no GPU required for inference — CPU inference with Ollama is workable for a 7B model at this quantization, just slower than GPU).

## Step 3: Push GGUF to HuggingFace Hub

Same private org repo used for checkpoints, or a separate `org/vigillm-gguf` repo — keep raw adapter checkpoints and final quantized releases separate so team members pulling for inference don't need to download training-only checkpoint metadata.

## Step 4: Local deployment per team member (Ollama)

```bash
ollama create vigillm -f Modelfile   # Modelfile points to the downloaded .gguf
ollama run vigillm
```

Each member runs this independently on their own machine — no shared server needed, no hosting cost. Query via Ollama's local REST API (`localhost:11434`) from any internal tool.

## Step 5: Shared demo (optional, for stakeholder review)

HuggingFace Spaces free CPU tier — host a Gradio/Streamlit demo UI backed by the merged model (or a smaller quantization if CPU inference is too slow). Free, shareable link, good enough for showing the model working without needing infrastructure spend. Not for production/high-volume use — CPU-tier Spaces is genuinely slow, treat it as a demo surface only.

## Step 6: Integration path (future)

Structured JSON output from VigiLLM's extraction/assessment tasks can feed directly into [`MedExtract`](../../MedExtract)'s `mapper/` stage (E2B field mapping) instead of MedExtract's current OCR-only pipeline — meaning VigiLLM eventually becomes a second, LLM-based input path into the same downstream E2B/CIOMS generation MedExtract already does. Not in scope until VigiLLM's extraction accuracy clears its evaluation gate (see `TRAINING_PLAN.md`).

## What "deployment" explicitly does NOT mean here

No cloud GPU server, no paid inference API, no autoscaling concerns. This is intentionally a "download the file, run it locally" deployment model — matches the zero-cost constraint and avoids maintaining infrastructure the team would otherwise need to pay for or keep patched.
