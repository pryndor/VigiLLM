#!/usr/bin/env python3
"""Query teacher models on synthetic scenario seeds to build distillation
training data, per docs/DISTILLATION_PLAN.md.

Two tasks, two teachers:
  - assess_causality: OpenBioLLM-8B reasons about WHO-UMC causality category
    for a given structured case. Cross-checked against the deterministic
    rule-based oracle (scripts/rule_engine/who_umc.py) -- disagreements are
    flagged, not silently trusted, per the plan's QC section.
  - write_narrative: BioMistral-7B turns the same structured case into
    free-text clinical narrative language. Paired with the known ground-truth
    case JSON to also produce reverse-direction extract_case_data examples.

Two backends, pick with --backend:
  - hf (default): transformers, loaded 4-bit via bitsandbytes. No local
    install/download needed on your machine -- run this on Kaggle Notebooks
    (free T4/P100 GPU, no thermal load on your own hardware) or Colab.
    See notebooks/query_teachers_kaggle.ipynb for a ready-to-run notebook.
  - ollama: local Ollama REST API (http://localhost:11434). Kept for
    reference/CPU-only runs -- do NOT use this to pull new multi-GB models
    on hardware that overheats/shuts down mid-download. Prefer --backend hf
    on Kaggle instead.

Usage (Kaggle/Colab, GPU runtime):
    python scripts/query_teachers.py --backend hf \
        --scenarios data/synthetic_scenarios.jsonl \
        --count 50 --out data/distilled_training_set.jsonl

Usage (local Ollama, only if you already have the models pulled):
    python scripts/query_teachers.py --backend ollama \
        --scenarios data/synthetic_scenarios.jsonl \
        --count 50 --out data/distilled_training_set.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.rule_engine.who_umc import classify_causality, VALID_CATEGORIES

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_OPENBIOLLM_MODEL = "openbiollm"
OLLAMA_BIOMISTRAL_MODEL = "cniongolo/biomistral"

# HF Hub repos for the --backend hf path (run on Kaggle/Colab GPU).
HF_OPENBIOLLM_MODEL = "aaditya/Llama3-OpenBioLLM-8B"
HF_BIOMISTRAL_MODEL = "BioMistral/BioMistral-7B"

CAUSALITY_PROMPT = """You are a pharmacovigilance expert applying the WHO-UMC system for standardised case causality assessment.

Given the structured case data below, assess the causal relationship between the suspect drug and the adverse event(s). Use these six categories: certain, probable, possible, unlikely, conditional, unassessable.

Case data:
{case_json}

Respond with ONLY a JSON object in this exact form, no other text:
{{"category": "<one of the six categories>", "reasoning": "<2-4 sentences explaining your assessment against the WHO-UMC criteria>"}}"""

NARRATIVE_PROMPT = """You are a clinical documentation specialist. Write a concise clinical case narrative (3-6 sentences) describing this adverse drug event case, in the style of a case report or ICSR narrative. Use natural clinical prose, not a list.

Case data:
{case_json}

Write only the narrative text, no headers or preamble."""


def ollama_generate(model: str, prompt: str, json_mode: bool = False, timeout: int = 240) -> str:
    import requests
    payload = {"model": model, "prompt": prompt, "stream": False}
    if json_mode:
        payload["format"] = "json"
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"]


_hf_pipelines: dict = {}


def hf_pipeline(model_id: str):
    """Load (and cache) a 4-bit quantized causal-LM pipeline. Meant for a
    Kaggle/Colab GPU runtime -- do not run this on CPU-only local hardware."""
    if model_id in _hf_pipelines:
        return _hf_pipelines[model_id]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    _hf_pipelines[model_id] = pipe
    return pipe


def hf_generate(model_id: str, prompt: str, max_new_tokens: int = 400) -> str:
    pipe = hf_pipeline(model_id)
    chat = [{"role": "user", "content": prompt}]
    result = pipe(
        chat,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    generated = result[0]["generated_text"]
    # chat-templated pipelines return the full conversation; take the last turn
    if isinstance(generated, list):
        return generated[-1]["content"].strip()
    return generated[len(prompt):].strip()


def generate(backend: str, task: str, prompt: str, json_mode: bool = False) -> str:
    if backend == "ollama":
        model = OLLAMA_OPENBIOLLM_MODEL if task == "causality" else OLLAMA_BIOMISTRAL_MODEL
        return ollama_generate(model, prompt, json_mode=json_mode)
    model = HF_OPENBIOLLM_MODEL if task == "causality" else HF_BIOMISTRAL_MODEL
    return hf_generate(model, prompt)


def case_for_prompt(scenario: dict) -> dict:
    """Strip the fields the teacher is being asked to produce, so it can't just echo them back."""
    stripped = dict(scenario)
    stripped.pop("causality_assessment", None)
    stripped.pop("narrative_text", None)
    stripped.pop("reportability", None)
    return stripped


def query_causality(scenario: dict, backend: str) -> dict | None:
    prompt = CAUSALITY_PROMPT.format(case_json=json.dumps(case_for_prompt(scenario), indent=2))
    raw = generate(backend, "causality", prompt, json_mode=True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    category = parsed.get("category", "").strip().lower()
    if category not in VALID_CATEGORIES:
        return None
    return {"category": category, "reasoning": parsed.get("reasoning", "")}


BAD_NARRATIVE_PREFIXES = ("please write", "i need", "include all", "write a causality")


def is_valid_narrative(text: str) -> bool:
    """BioMistral's instruction-following is inconsistent (documented in
    docs/ARCHITECTURE.md's zero-shot findings) -- it sometimes echoes back an
    instruction instead of writing a narrative. Cheap heuristic reject, not a
    substitute for the human spot-check DISTILLATION_PLAN.md calls for."""
    lowered = text.strip().lower()
    if len(text.split()) < 15:
        return False
    if lowered.startswith(BAD_NARRATIVE_PREFIXES):
        return False
    return True


def query_narrative(scenario: dict, backend: str) -> str | None:
    prompt = NARRATIVE_PROMPT.format(case_json=json.dumps(case_for_prompt(scenario), indent=2))
    text = generate(backend, "narrative", prompt, json_mode=False).strip()
    return text if is_valid_narrative(text) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["hf", "ollama"], default="hf",
                     help="hf = transformers on GPU (Kaggle/Colab, default). "
                          "ollama = local Ollama REST API -- avoid pulling new "
                          "models on hardware prone to overheating/shutdown.")
    ap.add_argument("--scenarios", default="data/synthetic_scenarios.jsonl")
    ap.add_argument("--count", type=int, default=50, help="how many scenarios to process")
    ap.add_argument("--offset", type=int, default=0, help="skip this many scenarios first (for resuming)")
    ap.add_argument("--out", default="data/distilled_training_set.jsonl")
    ap.add_argument("--flagged-out", default="data/distilled_flagged.jsonl",
                     help="causality examples where OpenBioLLM disagreed with the rule engine")
    ap.add_argument("--skip-causality", action="store_true", help="only generate narratives")
    ap.add_argument("--skip-narrative", action="store_true", help="only generate causality assessments")
    args = ap.parse_args()

    scenarios = []
    with open(args.scenarios) as f:
        for line in f:
            scenarios.append(json.loads(line))
    scenarios = scenarios[args.offset:args.offset + args.count]

    out_path = Path(args.out)
    flagged_path = Path(args.flagged_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_flagged = 0
    n_causality_failed = 0
    n_narrative_rejected = 0

    with out_path.open("a") as out_f, flagged_path.open("a") as flagged_f:
        for i, scenario in enumerate(scenarios):
            report_id = scenario["case_metadata"]["report_id"]

            if not args.skip_causality:
                t0 = time.time()
                try:
                    result = query_causality(scenario, args.backend)
                except Exception as e:
                    print(f"[{i+1}/{len(scenarios)}] {report_id} causality REQUEST_FAILED ({e.__class__.__name__}), skipping")
                    result = None
                if result is None:
                    n_causality_failed += 1
                else:
                    rule_category = classify_causality(scenario)
                    agrees = result["category"] == rule_category
                    record = {
                        "task": "assess_causality",
                        "input": case_for_prompt(scenario),
                        "output": result,
                        "rule_engine_category": rule_category,
                        "agrees_with_rule_engine": agrees,
                    }
                    if agrees:
                        out_f.write(json.dumps(record) + "\n")
                        n_written += 1
                    else:
                        flagged_f.write(json.dumps(record) + "\n")
                        n_flagged += 1
                print(f"[{i+1}/{len(scenarios)}] {report_id} causality={'ok' if result else 'PARSE_FAIL'} "
                      f"({time.time()-t0:.1f}s)")

            if not args.skip_narrative:
                t0 = time.time()
                try:
                    narrative = query_narrative(scenario, args.backend)
                except Exception as e:
                    print(f"[{i+1}/{len(scenarios)}] {report_id} narrative REQUEST_FAILED ({e.__class__.__name__}), skipping")
                    narrative = None
                if narrative is None:
                    n_narrative_rejected += 1
                    print(f"[{i+1}/{len(scenarios)}] {report_id} narrative REJECTED (low quality) "
                          f"({time.time()-t0:.1f}s)")
                else:
                    record = {
                        "task": "write_narrative",
                        "input": case_for_prompt(scenario),
                        "output": narrative,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    # reverse-direction pair: narrative -> structured extraction target
                    extract_record = {
                        "task": "extract_case_data",
                        "input": narrative,
                        "output": case_for_prompt(scenario),
                    }
                    out_f.write(json.dumps(extract_record) + "\n")
                    n_written += 2
                    print(f"[{i+1}/{len(scenarios)}] {report_id} narrative generated ({time.time()-t0:.1f}s)")

    print(f"\n{n_written} training examples written to {out_path}")
    print(f"{n_flagged} causality examples flagged (disagree with rule engine) -> {flagged_path}")
    if n_causality_failed:
        print(f"{n_causality_failed} causality queries failed to parse (skipped)")
    if n_narrative_rejected:
        print(f"{n_narrative_rejected} narratives rejected by quality filter (skipped)")


if __name__ == "__main__":
    main()
