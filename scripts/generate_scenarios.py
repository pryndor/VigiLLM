#!/usr/bin/env python3
"""Generate randomized synthetic case scenarios conforming to schema/case_schema.json.

These scenarios are the seed input for the distillation pipeline (docs/DISTILLATION_PLAN.md) —
teacher models (OpenBioLLM-8B, BioMistral-7B) are queried against these to produce
narrative text and causality reasoning as training targets. This script only produces
the structured side; it does not call any model.

Usage:
    python scripts/generate_scenarios.py --count 200 --out data/synthetic_scenarios.jsonl
    python scripts/generate_scenarios.py --count 200 --drug-list data/faers_drugs.txt \
        --event-list data/faers_events.txt --out data/synthetic_scenarios.jsonl
"""
import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

# Fallback seed lists — replace with real extracted terms from FAERS/VAERS as soon as
# those downloads exist (see docs/DATA_SOURCES.md). These are placeholders only,
# enough to exercise the pipeline end-to-end before real data is wired in.
DEFAULT_DRUGS = [
    "Metformin", "Atorvastatin", "Amoxicillin", "Ibuprofen", "Losartan",
    "Omeprazole", "Sertraline", "Levothyroxine", "Amlodipine", "Metoprolol",
    "Paracetamol", "Ciprofloxacin", "Warfarin", "Insulin glargine", "Prednisone",
]
DEFAULT_EVENTS = [
    "Nausea", "Headache", "Rash", "Dizziness", "Fatigue",
    "Vomiting", "Diarrhoea", "Angioedema", "Hepatotoxicity", "Anaphylactic reaction",
    "Acute kidney injury", "Hypoglycaemia", "QT prolongation", "Thrombocytopenia",
    "Stevens-Johnson syndrome",
]
DEFAULT_CONDITIONS = [
    "Type 2 diabetes mellitus", "Hypertension", "Hyperlipidemia", "Asthma",
    "Chronic kidney disease", "Osteoarthritis", "Depression", "Hypothyroidism",
    "Atrial fibrillation", "Gastroesophageal reflux disease",
]

OUTCOMES = ["recovered", "recovering", "not_recovered", "recovered_with_sequelae", "fatal", "unknown"]
SERIOUSNESS = ["death", "life_threatening", "hospitalization", "disability",
               "congenital_anomaly", "other_medically_important"]
CHALLENGE = ["positive", "negative", "not_done", "unknown"]
SOURCE_TYPES = ["spontaneous", "literature", "clinical_trial", "solicited_pss", "regulatory_authority"]
REPORTER_TYPES = ["physician", "pharmacist", "nurse", "other_hcp", "consumer", "lawyer"]
COUNTRIES = ["IN", "US", "GB", "DE", "FR", "CA", "AU", "JP"]


def make_case_metadata(case_id: int) -> dict:
    # Real intake isn't always clean — model a realistic mix of complete/incomplete cases
    # so downstream validity-check and follow-up logic has something to actually exercise.
    min_criteria = {
        "identifiable_patient": random.random() > 0.05,
        "identifiable_reporter": random.random() > 0.10,
        "suspect_drug": random.random() > 0.02,
        "adverse_event": random.random() > 0.02,
    }
    is_valid = all(min_criteria.values())
    return {
        "report_id": f"SYN-{case_id:06d}",
        "source_type": random.choice(SOURCE_TYPES),
        "reporter_type": random.choice(REPORTER_TYPES),
        "receipt_date": random_date((2025, 2026)),
        "country": random.choice(COUNTRIES),
        "minimum_criteria_met": min_criteria,
        "duplicate_check_status": "not_checked",
        "follow_up_status": "not_required" if is_valid else "requested",
        "follow_up_requested_info": None if is_valid else "missing minimum criteria — see minimum_criteria_met",
    }


def load_list(path: str | None, default: list[str]) -> list[str]:
    if not path:
        return default
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"list file not found: {path}")
    items = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return items or default


CURRENT_YEAR = 2026


def random_date(base_year_range=(2018, 2026)) -> str:
    year = random.randint(*base_year_range)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def random_date_after_birth(birth_year: int, min_offset: int = 0, max_offset: int | None = None) -> str:
    """Random date guaranteed >= birth_year — prevents e.g. a 4-year-old with a
    medical-history condition dated 2005 (predates birth), which fixed decade
    ranges produced regardless of sampled age."""
    lo = birth_year + min_offset
    hi = CURRENT_YEAR if max_offset is None else min(CURRENT_YEAR, birth_year + max_offset)
    hi = max(hi, lo)
    return random_date((lo, hi))


def days_after(iso_date: str, n_days: int) -> str:
    return (date.fromisoformat(iso_date) + timedelta(days=n_days)).isoformat()


def make_scenario(drugs: list[str], events: list[str], conditions: list[str], case_id: int) -> dict:
    case_metadata = make_case_metadata(case_id)
    age = random.randint(1, 95)
    sex = random.choice(["male", "female"])
    birth_year = CURRENT_YEAR - age

    n_history = random.randint(0, 3)
    medical_history = [
        {"condition": c, "start_date": random_date_after_birth(birth_year, 0, max(age - 1, 0)),
         "end_date": None, "notes": None}
        for c in random.sample(conditions, k=min(n_history, len(conditions)))
    ]

    n_current = random.randint(0, 2)
    remaining_conditions = [c for c in conditions if c not in {h["condition"] for h in medical_history}]
    current_conditions = [
        {"condition": c, "start_date": random_date_after_birth(birth_year, max(age - 5, 0), age), "notes": None}
        for c in random.sample(remaining_conditions, k=min(n_current, len(remaining_conditions)))
    ]

    n_concomitant = random.randint(0, 2)
    concomitant_medications = [
        {"name": d, "dose": None, "route": "oral", "indication": None,
         "start_date": random_date_after_birth(birth_year, max(age - 3, 0), age), "stop_date": None}
        for d in random.sample(drugs, k=min(n_concomitant, len(drugs)))
    ]

    remaining_drugs = [d for d in drugs if d not in {c["name"] for c in concomitant_medications}]
    suspect_drug = random.choice(remaining_drugs if remaining_drugs else drugs)
    drug_start = random_date_after_birth(birth_year, max(age - 1, 0), age)
    suspect_drugs = [{
        "name": suspect_drug, "dose": None, "route": "oral", "indication": None,
        "start_date": drug_start, "stop_date": None, "batch_number": None,
    }]

    n_events = random.randint(1, 2)
    onset = random_date_after_birth(birth_year, max(age - 1, 0), age)
    adverse_events = [{
        "term": e,
        "meddra_pt": None,
        "meddra_llt": None,
        "meddra_hlt": None,
        "meddra_soc": None,
        "onset_date": onset,
        "resolution_date": days_after(onset, random.randint(1, 180)) if random.random() > 0.3 else None,
        "outcome": random.choice(OUTCOMES),
        "seriousness_criteria": random.sample(SERIOUSNESS, k=random.randint(0, 2)),
    } for e in random.sample(events, k=min(n_events, len(events)))]

    return {
        "case_metadata": case_metadata,
        "patient": {"age": age, "age_unit": "years", "sex": sex, "weight_kg": None},
        "medical_history": medical_history,
        "current_conditions": current_conditions,
        "concomitant_medications": concomitant_medications,
        "suspect_drugs": suspect_drugs,
        "adverse_events": adverse_events,
        "rechallenge": random.choice(CHALLENGE),
        "dechallenge": random.choice(CHALLENGE),
        "causality_assessment": None,   # filled by teacher model, see DISTILLATION_PLAN.md
        "reportability": None,          # filled by rule engine, see ARCHITECTURE.md
        "narrative_text": None,         # filled by teacher model
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=100, help="number of scenarios to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drug-list", default=None, help="optional text file, one drug name per line")
    ap.add_argument("--event-list", default=None, help="optional text file, one event term per line")
    ap.add_argument("--condition-list", default=None, help="optional text file, one condition per line")
    ap.add_argument("--out", default="data/synthetic_scenarios.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    drugs = load_list(args.drug_list, DEFAULT_DRUGS)
    events = load_list(args.event_list, DEFAULT_EVENTS)
    conditions = load_list(args.condition_list, DEFAULT_CONDITIONS)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in range(args.count):
            scenario = make_scenario(drugs, events, conditions, case_id=i)
            f.write(json.dumps(scenario) + "\n")

    print(f"wrote {args.count} scenarios to {out_path}")


if __name__ == "__main__":
    main()
