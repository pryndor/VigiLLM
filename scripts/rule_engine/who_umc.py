#!/usr/bin/env python3
"""Deterministic WHO-UMC causality classifier -- the rule-based oracle used to
cross-check teacher-model (OpenBioLLM) causality labels during distillation
QC, per docs/DISTILLATION_PLAN.md's "Quality control on distilled data" section.

Source: WHO-UMC's own published causality criteria (data/raw/who_umc_causality.pdf,
free/public -- "The use of the WHO-UMC system for standardised case causality
assessment"). The six categories and their assessment criteria are quoted almost
verbatim from that document's Table 2.

This is a deliberately SIMPLIFIED approximation. The real WHO-UMC criteria require
clinical judgment calls that free-text-independent structured fields alone can't
answer -- "plausible vs reasonable time relationship", "event definitive
pharmacologically or phenomenologically", "could be explained by disease" all need
a clinician reading the actual case. This function only uses what's mechanically
available in schema/case_schema.json: rechallenge/dechallenge outcomes and the
presence of confounding concomitant medications/current conditions as a proxy for
"could also be explained by disease or other drugs". It exists purely to flag
teacher-model outputs that are wildly inconsistent with these hard structural
signals (e.g. an LLM calling something "certain" with a negative dechallenge) --
not to replace expert causality assessment. Disagreements get discarded/flagged
per the distillation plan, not auto-corrected.

Usage:
    from scripts.rule_engine.who_umc import classify_causality
    category = classify_causality(case_dict)
"""
from __future__ import annotations

from datetime import date

VALID_CATEGORIES = [
    "certain", "probable", "possible", "unlikely", "conditional", "unassessable",
]

IMPLAUSIBLE_INTERVAL_DAYS = 730  # >2 years of stable use before onset -> weak time relationship


def _earliest_onset_after_start(case: dict) -> int | None:
    """Days between the earliest suspect-drug start_date and the earliest
    adverse-event onset_date, or None if either is missing/unparseable."""
    drug_dates = [d.get("start_date") for d in (case.get("suspect_drugs") or []) if d.get("start_date")]
    event_dates = [e.get("onset_date") for e in (case.get("adverse_events") or []) if e.get("onset_date")]
    if not drug_dates or not event_dates:
        return None
    try:
        start = min(date.fromisoformat(d) for d in drug_dates)
        onset = min(date.fromisoformat(d) for d in event_dates)
    except ValueError:
        return None
    return (onset - start).days


def _has_data_gaps(case: dict) -> bool:
    """Missing rechallenge/dechallenge info AND no adverse-event onset date is
    treated as insufficient/contradictory data -- WHO-UMC's 'unassessable'
    trigger ("cannot be judged because information is insufficient")."""
    rechallenge = case.get("rechallenge", "unknown")
    dechallenge = case.get("dechallenge", "unknown")
    events = case.get("adverse_events") or []
    has_onset = any(e.get("onset_date") for e in events)
    min_criteria = case.get("case_metadata", {}).get("minimum_criteria_met") or {}
    meets_minimum = all(min_criteria.values()) if min_criteria else True

    return (
        rechallenge == "unknown"
        and dechallenge == "unknown"
        and not has_onset
    ) or not meets_minimum


def _has_confounders(case: dict) -> bool:
    """Proxy for 'could also be explained by ... other drugs': any concomitant
    medication on record. Deliberately excludes current_conditions -- in
    practice that field is usually just the indication the suspect drug was
    prescribed for (e.g. "pulmonary tuberculosis" alongside isoniazid), not an
    independent alternative explanation for the adverse event, so treating it
    as a confounder made this trigger on almost every real case."""
    return bool(case.get("concomitant_medications"))


def classify_causality(case: dict) -> str:
    """Return one of VALID_CATEGORIES based on schema-available structured signals."""
    if _has_data_gaps(case):
        return "unassessable"

    follow_up_status = case.get("case_metadata", {}).get("follow_up_status")
    if follow_up_status == "requested":
        # Data explicitly pending (e.g. biopsy, PCR panel) -- WHO-UMC 'conditional':
        # "more data for proper assessment needed, or additional data under examination".
        return "conditional"

    interval = _earliest_onset_after_start(case)
    if interval is not None and interval > IMPLAUSIBLE_INTERVAL_DAYS:
        # Event onset more than 2 years after starting a drug with no recorded dose
        # change is a poor time relationship -- WHO-UMC 'unlikely'.
        return "unlikely"

    rechallenge = case.get("rechallenge", "unknown")
    dechallenge = case.get("dechallenge", "unknown")
    confounders = _has_confounders(case)

    if dechallenge == "negative":
        # Event persisted/worsened despite withdrawal -- time relationship argument
        # for the drug weakens sharply; WHO-UMC 'unlikely'.
        return "unlikely"

    if dechallenge == "positive" and rechallenge == "positive" and not confounders:
        # Rechallenge satisfactory + no alternative explanation -- WHO-UMC 'certain'
        # requires more (a definitive pharmacological/phenomenological event, which
        # isn't inferable structurally), so this rule undershoots real 'certain'
        # cases but never overshoots into it without the strongest available signal.
        return "certain"

    if dechallenge == "positive" and not confounders:
        # Reasonable time relationship assumed from a positive dechallenge signal;
        # unlikely to be attributed to disease/other drugs since none are on record.
        return "probable"

    if dechallenge == "positive" and confounders:
        # Positive dechallenge, but an alternative explanation exists on record --
        # WHO-UMC 'possible': "could also be explained by disease or other drugs".
        return "possible"

    # dechallenge not_done/unknown with a complete case and no other strong signal:
    # default to 'possible', WHO-UMC's own stated default for the two most common
    # real-world categories when the reasonable time-relationship criterion is met
    # but the rest is ambiguous.
    return "possible"


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases_jsonl", help="path to a jsonl file of case_schema.json-shaped records")
    ap.add_argument("--compare-field", default="causality_assessment",
                     help="if present and non-null, compare rule output against case[field]['result']")
    args = ap.parse_args()

    total = 0
    agree = 0
    with open(args.cases_jsonl) as f:
        for line in f:
            case = json.loads(line)
            predicted = classify_causality(case)
            total += 1
            existing = case.get(args.compare_field)
            if existing and existing.get("result"):
                actual = existing["result"]
                match = "MATCH" if predicted == actual else "DIFFER"
                if predicted == actual:
                    agree += 1
                report_id = case.get("case_metadata", {}).get("report_id", f"#{total}")
                print(f"{report_id}: rule={predicted:12s} existing={actual:12s} {match}")
            else:
                print(f"case {total}: rule={predicted}")
    if agree:
        print(f"\n{agree}/{total} agree with existing causality_assessment")
