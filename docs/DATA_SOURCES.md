# Real Data Sources

All free, public, usable for training data. Categorized by what each source is actually good for — most PV databases give structured fields but weak/no narrative text; a few sources give rich narrative text. Need both.

## A. Rich free-text narratives (best for extraction + narrative-writing tasks)

| Source | URL | What you get | Access |
|---|---|---|---|
| **VAERS** | vaers.hhs.gov/data/datasets.html | Real adverse-event case narratives (`SYMPTOM_TEXT` field) + structured fields (age, sex, vaccine, onset, outcome) | Free CSV download, no registration |
| **PubMed/PMC case reports** | ncbi.nlm.nih.gov/pmc (Open Access subset) | Full-text published adverse-drug-reaction case reports — richly written, includes medical history, current conditions, drug timeline, outcome, discussion. Closest real-world match to what you're training extraction on | Free via PMC OA API/bulk FTP, no registration for OA subset |
| **CADEC corpus** | data.csiro.au (search "CADEC") | Forum posts (AskaPatient) annotated with adverse drug events, drug names, symptoms | Free download, research use |
| **ADE-Corpus-V2** | huggingface.co/datasets/ade_corpus_v2 | Sentences from case reports labeled drug↔adverse-event relations, ready-to-use format | Free, direct HF `datasets` library load |
| **n2c2 / i2b2 NLP shared task datasets** | n2c2.dbmi.hms.harvard.edu | Clinical notes labeled for medication extraction, condition/timeline extraction — not PV-specific but directly transferable to your extraction task's entity/field structure | Free, requires registration + data use agreement |
| **MIMIC-III/IV** | physionet.org | Large ICU clinical note corpus — medical history, current conditions, medication sections richly represented | Free, requires credentialed access (short training course + agreement, no cost) |

## B. Structured adverse-event data (good for schema-field grounding, weak on narrative)

| Source | URL | What you get | Access |
|---|---|---|---|
| **openFDA FAERS API** | api.fda.gov/drug/event.json | Structured ICSR data: drug, reaction (MedDRA PT already coded!), demographics, outcome, seriousness | Free API, no key needed for low volume, free key for higher rate limits |
| **FAERS quarterly data files** | fda.gov/faers | Bulk ASCII/XML files, same data as API but full historical volume | Free download |
| **VigiAccess (WHO)** | vigiaccess.org | Aggregated case counts by drug/reaction — good for sanity-checking event frequency, not case-level narrative | Free, public interface only (no bulk case export) |
| **EudraVigilance (EMA)** | adrreports.eu | Aggregated European ICSR line listings by drug/reaction | Free, public portal, aggregated only |
| **MHRA Yellow Card (UK)** | yellowcard.mhra.gov.uk | Interactive Drug Analysis Profiles — aggregated reaction data per drug | Free, public |
| **TGA DAEN (Australia)** | tga.gov.au (search "DAEN") | Database of Adverse Event Notifications, some narrative-like description field | Free, public database search/export |
| **Health Canada Canada Vigilance** | health-products.canada.ca/canada-vigilance | Structured ADR database | Free, public, bulk extract available |

## C. Drug/label reference data (for expectedness & reportability logic)

| Source | URL | What you get | Access |
|---|---|---|---|
| **DailyMed (NLM)** | dailymed.nlm.nih.gov | Full drug label text including Adverse Reactions section — needed to determine "listed vs unlisted" (expectedness) for reportability rule engine | Free, bulk XML download, no registration |
| **SIDER** | sideeffects.embl.de | Drug↔side-effect associations extracted from package inserts, structured | Free download |
| **ClinicalTrials.gov results** | clinicaltrials.gov | Structured AE tables from posted trial results — real trial-context adverse event data | Free API/bulk download |

## D. MedDRA reference (for the coding task's retrieval index)

| Source | URL | What you get | Access |
|---|---|---|---|
| **MedDRA MSSO** | meddra.org | Official term hierarchy (PT/LLT/HLT/HLGT/SOC) — needed to build the embedding index in `scripts/build_meddra_index.py` | **Licensed** — free for some categories (e.g., academic/nonprofit, or low-revenue subscribers), check current MSSO subscription terms; NOT freely redistributable in bulk without a license |
| **[[meddra_learn]] curated content** | `/home/bala/Desktop/Bak.dev/meddra-learn/data/sample_meddra.json` | Your own already-curated 27 SOC profiles + terminology rules — usable as seed/example data without licensing concern since it's your own written educational content, not the raw MedDRA term list itself | Already in-house |

**Important**: MedDRA's actual term list is licensed by MSSO, not freely redistributable in bulk even though summaries/education about it (like meddra-learn's content) are fine to write yourself. Check your organization's MedDRA subscription status before bulk-downloading/embedding the full term list — if no active subscription, start the embedding index with a smaller openly-available subset (e.g., terms that appear in openFDA FAERS data, which already ships MedDRA-coded reactions under FDA's public data terms) rather than the full official hierarchy.

## Recommended combination for Phase 1

1. **VAERS** narratives + **PMC case reports** → primary real-data anchor for extraction/narrative-writing training (Phase 3)
2. **openFDA FAERS** → structured field grounding + already-MedDRA-coded reactions (safe subset for the coding index, avoids MSSO licensing question)
3. **DailyMed** → expectedness/reportability rule engine input (Phase 5)
4. **ADE-Corpus-V2 + CADEC** → quick-start NER-style training pairs, useful even before synthetic generator is built
5. Synthetic generation (per `DISTILLATION_PLAN.md`) → fills volume gaps real sources can't cover, especially for causality-assessment training examples which real public databases don't provide with reasoning attached

## Licensing note

VAERS/FAERS/DailyMed/ClinicalTrials.gov data is US government public domain — no restriction. PMC Open Access subset is explicitly licensed for reuse (check individual article license, most are CC-BY). CADEC and ADE-Corpus-V2 are released for research use — fine for this project. MedDRA is the one genuine licensing constraint — handle per the note in section D above.
