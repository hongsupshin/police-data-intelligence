# Discrepancy Audit

The discrepancy audit inverts the enrichment pipeline's premise. Enrichment
treats the TJI/OAG database as ground truth and fills missing fields from
news coverage; the audit runs the same pipeline over incidents whose
records are **complete** and flags fields where coverage contradicts the
official record. The official record is the audit object, not the
reference. (Motivation: TJI reports that the government data itself
contains errors that TJI sometimes has to fix.)

## Semantics: ADVISE-only, human-verified

A flagged mismatch has three possible causes — government error, news
error, extraction error — and nothing in the system can distinguish them.
Flags are therefore ADVISE-only:

- The audit never writes to the database.
- No flag is reported as an error until a human verifies it. Publishing
  unverified flags would be unvalidated accusations against official
  records.
- Every flag carries its provenance (source URLs, exact quotes, extraction
  confidence) so a verifier can adjudicate from the worksheet alone.

The three shipped agents apply unchanged and only make the audit more
conservative: the relevance judge's veto escalates wrong-article runs
(escalated incidents are never flagged), the race verifier nulls
unattested race values (no extraction → no race flag), and the conflict
annotator remains advisory.

## Flag lifecycle

1. `scripts/run_audit.py` runs the pipeline per incident and compares
   COMPLETE outcomes against the DB via `AUDIT_FIELD_COMPARATORS`.
2. A flag is emitted only on genuine contradiction: both values present,
   neither an exact nor a fuzzy match. Lower specificity is not a
   contradiction ("Sgt. Duran" is consistent with "John Duran"; a city is
   consistent with a street address in that city).
3. Flags below MEDIUM extraction confidence are emitted with
   `suppressed=true`: kept in the JSON report for taxonomy analysis,
   excluded from the worksheet and headline counts.
4. Reportable flags land in `output/audit/audit_{run_id}_worksheet.csv`,
   ordered by severity. A verifier fills in `verification_status`:
   `db_error` (official record wrong), `news_error`, `extraction_error`,
   or `unresolved`.
5. `src.audit.report.summarize_verified()` recomputes verified flag
   precision (share of resolved flags confirmed as `db_error`) with a 95%
   Wilson interval — the headline metric.

## Severity scheme

| Severity | Fields | Rationale |
| --- | --- | --- |
| high | officer_name, civilian_name, outcome, civilian_race | Wrong person, wrong survival status, misrecorded race — the accountability-relevant errors |
| medium | weapon, civilian_age, time_of_day | Consequential but noisier categories |
| low | location_detail | Granularity-prone (city vs street address) |

Demotions: an age difference of ≤1 is `low` (rounding/birthday noise);
officers_shot `civilian_race` is `medium` because the race verifier only
runs on civilians_shot, so that extraction is unverified.

## What is not audited, and why

- **circumstance** — free text on the news side, only semi-structured in
  the DB (`incident_result_of`, narratives; nothing for officers_shot). A
  deterministic comparator would be noise; an LLM entailment judge would
  be a new unvalidated judge, which the earn-it methodology forbids
  shipping ungated. Extractions are still recorded as verifier context.
  Extension point: add a validated comparator to `AUDIT_FIELD_COMPARATORS`.
- **officer_name for civilians_shot** — structurally absent from the
  official record (the OAG civilian-shot form records officer
  age/race/gender but no names; 0 of 1,674 records have one). The
  comparison yields `no_ground_truth`, never a flag.
- **weapon / time_of_day for officers_shot** — no such DB columns.

## Sampling restrictions (pilot)

`src.audit.sampling.select_audit_incidents` requires:

- Every auditable-and-populatable column non-NULL (a missing DB value
  cannot be contradicted). Clean-DB pools: 506 civilians / 151 officers.
- Exactly one victim row. The reference fetch uses `LIMIT 1` joins, so
  multi-victim incidents would compare coverage against an arbitrary
  victim; the offline gate confirmed this produces spurious name/race
  flags on saved artifacts.
- DEV ∪ TEST ids excluded. Pilot threshold iteration is tuning, and
  tuning must never touch the frozen eval splits.

Preflight (`src.audit.preflight.verify_db_clean`) refuses to run against
a database whose row counts deviate from the clean load (1,674 / 282).

## Evaluation design (no free ground truth)

- **Precision** — human verification of every reported flag via the
  worksheet; `summarize_verified` computes it.
- **Recall** — a TJI known-errors seed set (records TJI already knows are
  wrong); measure whether the audit flags them. Pending TJI input.
- **Specificity** — the false-flag suite (`--false-flag-suite`): audit
  incidents where saved holdout reports show every evaluated field
  matched exactly (gov and news demonstrably agreed, ≥3 exact fields,
  unanimous across reports). Any flag there is presumed false. This is
  the audit's analog of the adversarial suite.

## Reference sources

Flags carry `reference_source` (default `tji_db`). The comparison layer
takes any `ReferenceProvider`, so a raw-OAG provider (data.world
`tji/raw-and-processing`, `original/OIS.xlsx`, joined on the
`ois_report_no` column both incident tables already carry) can be added
without touching comparators, flags, or reports — enabling a
TJI-clean-vs-OAG-raw arm later.

## CLI

```bash
# Inspect the stratified sample + cost estimate (no runs, no LLM)
python scripts/run_audit.py civilians_shot --dry-run --limit 100

# Offline: replay saved enrichment outputs through the flag layer (no LLM)
python scripts/run_audit.py civilians_shot --from-saved output/enrichment

# Live pilot (checkpointed; ~$0.20/incident)
python scripts/run_audit.py civilians_shot --limit 100

# Resume a killed/failed run (skips checkpointed incidents)
python scripts/run_audit.py civilians_shot --resume <run_id>

# Smoke test on explicit incidents
python scripts/run_audit.py civilians_shot --incident-ids 160 161 162

# False-flag suite (live, or offline with --from-saved)
python scripts/run_audit.py civilians_shot --false-flag-suite --limit 30
```

Outputs: report JSON (`output/audit/audit_{run_id}.json`, per-field flag
rates with Wilson CIs), verification worksheet CSV, and per-incident
checkpoints under `output/audit/runs/{run_id}/`.
