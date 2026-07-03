# Relevance-Judge Veto Audits: Provenance

This document records every audit of the relevance judge's veto precision: what
was audited, on which data, with what result, and which artifacts survive. It
exists because the original earn-it audit scripts were run from `/tmp` and were
not preserved; an evaluation methodology built on audits should not itself have
unauditable audits.

## 1. Verification of the ten live holdout vetoes (2026-07-02)

The canonical June-2026 holdout (`output/eval/holdout_*_20260613_231717.json`)
records ten `irrelevant_sources` escalations — completions the relevance judge
vetoed into human review: civilians_shot incidents 5, 9, 158, 277, 284, 591,
1332 and officers_shot incidents 59, 357, 33. Each was re-verified offline by
reading the saved retrieved articles (`output/enrichment/*_escalate.json`)
against the database anchor. `scripts/verify_relevance_vetoes.py` reconstructs
the full dossiers; the verdicts below are a human reading of that output.

**Result: 10 of 10 vetoes genuine wrong-article. Zero false vetoes.**

| Incident | Anchor (date, city, subject) | What the extraction had committed | Verdict |
| --- | --- | --- | --- |
| civ 5 | 2015-09-05, Houston, 21M non-fatal | `officer_name='Darren Goforth'`, outcome Fatal — Deputy Goforth was *murdered* Aug 28, 2015; articles cover his killing and funeral | genuine |
| civ 9 | 2015-09-18, San Antonio, 22M non-fatal | `civilian_name='Mark Anthony Coleman'` — a Sep 18 manhunt story for a suspect in an earlier non-police shooting | genuine |
| civ 158 | 2016-07-04, Scurry, 29M fatal | Circumstance/officer from a Snyder motel *murder* (civilian suspect Nicole Mann, victim 33) — no police involvement | genuine |
| civ 277 | 2017-03-08, Rosharon, 22M non-fatal | The Leon Jacob murder-for-hire case: same date, "Rosharon" matched only because Jacob is imprisoned there; nobody was shot | genuine |
| civ 284 | 2017-03-23, Houston, 36M non-fatal | `civilian_name='Jeremy R. Dowell'` from a *Washington State* OIS (Lynnwood, Jan 30, 2017, fatal); only link is "36-year-old male" | genuine |
| civ 591 | 2019-01-09, Spring, 57M non-fatal | The Midland mass shooting (wrong city, wrong event, Fatal) | genuine¹ |
| civ 1332 | 2023-01-14, Call, 54M non-fatal | Names from the same event complex (Gosey/Chief Jackson) but no retrieved text states police shot the civilian; `time_of_day` provably from a 2026 Fort Worth release | genuine² |
| off 59 | 2017-07-22, Houston, officer D. Chippy injured | `civilian_age='15'`, outcome fatal — the Balch Springs killing of Jordan Edwards; "Chippy" also matched a sports idiom and a Glasgow chip-shop forum | genuine |
| off 357 | 2018-03-06, San Antonio, officer injured, suspect 61 Hispanic | `officer_name='Robert Deckard'`, outcome fatal — trial coverage of Deckard's *2013* killing, which began the same day | genuine³ |
| off 33 | 2016-09-07, Katy, officer injured | `officer_name='Benjamin Marconi'` — Det. Marconi was assassinated in San Antonio in *November* 2016; name came from a sitemap aggregation page | genuine |

¹ One retrieved article (a Jan 8, 2019 southwest-Houston hotel OIS, man injured)
is the nearest miss in the set: right date window, wrong part of the metro (the
anchor city is specifically SPRING). It is not what the extraction committed
from, and no retrieved text ties it to the anchor record.

² The most instructive case: the article describes the *officers_shot side* of
the same real-world event (L.C. Gosey Jr. shooting Newton Police Chief Will
Jackson while officers served an eviction notice). No retrieved text states
that police shot Gosey — the fact the civilians_shot record asserts — so a
completion would have committed unsupported values. The dataset divergence the
pipeline handles at extraction reappears here as a relevance trap.

³ Incident 357's report ID predates the idempotent-ETL rebuild (see §3); its
anchor was recovered by ground-truth match (city + suspect age) to clean rows
75/76 — one event, two officer victims: San Antonio, 2018-03-06, officer
injured, 61-year-old Hispanic suspect.

**Pattern:** three of ten extractions named famous fallen officers from
unrelated, heavily covered events (Goforth, Deckard, Marconi), and a fourth
(Jordan Edwards) drew on a nationally covered killing. Famous-case gravity is
the dominant wrong-article failure mode; date or city coincidence supplies the
match, prominence supplies the text.

### Reproduce

```bash
python scripts/verify_relevance_vetoes.py > veto_dossiers.txt
```

The script is read-only. It refuses to run unless the database matches the
clean row counts (1,674 / 282), cross-checks every fetched anchor against the
canonical report's per-field ground truth (IDs were renumbered by the ETL
rebuild, so row identity is proven per incident, not assumed), and recovers
pre-rebuild IDs by ground-truth match.

## 2. Historical earn-it audits (June 2026, before the canonical run)

These audits justified enabling each gate. They were run from `/tmp` and the
scripts were **not preserved**; the numbers below come from working notes and,
where pre-clean-DB data was involved, are not reproducible against the current
database. This is the gap §1 closes with a committed script.

- **Officer relevance judge** (2026-06-12, pre-clean-DB officer data): a $0
  reasoning-over-content analysis found roughly 18% of officer completions
  rested on a wrong article, motivating the judge. The Tier-A earn-it audit
  reviewed the judge's vetoes offline; after the judge-anchor fix (PR #58,
  which added `civilian_outcome` to the anchor and reworded the prompt), the
  re-run vetoed 7% of completions, every veto judged a genuine wrong-article
  match and previously flagged false vetoes recovered as KEEPs.
- **Civilian relevance gate** (2026-06-13, offline A/B over saved completions
  with anchors re-fetched from the database): of 123 records reviewed, the
  dataset-aware judge vetoed 17 — all 17 genuine wrong-article matches
  (name collisions, victim-list pages that never name the victim, wrong
  city/date/role), zero false vetoes.

## 3. Provenance of the canonical reports these vetoes come from

- The canonical **civilians** report (`20260613_231717`) is an assembly: 98
  incidents carried from the same-day 18:44:42 run plus a 2-incident top-up
  (421, 422) run at 22:54 after the ETL fix, with 182/183 dropped. All 7
  civilian vetoes are in the carried 98; their `_escalate.json` files (mtimes
  17:23–18:41) are the artifacts the judge acted on, and the veto set is
  identical in both reports. The **officers** report is a single run ending
  23:17:17.
- Sixteen report IDs predate the idempotent-ETL rebuild (15 officer, 1
  civilian). Among the vetoes only officer 357 is affected; §1 note ³ records
  its recovery.
- The two officer **bracket reports**
  (`holdout_officers_shot_20260611_160440.json`, 38/100 completed, 54
  `insufficient_sources`; `..._233624.json`, 95/100, 0) bracket the
  dataset-aware-prompt merge (PR #52, 2026-06-11 20:34) on identical incident
  IDs: 0 regressions, 57 recoveries, all 54 `insufficient_sources` recovered
  (Wilson 95% CIs 29–48% vs 89–98%). They are committed as **attribution
  evidence only** — the absolute rates are pre-dedup measurements (each report
  has 100 slots but 95 unique IDs; incidents 82 and 100–103 occupy two slots
  each, an artifact of the then-doubled database), so publications cite the
  clean-database numbers (92%, insufficient-source escalations 1 of 100), not
  38→95. Committing report JSONs is an exception to the repo's summary-only
  convention for `output/`, made so the paper's causal claim (the prompt, not
  an agent, drove the officer recovery) rests on tracked artifacts.
