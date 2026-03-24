# Holdout Eval Samples (v1)

Initial 10-incident holdout set used for the first evaluation run on 2026-03-03.

## Selection Criteria

- Dataset: `civilians_shot`
- `--limit 10 --min-fields 2`
- Ranked by number of non-NULL evaluable columns (age, race, weapon,
  location_detail, time_of_day, outcome)
- Required `date_incident IS NOT NULL` and `incident_city IS NOT NULL`

## Samples

| #   | incident_id | Pipeline Outcome | Escalation Reason | Ground Truth Fields                                                               |
| --- | ----------- | ---------------- | ----------------- | --------------------------------------------------------------------------------- |
| 1   | 3710        | escalate         | max_retries       | age=55, race=WHITE, weapon=SHOTGUN, address=5021 GLENVIEW DR., time=19:37         |
| 2   | 5388        | complete         | conflict          | age=39, race=OTHER, weapon=HANDGUN, address=20402 SH 195, time=13:18              |
| 3   | 3630        | escalate         | max_retries       | age=47, race=WHITE, weapon=KNIFE, address=1070 RIO RANCHERO, time=20:49           |
| 4   | 3669        | complete         | --                | age=35, race=WHITE, weapon=SHOTGUN, address=100 BLOCK OF BISHOP ROAD, time=11:35  |
| 5   | 833         | complete         | conflict          | age=31, race=BLACK, weapon=VEHICLE, address=1606 E REGIS, time=14:50              |
| 6   | 792         | complete         | --                | age=26, race=BLACK, weapon=KNIFE, address=20035 NORTH FREEWAY, time=16:52         |
| 7   | 5168        | complete         | conflict          | age=27, race=WHITE, weapon=KNIFE, address=900 N. WELLS STREET, time=16:00         |
| 8   | 697         | complete         | conflict          | age=41, race=BLACK, weapon=GUN, address=900 BLK ORANGE, time=21:21                |
| 9   | 3744        | complete         | --                | age=20, race=HISPANIC, weapon=VEHICLE, address=2137 AUTUMN SAGE DRIVE, time=15:00 |
| 10  | 330         | complete         | conflict          | age=38, race=HISPANIC, weapon=GLOCK 40, address=951 FALCON, time=13:20            |

## Aggregate Results

- **Completion rate**: 80% (8 completed, 2 escalated)
- **Escalation breakdown**: 2 max_retries

| Field           | Evaluable | Extracted | Exact Acc | Fuzzy Acc | Coverage |
| --------------- | --------- | --------- | --------- | --------- | -------- |
| civilian_age    | 10        | 5         | 100%      | 100%      | 50%      |
| civilian_race   | 10        | 1         | 100%      | 100%      | 10%      |
| weapon          | 10        | 8         | 75%       | 75%       | 80%      |
| location_detail | 10        | 5         | 0%        | 100%      | 50%      |
| time_of_day     | 10        | 4         | 100%      | 100%      | 40%      |
| outcome         | 10        | 7         | 100%      | 100%      | 70%      |

> **Note**: outcome was 0 evaluable in v1-v2 because `civilian_died` was NULL
> for all DB rows. Fixed by backfill migration
> (`data/backfill_civilian_died.py`, applied 2026-03-04).

## Bugs Found

1. **Incident 697** -- DV victims PDF contamination. A statewide compilation PDF
   passed validation (matched on perpetrator name + city), causing the LLM to
   extract unrelated victim data. Fix: source-type filtering in validate_node
   (reject `.pdf`, `.csv`, `fatalencounters.org` URLs).

2. **Incident 5388** -- Name synonym conflict in merge. `"Alva Joe Gwinn"` vs
   `"Master Sgt. Alva Joe Gwinn"` scored `fuzz.ratio=70` (below threshold 80)
   due to the rank prefix. Fix: `normalize_name()` strips honorifics/quotes
   before fuzzy comparison on name fields.

## Report Files

- `output/eval/holdout_civilians_shot_20260303_124041.json` (v1 — before
  location fix)
- `output/eval/holdout_civilians_shot_20260304_101804.json` (v2 — after location
  prompt + eval GT fix)
- `output/eval/holdout_civilians_shot_20260304_115748.json` (v3 — after
  civilian_died backfill)
