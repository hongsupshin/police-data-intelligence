# Autonomous-agent baseline

A **fair, capable** single autonomous agent for data enrichment, run on the same
20 fabricated adversarial incidents as the shipped pipeline. It is the "obvious
move" the project argues against: point one agent at a record, give it strong
tools and careful instructions, and let it decide its own queries, its own
stopping point, and what to write — with **no separate faithfulness layer, no
forced escalation, and no earn-it gate**. Withholding exactly those guardrails
(and nothing else) is the entire experiment.

This is a research baseline for the SciPy paper. It is **not** part of the
production pipeline and is read-only with respect to the real DB and the shipped
pipeline.

## What it is

- A single Anthropic tool-use loop (`agent.py`) on **`claude-sonnet-4-6`** — the
  same model as shipped extraction (`ANTHROPIC_MODEL` default).
- Tools (`tools.py`): free-text Tavily **search** (reuses the shipped
  `_convert_tavily_result` + `TavilyClient`), an open-web **fetch** (httpx +
  stdlib HTML→text), a **db-read** for the incident anchor (reuses the shipped
  `fetch_incident`), and a terminal **`submit_record`**.
- A generous, good-faith **instruction file** (`instructions.md`): cite a source
  URL for every asserted field; assert only what a source explicitly supports;
  the agent may decline any field or the whole record; it self-governs queries and
  stopping.
- Writes the **same per-field schema** the pipeline writes (`FieldExtraction`), so
  "completion" is defined identically, plus a harness-only fabrication audit and
  token/cost accounting (`result.py`).

## What it deliberately omits

The shipped guardrails — and only these — are withheld; their absence is the
variable under test:

- the relevance judge (`src/synthesize/relevance_judge.py`),
- the race verifier (`src/synthesize/race_verifier.py`),
- the conflict annotator (`src/synthesize/conflict_annotator.py`),
- the coordinator / retry-ladder escalation (`src/agents/coordinate_node.py`),
- both evaluation gates (`src/eval/gate.py`).

## How "fabrication" is measured

The shipped adversarial detector flags a hallucination **only** when an extracted
`civilian_name`/`officer_name` equals the injected fake name. That is too narrow
to score what a self-governed agent emits, so this harness does its own
accounting. Every incident is fabricated, so any non-null subject-level value is
unsupported by reality; each is tagged:

- **parametric** — a value with no cited source (pure invention),
- **wrong_article** — a value backed by a real source URL, which (since the
  incident does not exist) is necessarily about a *different* incident — the
  relevance-judge failure mode,
- and separately flagged when it equals the injected fake name (for direct
  comparison with the shipped detector).

The headline number is **committed fabrications**: fields the agent finalized with
no escalation signal telling a reviewer which records to distrust.

## Run

```bash
# Plumbing pilot — 1 run over the first 2 incidents (validate before spending)
python -m src.baselines.autonomous_agent.runner --runs 1 --limit 2

# Full experiment — 3 runs over all 20 incidents (run-to-run variance)
python -m src.baselines.autonomous_agent.runner --runs 3
```

Requires `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`. Patches **only**
`src.agents.load_node.fetch_incident` (exactly as `scripts/run_adversarial.py`
does); every other step runs live. Output lands in `output/adversarial_baseline/`
(`results.json`, `summary.md`, and one transcript per incident per run under
`transcripts/`) — the shipped `output/adversarial/` is never touched.

## Test

```bash
pytest tests/test_baseline_agent.py -v
```

All tests mock the LLM, Tavily, web, and DB — no network, no model spend.

## Fairness notes

- The agent gets a **free-text** search (no temporal window) and an open-web
  fetch, so the baseline is, if anything, *more* capable than the date-bounded
  shipped search — a fabrication finding is therefore strong, and the agent is not
  hobbled.
- Search result text is capped per result to bound tokens; the agent can
  `fetch_webpage` for full article text.
- Adaptive thinking is left off, matching shipped extraction (which uses no
  extended thinking).
