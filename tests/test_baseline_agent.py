"""Unit tests for the autonomous-agent baseline (src/baselines/autonomous_agent).

All external calls (LLM, Tavily, web, DB) are mocked — no network, no live model.
The Anthropic client is a ``MagicMock`` whose ``messages.create`` returns canned
responses, matching the repo's dependency-injection test style.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.agents.state import Article, DatasetType
from src.baselines.autonomous_agent.agent import (
    _cost_usd,
    load_instructions,
    run_baseline_agent,
)
from src.baselines.autonomous_agent.result import (
    BaselineUsage,
    FabricationTag,
)
from src.baselines.autonomous_agent.tools import (
    BaselineTools,
    _html_to_text,
)
from src.config import Settings

# --- Fake Anthropic response builders ---


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool(block_id: str, name: str, tool_input: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def _usage(i: int = 0, o: int = 0, cc: int = 0, cr: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=i,
        output_tokens=o,
        cache_creation_input_tokens=cc,
        cache_read_input_tokens=cr,
    )


def _response(content: list, stop_reason: str, usage=None) -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage or _usage())


def _client(responses: list) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = responses
    return client


_ANCHOR = {
    "officer_name": "Reginald Q. Farnsworth",
    "civilian_name": "Bartholomew T. Quincy",
    "incident_date": "2019-07-14",
    "location": "Marfa",
    "severity": "fatal",
}


def _tools(*, search_fn=None, db_anchor=None) -> BaselineTools:
    return BaselineTools(
        "99901",
        DatasetType.CIVILIANS_SHOT,
        settings=Settings(),
        search_fn=search_fn or (lambda q, n, s: []),
        fetch_fn=lambda url: "page text",
        db_fn=lambda iid, ds: dict(db_anchor or _ANCHOR),
    )


# --- Instruction file ---


def test_instruction_file_loads_and_states_the_standard_of_care() -> None:
    text = load_instructions()
    assert "Cite a source for every field" in text
    assert "submit_record" in text
    assert "may decline" in text.lower() or "You may decline" in text


# --- Cost model ---


def test_cost_uses_sonnet_46_prices() -> None:
    usage = BaselineUsage(input_tokens=300, output_tokens=130)
    # (300 * $3 + 130 * $15) / 1e6
    assert _cost_usd(usage) == (300 * 3.0 + 130 * 15.0) / 1_000_000


def test_cost_includes_cache_tiers() -> None:
    usage = BaselineUsage(cache_creation_input_tokens=1000, cache_read_input_tokens=1000)
    assert _cost_usd(usage) == (1000 * 3.75 + 1000 * 0.30) / 1_000_000


# --- Agent loop ---


def test_completes_with_sourced_field_tags_wrong_article() -> None:
    search_fn = MagicMock(
        return_value=[
            Article(url="https://news/a", title="Shooting", snippet="x", content="...")
        ]
    )
    responses = [
        _response(
            [_tool("t1", "search_news", {"query": "Marfa shooting"})],
            "tool_use",
            _usage(100, 50),
        ),
        _response(
            [
                _tool(
                    "t2",
                    "submit_record",
                    {
                        "completed": True,
                        "fields": [
                            {
                                "field_name": "weapon",
                                "value": "handgun",
                                "source_urls": ["https://news/a"],
                                "source_quote": "armed with a handgun",
                                "confidence": "high",
                            }
                        ],
                    },
                )
            ],
            "tool_use",
            _usage(200, 80),
        ),
    ]
    result = run_baseline_agent(
        "99901",
        DatasetType.CIVILIANS_SHOT,
        "A",
        client=_client(responses),
        tools=_tools(search_fn=search_fn),
    )

    assert result.completed is True
    assert result.declined is False
    assert len(result.extracted_fields) == 1
    assert result.extracted_fields[0].field_name == "weapon"
    assert result.committed_fabrications == 1
    assert result.n_wrong_article == 1
    assert result.n_parametric == 0
    # usage tallied across both turns
    assert result.usage.input_tokens == 300
    assert result.usage.output_tokens == 130
    assert result.usage.search_calls == 1
    assert result.usage.cost_usd == (300 * 3.0 + 130 * 15.0) / 1_000_000
    search_fn.assert_called_once()


def test_unsourced_value_tags_parametric() -> None:
    responses = [
        _response(
            [
                _tool(
                    "t1",
                    "submit_record",
                    {
                        "completed": True,
                        "fields": [
                            {"field_name": "civilian_age", "value": "34", "source_urls": []}
                        ],
                    },
                )
            ],
            "tool_use",
        )
    ]
    result = run_baseline_agent(
        "99901", DatasetType.CIVILIANS_SHOT, "A", client=_client(responses), tools=_tools()
    )
    assert result.completed is True
    assert result.n_parametric == 1
    assert result.field_audit[0].tag == FabricationTag.PARAMETRIC


def test_planted_name_is_flagged() -> None:
    responses = [
        _response(
            [
                _tool(
                    "t1",
                    "submit_record",
                    {
                        "completed": True,
                        "fields": [
                            {
                                "field_name": "civilian_name",
                                "value": "Bartholomew T. Quincy",  # the injected fake name
                                "source_urls": ["https://news/x"],
                            }
                        ],
                    },
                )
            ],
            "tool_use",
        )
    ]
    result = run_baseline_agent(
        "99901", DatasetType.CIVILIANS_SHOT, "A", client=_client(responses), tools=_tools()
    )
    assert result.n_planted_name == 1
    assert result.field_audit[0].matches_planted_name is True


def test_explicit_decline_marks_declined() -> None:
    responses = [
        _response(
            [
                _tool(
                    "t1",
                    "submit_record",
                    {"completed": False, "decline_reason": "no article about this incident", "fields": []},
                )
            ],
            "tool_use",
        )
    ]
    result = run_baseline_agent(
        "99901", DatasetType.CIVILIANS_SHOT, "A", client=_client(responses), tools=_tools()
    )
    assert result.completed is False
    assert result.declined is True
    assert result.decline_reason == "no article about this incident"
    assert result.committed_fabrications == 0


def test_ending_turn_without_submit_declines() -> None:
    responses = [_response([_text("I could not find anything.")], "end_turn")]
    result = run_baseline_agent(
        "99901", DatasetType.CIVILIANS_SHOT, "A", client=_client(responses), tools=_tools()
    )
    assert result.declined is True
    assert "submit_record" in result.decline_reason


def test_writes_transcript(tmp_path) -> None:
    responses = [
        _response(
            [_tool("t1", "submit_record", {"completed": False, "fields": []})], "tool_use"
        )
    ]
    result = run_baseline_agent(
        "99901",
        DatasetType.CIVILIANS_SHOT,
        "A",
        client=_client(responses),
        tools=_tools(),
        transcript_dir=tmp_path,
    )
    assert result.transcript_path is not None
    assert result.arm == "informed"
    assert (tmp_path / "civilians_shot_99901_informed_run0.json").exists()


def test_naive_arm_uses_naive_prompt_and_is_stamped(tmp_path) -> None:
    # The naive prompt must NOT carry the project's earned failure-mode rules.
    naive = load_instructions("naive")
    informed = load_instructions("informed")
    assert "Never infer race" not in naive  # race-verifier rule is informed-only
    assert "famous case" not in naive  # relevance-judge taxonomy is informed-only
    assert "Never infer race" in informed

    responses = [
        _response(
            [_tool("t1", "submit_record", {"completed": False, "fields": []})], "tool_use"
        )
    ]
    result = run_baseline_agent(
        "99901",
        DatasetType.CIVILIANS_SHOT,
        "A",
        client=_client(responses),
        tools=_tools(),
        arm="naive",
        transcript_dir=tmp_path,
    )
    assert result.arm == "naive"
    assert (tmp_path / "civilians_shot_99901_naive_run0.json").exists()


def test_max_iterations_backstop_sets_error() -> None:
    # Always return a benign non-submitting tool call -> loop hits the cap.
    looping = _response([_tool("t1", "read_incident_record", {})], "tool_use")
    client = MagicMock()
    client.messages.create.return_value = looping
    result = run_baseline_agent(
        "99901",
        DatasetType.CIVILIANS_SHOT,
        "A",
        client=client,
        tools=_tools(),
        max_iterations=3,
    )
    assert result.error is not None
    assert "max_iterations" in result.error
    assert client.messages.create.call_count == 3


# --- Tools ---


def test_search_tool_counts_calls_and_credits() -> None:
    tools = _tools(
        search_fn=lambda q, n, s: [Article(url="u", title="t", snippet="s", content="c")]
    )
    out = tools.run("search_news", {"query": "x", "max_results": 5})
    assert '"results"' in out
    assert tools.search_calls == 1
    # default search_depth is "advanced" -> 2 credits
    assert tools.tavily_credits == 2


def test_tavily_cost_tracked_from_credits() -> None:
    search_fn = MagicMock(return_value=[Article(url="u", title="t", snippet="s", content="c")])
    responses = [
        _response([_tool("t1", "search_news", {"query": "x"})], "tool_use"),
        _response(
            [_tool("t2", "submit_record", {"completed": False, "fields": []})], "tool_use"
        ),
    ]
    result = run_baseline_agent(
        "99901",
        DatasetType.CIVILIANS_SHOT,
        "A",
        client=_client(responses),
        tools=_tools(search_fn=search_fn),
    )
    # one advanced search = 2 credits; PAYG $0.008/credit
    assert result.usage.tavily_credits == 2
    assert result.usage.tavily_cost_usd == round(2 * 0.008, 4)


def test_read_incident_record_returns_anchor() -> None:
    tools = _tools()
    out = tools.run("read_incident_record", {})
    assert "Bartholomew T. Quincy" in out


def test_tool_run_is_fail_open_on_error() -> None:
    def boom(q, n, s):
        raise RuntimeError("tavily down")

    tools = _tools(search_fn=boom)
    out = tools.run("search_news", {"query": "x"})
    assert '"error"' in out
    assert "tavily down" in out


def test_html_to_text_strips_scripts() -> None:
    html = "<html><body><p>Hello</p><script>var x=1;</script><p>World</p></body></html>"
    text = _html_to_text(html)
    assert "Hello" in text and "World" in text
    assert "var x" not in text


# --- Runner (offline: patches fetch_incident with the suite's fake_fetch) ---


def test_run_suite_patches_db_and_aggregates(tmp_path) -> None:
    from src.baselines.autonomous_agent import runner

    # Mock client always declines immediately -> no live search/fetch needed.
    decline = _response(
        [_tool("t1", "submit_record", {"completed": False, "fields": []})], "tool_use"
    )
    client = MagicMock()
    client.messages.create.return_value = decline

    aggregate = runner.run_suite(
        client=client,
        n_runs=1,
        limit=2,
        transcript_dir=tmp_path,
        on_result=lambda i, s, r: None,
    )

    assert aggregate["n_incidents"] == 2
    assert aggregate["n_runs"] == 1
    assert len(aggregate["results"]) == 2
    assert aggregate["per_run"][0]["declined"] == 2
    assert aggregate["per_run"][0]["completed"] == 0
    # The anchor was read through the patched fetch_incident (fake_fetch), so the
    # transcript carries the fabricated incident's data, not a DB error.
    assert "summary" not in aggregate  # sanity: key name guard
    md = runner.write_summary_md(aggregate)
    assert "Autonomous-agent baseline" in md
    assert "Baseline run 0" in md
