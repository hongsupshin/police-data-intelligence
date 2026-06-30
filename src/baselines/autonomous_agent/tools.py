"""Tools for the autonomous-agent baseline.

Three working tools the agent drives itself — a free-text Tavily **search**, an
open-web **fetch**, and a **db-read** for the incident anchor — plus the terminal
``submit_record`` schema. The real wrappers are reused at the granularity the repo
intends: ``search_node.py``'s docstring anticipates a "free-text ReAct tool [that]
share[s] the window/fetch/convert tail", so the search tool reuses
``_convert_tavily_result`` + the ``TavilyClient`` call rather than routing a
free-text query through ``execute_search`` (which only builds anchored queries).

All external calls are injected (``search_fn`` / ``fetch_fn`` / ``db_fn``) so the
agent is testable with ``MagicMock`` and never touches the network in unit tests.
"""

import json
import os
from collections.abc import Callable
from html.parser import HTMLParser

import httpx
from tavily import TavilyClient

from src.agents import load_node
from src.agents.state import Article, DatasetType
from src.config import Settings
from src.retrieval.search_node import _convert_tavily_result

# Reference (non-news) sites excluded from search, matching the shipped retrieval
# path (src/retrieval/search_node.py).
_EXCLUDE_DOMAINS = ["wikipedia.org", "fatalencounters.org"]

# Anthropic tool-use schemas. submit_record is the terminal tool; the agent loop
# captures it and stops rather than executing it here.
TOOL_SCHEMAS = [
    {
        "name": "search_news",
        "description": (
            "Search news and the web for articles about a police-shooting "
            "incident. Returns each result's url, title, published date, and "
            "article text. Choose your own query; run as many searches as you need."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search query."},
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (1-20).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_incident_record",
        "description": (
            "Re-read the database anchor for this incident (id, date, "
            "city/county, and any names on file)."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "fetch_webpage",
        "description": (
            "Fetch the full text of a URL (e.g. an article you found and want to "
            "read in full). Returns extracted page text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."}
            },
            "required": ["url"],
        },
    },
    {
        "name": "submit_record",
        "description": (
            "Record your final answer for this incident. Call exactly once. Set "
            "completed=true with a non-empty fields list to enrich the record, or "
            "completed=false with a decline_reason and an empty fields list to "
            "decline it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "completed": {"type": "boolean"},
                "decline_reason": {
                    "type": "string",
                    "description": "One line; required when completed=false.",
                },
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_name": {"type": "string"},
                            "value": {"type": "string"},
                            "source_urls": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "URL(s) whose text supports this value.",
                            },
                            "source_quote": {
                                "type": "string",
                                "description": "Supporting sentence from the source.",
                            },
                            "reasoning": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                        "required": ["field_name", "value"],
                    },
                },
            },
            "required": ["completed"],
        },
    },
]


class _TextExtractor(HTMLParser):
    """Collect visible text from HTML, skipping script/style (stdlib only)."""

    def __init__(self) -> None:
        super().__init__()
        self._skip = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self._parts)


def _html_to_text(html: str) -> str:
    """Convert an HTML document to plain visible text (no new dependency)."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:  # pragma: no cover - malformed HTML is best-effort
        pass
    return parser.text()


def default_search(query: str, max_results: int, settings: Settings) -> list[Article]:
    """Run a free-text Tavily search and map results to ``Article`` objects.

    Reuses ``_convert_tavily_result`` and the ``TavilyClient`` construction from
    the shipped retrieval path. No temporal window is applied — the agent governs
    its own queries and relevance, so the baseline is, if anything, more capable
    than the date-bounded shipped search.

    Args:
        query: The agent's free-text query.
        max_results: Number of results to request (clamped to 1-20).
        settings: Pipeline settings (provides ``search_depth``).

    Returns:
        List of ``Article`` objects (possibly empty).
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(
        query,
        max_results=max(1, min(20, max_results)),
        search_depth=settings.search_depth,
        exclude_domains=_EXCLUDE_DOMAINS,
    )["results"]
    return [_convert_tavily_result(r) for r in results]


def default_fetch(url: str) -> str:
    """Fetch a URL and return extracted page text (httpx + stdlib HTML strip)."""
    resp = httpx.get(
        url,
        timeout=20.0,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; TJI-research/1.0)"},
    )
    resp.raise_for_status()
    return _html_to_text(resp.text)


def default_db_read(incident_id: str, dataset_type: DatasetType) -> dict:
    """Read the incident anchor via the shipped ``fetch_incident``.

    Calls through the ``load_node`` module attribute (not a bound import) so the
    adversarial suite's ``patch("src.agents.load_node.fetch_incident", ...)`` is
    honored and the real PostgreSQL database is never touched.
    """
    return load_node.fetch_incident(None, int(incident_id), dataset_type)


class BaselineTools:
    """Executes the agent's non-terminal tools and tallies tool usage.

    Args:
        incident_id: The incident the agent is enriching.
        dataset_type: civilians_shot or officers_shot.
        settings: Pipeline settings (search depth, etc.).
        search_fn: ``(query, max_results, settings) -> list[Article]``.
        fetch_fn: ``(url) -> str``.
        db_fn: ``(incident_id, dataset_type) -> dict`` anchor.
        max_result_chars: Cap on per-result article text returned to the model.
        max_fetch_chars: Cap on fetched page text returned to the model.
    """

    def __init__(
        self,
        incident_id: str,
        dataset_type: DatasetType,
        *,
        settings: Settings,
        search_fn: Callable[[str, int, Settings], list[Article]] | None = None,
        fetch_fn: Callable[[str], str] | None = None,
        db_fn: Callable[[str, DatasetType], dict] | None = None,
        max_result_chars: int = 2500,
        max_fetch_chars: int = 6000,
    ) -> None:
        self.incident_id = incident_id
        self.dataset_type = dataset_type
        self.settings = settings
        self._search_fn = search_fn or default_search
        self._fetch_fn = fetch_fn or default_fetch
        self._db_fn = db_fn or default_db_read
        self.max_result_chars = max_result_chars
        self.max_fetch_chars = max_fetch_chars

        self.search_calls = 0
        self.fetch_calls = 0
        self.tavily_credits = 0
        self._anchor: dict | None = None

    def get_anchor(self) -> dict:
        """Return the incident anchor, reading it once and caching it."""
        if self._anchor is None:
            self._anchor = self._db_fn(self.incident_id, self.dataset_type)
        return self._anchor

    def run(self, name: str, tool_input: dict) -> str:
        """Execute one non-terminal tool and return a JSON string for the model.

        Args:
            name: Tool name (search_news | read_incident_record | fetch_webpage).
            tool_input: The tool's input dict from the model.

        Returns:
            A JSON-encoded result string (errors are returned as JSON, not raised,
            so the agent can adapt — fail-open).
        """
        try:
            if name == "search_news":
                return self._search(tool_input)
            if name == "read_incident_record":
                return json.dumps(self._anchor_for_model(), default=str)
            if name == "fetch_webpage":
                return self._fetch(tool_input)
            return json.dumps({"error": f"unknown tool: {name}"})
        except Exception as e:  # fail-open: surface the error to the agent
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def _search(self, tool_input: dict) -> str:
        query = tool_input.get("query", "")
        max_results = int(tool_input.get("max_results", self.settings.max_search_results))
        articles = self._search_fn(query, max_results, self.settings)
        self.search_calls += 1
        self.tavily_credits += 2 if self.settings.search_depth == "advanced" else 1
        results = [
            {
                "url": a.url,
                "title": a.title,
                "published_date": str(a.published_date) if a.published_date else None,
                "content": (a.content or a.snippet or "")[: self.max_result_chars],
            }
            for a in articles
        ]
        return json.dumps({"query": query, "results": results}, default=str)

    def _fetch(self, tool_input: dict) -> str:
        url = tool_input.get("url", "")
        text = self._fetch_fn(url)
        self.fetch_calls += 1
        return json.dumps({"url": url, "text": text[: self.max_fetch_chars]})

    def _anchor_for_model(self) -> dict:
        anchor = self.get_anchor()
        return {
            "incident_id": self.incident_id,
            "dataset_type": str(self.dataset_type),
            **{k: (str(v) if v is not None else None) for k, v in anchor.items()},
        }
