"""CLI entrypoint for the LangGraph enrichment pipeline.

Invokes the enrichment graph on a single incident, wiring up LLM client,
checkpointer, and settings from environment variables.

Usage:
    python -m src.run <incident_id> <dataset_type>
"""

import argparse
import logging
import os
import sqlite3

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.graph import build_graph
from src.agents.state import DatasetType, EnrichmentState
from src.config import Settings

load_dotenv(override=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run(incident_id: str, dataset_type: str) -> dict:
    """Build and invoke the enrichment graph for a single incident.

    Sets up the LLM client, SQLite checkpointer, and pipeline settings,
    then runs the full enrichment graph to completion.

    Args:
        incident_id: TJI record identifier to enrich.
        dataset_type: One of 'civilians_shot' or 'officers_shot'.

    Returns:
        Final graph state as a dict, including 'outcome_summary'.

    Raises:
        KeyError: If OPENAI_API_KEY is not set in the environment.
    """
    state = EnrichmentState(incident_id=incident_id, dataset_type=dataset_type)
    llm_client = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    conn = sqlite3.connect("./checkpoints.db", check_same_thread=False)

    checkpointer = SqliteSaver(conn=conn)
    graph = build_graph(checkpointer=checkpointer)
    config = RunnableConfig(
        {
            "configurable": {
                "settings": Settings(),
                "llm_client": llm_client,
                "thread_id": f"{dataset_type}_{incident_id}",
            }
        }
    )
    result = graph.invoke(state, config)
    return result


def main() -> None:
    """Parse CLI arguments and invoke the enrichment pipeline.

    Validates dataset_type against the DatasetType enum, then calls
    run() with the parsed arguments. Logs start, completion, and
    outcome summary.

    Raises:
        ValueError: If dataset_type is not a valid DatasetType.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("incident_id", type=str, help="Incident ID")
    parser.add_argument(
        "dataset_type", type=str, help="Dataset type (officer or civilian)"
    )
    args = parser.parse_args()

    dataset_type = args.dataset_type
    incident_id = args.incident_id

    try:
        DatasetType(dataset_type)
    except ValueError:
        msg = f"Invalid DatasetType: '{dataset_type}'"
        logger.error(msg)
        raise ValueError(msg) from None

    logger.info(
        f"Data enrichment started for {dataset_type} dataset with incident id #{incident_id}..."
    )
    result = run(incident_id, dataset_type)
    logger.info("Data enrichment completed.")
    logger.info(result["outcome_summary"])


if __name__ == "__main__":
    main()
