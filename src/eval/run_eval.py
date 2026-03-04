"""CLI entrypoint for holdout evaluation.

Usage:
    python -m src.eval.run_eval civilians_shot --limit 20 --min-fields 2
    python -m src.eval.run_eval civilians_shot --limit 40 --stratified
    python -m src.eval.run_eval officers_shot --limit 10 --no-stratified
"""

import argparse
import logging

from src.agents.state import DatasetType
from src.eval.holdout import (
    evaluate_holdout,
    evaluate_holdout_stratified,
    print_report,
    save_report,
)

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments and run holdout evaluation."""
    parser = argparse.ArgumentParser(
        description="Run holdout evaluation for the enrichment pipeline",
    )
    parser.add_argument(
        "dataset_type",
        type=str,
        help="Dataset to evaluate (civilians_shot or officers_shot)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of incidents to evaluate (default: 20)",
    )
    parser.add_argument(
        "--min-fields",
        type=int,
        default=2,
        help="Minimum non-NULL eval fields per incident (default: 2)",
    )
    parser.add_argument(
        "--stratified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use year-stratified sampling (default: True)",
    )
    parser.add_argument(
        "--exclude-dev-set",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude dev-set incidents from holdout (default: True)",
    )
    parser.add_argument(
        "--incident-ids",
        type=str,
        default=None,
        help="Comma-separated incident IDs to evaluate (skips DB selection)",
    )
    args = parser.parse_args()

    try:
        dataset_type = DatasetType(args.dataset_type)
    except ValueError:
        valid = [dt.value for dt in DatasetType]
        msg = f"Invalid dataset_type: '{args.dataset_type}'. Must be one of {valid}"
        logger.error(msg)
        raise SystemExit(1) from None

    logger.info(
        "Starting holdout evaluation: dataset=%s, limit=%d, min_fields=%d, "
        "stratified=%s, exclude_dev_set=%s",
        dataset_type.value,
        args.limit,
        args.min_fields,
        args.stratified,
        args.exclude_dev_set,
    )

    if args.incident_ids:
        ids = [int(x.strip()) for x in args.incident_ids.split(",")]
        report = evaluate_holdout(dataset_type, incident_ids=ids)
    elif args.stratified:
        report = evaluate_holdout_stratified(
            dataset_type, args.limit, args.min_fields, args.exclude_dev_set
        )
    else:
        report = evaluate_holdout(dataset_type, args.limit, args.min_fields)
    print_report(report)

    filepath = save_report(report)
    logger.info("Report saved to %s", filepath)


if __name__ == "__main__":
    main()
