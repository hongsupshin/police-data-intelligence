"""Fair, capable autonomous-agent baseline for data enrichment.

A single Anthropic tool-use agent that, given one incident's anchor, decides its
own queries, its own stopping point, and what to write — with **no** separate
faithfulness layer, **no** forced escalation, and **no** earn-it gate. Those are
exactly the shipped guardrails this baseline omits; their absence is the variable
under test. See ``README.md`` for what it deliberately leaves out.
"""

from src.baselines.autonomous_agent.agent import run_baseline_agent
from src.baselines.autonomous_agent.result import (
    BaselineFieldAudit,
    BaselineResult,
    BaselineUsage,
    FabricationTag,
)

__all__ = [
    "run_baseline_agent",
    "BaselineResult",
    "BaselineFieldAudit",
    "BaselineUsage",
    "FabricationTag",
]
