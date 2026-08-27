"""Compatibility import for the AI-facing financial fact store.

The implementation belongs to agent.financial_facts so the extraction engine
remains independent of model/agent concerns.
"""

from agent.financial_facts import build_fact_store, total_debt_candidates

__all__ = ["build_fact_store", "total_debt_candidates"]
