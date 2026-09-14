"""Observation-only query accounting for R2.5; no timers or model imports.

A service call is one submitted cube to one proxy endpoint, regardless of
internal reuse (e.g. shared VCA extraction). Local q0 and surrogate gradients
are reported separately. This ledger does not change budgets or cache policy.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Any


PHASES = ("reference", "search", "selection")


class QueryLedger:
    def __init__(self) -> None:
        self.requests = Counter()
        self.evaluations = Counter()
        self.cache_hits = Counter()
        self.budget_returns = Counter()
        self.calls = {phase: Counter() for phase in PHASES}
        self.completed = {phase: Counter() for phase in PHASES}
        self.failed = {phase: Counter() for phase in PHASES}
        self.local = Counter()

    def call(self, phase: str, name: str, query: Callable, cube: Any) -> Any:
        self.calls[phase][name] += 1
        try:
            result = query(cube)
        except BaseException:
            self.failed[phase][name] += 1
            raise
        self.completed[phase][name] += 1
        return result

    def snapshot(self) -> dict:
        phases = {}
        for phase in PHASES:
            phases[phase] = {
                "candidate_requests": self.requests[phase],
                "unique_candidate_evaluations": self.evaluations[phase],
                "candidate_cache_hits": self.cache_hits[phase],
                "budget_fallback_returns": self.budget_returns[phase],
                "service_calls": sum(self.calls[phase].values()),
                "completed_service_calls": sum(self.completed[phase].values()),
                "failed_service_calls": sum(self.failed[phase].values()),
                "per_service_calls": dict(self.calls[phase]),
            }
        return {
            "schema_version": "r25.v1",
            "status": "instrumented_counts",
            "scope": "one_attack_run_excluding_held_out_victim_evaluation",
            "service_unit": "one_cube_to_one_proxy_endpoint",
            "candidate_requests": sum(self.requests.values()),
            "unique_candidate_evaluations": sum(self.evaluations.values()),
            "candidate_cache_hits": sum(self.cache_hits.values()),
            "budget_fallback_returns": sum(self.budget_returns.values()),
            "reference_service_calls": phases["reference"]["service_calls"],
            "search_service_calls": phases["search"]["service_calls"],
            "selection_service_calls": phases["selection"]["service_calls"],
            "total_service_calls": sum(p["service_calls"] for p in phases.values()),
            "phases": phases,
            "local_operations": dict(self.local),
            "runtime": {
                "status": "not_measured_deferred_shared_hardware",
                "measured_seconds": None,
                "estimated_seconds": None,
                "estimate_status": "not_available_in_query_ledger",
                "publication_eligible": False,
            },
        }


def aggregate_query_ledgers(runs: list[dict]) -> dict:
    """Keep restart costs, including each run's clean initialization.

    Unique counts are per-run: summing does NOT imply global deduplication
    across restarts. No selected-restart-only accounting is allowed.
    """
    keys = ("candidate_requests", "unique_candidate_evaluations",
            "candidate_cache_hits", "budget_fallback_returns",
            "reference_service_calls", "search_service_calls",
            "selection_service_calls", "total_service_calls")
    return {
        "schema_version": "r25.v1",
        "status": "instrumented_counts",
        "scope": "all_restarts_excluding_held_out_victim_evaluation",
        "service_unit": "one_cube_to_one_proxy_endpoint",
        "unique_count_scope": "sum_of_per_run_uniques_not_global_uniques",
        **{key: sum(run[key] for run in runs) for key in keys},
        "run_count": len(runs),
        "runs": runs,
        "runtime": {
            "status": "not_measured_deferred_shared_hardware",
            "measured_seconds": None,
            "estimated_seconds": None,
            "estimate_status": "not_available_in_query_ledger",
            "publication_eligible": False,
        },
    }
