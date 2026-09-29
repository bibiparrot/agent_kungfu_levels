"""Derive feasible hypotheses, falsify a revenue-only objective, retain supported improvement."""
import json
import math

from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS

from ...database import SakilaDB
from ...llm import OllamaLLM
from ..common import load_level_config, resolved_settings
from ..level9.causal_agent import run as causal_run


def search_interventions(*, demand, copies, price, unit_cost, transfer_cost, purchase_cost):
    if (len(demand) != 2 or len(copies) != 2 or sum(copies) <= 0
            or any(type(v) is not int or v < 0 for v in [*demand, *copies])
            or any(not math.isfinite(v) or v < 0 for v in [price, unit_cost, transfer_cost, purchase_cost])):
        raise ValueError("Require two nonnegative integer demand/capacity values and finite costs")

    def evaluate(kind, allocation, expense):
        served = sum(min(d, c) for d, c in zip(demand, allocation))
        revenue = served * price
        contribution = revenue - served * unit_cost - expense
        return {"kind": kind, "copies": allocation, "served": served, "revenue": revenue,
                "contribution": contribution, "score": contribution / sum(allocation)}

    baseline = evaluate("baseline", list(copies), 0)
    candidates = []
    # Conservation of copies derives the finite feasible intervention space.
    for transfer in range(-copies[0], copies[1] + 1):
        if transfer:
            candidates.append(evaluate("rebalance", [copies[0] + transfer, copies[1] - transfer],
                                       abs(transfer) * transfer_cost))
    extra = [max(0, d - c) for d, c in zip(demand, copies)]
    if sum(extra):
        candidates.append(evaluate("purchase", [c + e for c, e in zip(copies, extra)],
                                   sum(extra) * purchase_cost))
    for candidate in candidates:
        candidate["rejected"] = candidate["score"] <= baseline["score"]
        candidate["reason"] = ("No contribution-per-inventory-period improvement" if candidate["rejected"]
                               else "Improves the revised objective under stated assumptions")
    supported = [candidate for candidate in candidates if not candidate["rejected"]]
    best = max(supported, key=lambda candidate: candidate["score"]) if supported else None
    return {"baseline": baseline, "candidates": candidates, "selected": best,
            "revenue_winner": max([baseline, *candidates], key=lambda candidate: candidate["revenue"]),
            "stop_reason": "finite_hypothesis_space_evaluated" if best else "no_supported_improvement",
            "scope": "synthetic structural simulator, deterministic demand, one rental per copy per period"}


def run(question, *, settings=None, offline=False):
    resolved = resolved_settings(settings)
    config = load_level_config(10)["experiment"]
    result = causal_run(question, settings=resolved, offline=True)
    price = float(SakilaDB(resolved).query("SELECT AVG(amount) AS amount FROM payment").item())
    report = search_interventions(price=price, **config)
    balanced = search_interventions(price=price, **{**config, "demand": config["copies"]})
    principles = {
        "objective": "contribution per inventory period, rather than revenue alone",
        "invariants": ["one copy serves at most one rental per period", "served rentals cannot exceed demand",
                       "contribution = revenue - operating cost - intervention cost"],
        "constraints": ["two simulated stores", "nonnegative integer copies", "rebalancing conserves total copies",
                        "explicit intervention costs"],
        "mechanisms": ["reallocate idle copies", "purchase additional copies", "retain baseline"],
        "experiments": ["evaluate all feasible transfers", "falsify revenue-only purchasing",
                        "balanced-demand negative control"],
    }
    explanation = (f"合成实验 / Synthetic experiment: evaluated {len(report['candidates'])} hypotheses; "
                   f"selected={report['selected']}; negative control={balanced['stop_reason']}. "
                   "This is a finite hypothesis search under declared assumptions, not validated real-world innovation.")
    if not offline:
        explanation += "\n" + OllamaLLM(resolved).complete(json.dumps({"principles": principles, "experiment": report}),
            system="Explain which assumption was falsified and which feasible intervention survived. Label all results synthetic.")
    directory = resolved.output_dir / "level10_first_principles"
    directory.mkdir(parents=True, exist_ok=True)
    evidence = directory / "hypothesis_tests.json"
    evidence.write_text(json.dumps({"experiment": report, "negative_control": balanced}, indent=2), encoding="utf-8")
    wiki = directory / "Tested_Principles.md"
    wiki.write_text(f"# First-principles experiments / 第一性原理实验\n\n{explanation}\n\n{json.dumps(principles, indent=2)}", encoding="utf-8")
    ontology = Graph()
    ns = Namespace("https://example.org/sakila/innovation#")
    for term in ("InventoryPeriod", "FeasibleTransfer", "InterventionCost", "Contribution"):
        ontology.add((ns[term], RDF.type, OWL.Class))
        ontology.add((ns[term], RDFS.comment, Literal("Candidate concept derived from synthetic objective revision")))
    owl = directory / "candidate-principles.owl"
    ontology.serialize(str(owl), format="xml")
    result.level, result.driver = 10, "First-Principles Driven Agent"
    result.answer = explanation
    result.metadata.update(first_principles=principles, innovation_experiment=report,
                           negative_control=balanced, candidate_source="constraint-enumeration")
    result.artifacts.extend(map(str, [evidence, wiki, owl]))
    return result
