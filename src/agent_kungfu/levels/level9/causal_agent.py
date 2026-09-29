"""Known-ground-truth causal laboratory, separate from observational Sakila facts."""
import json
import math
import random

import polars as pl
from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS

from ...database import SakilaDB
from ...llm import OllamaLLM
from ..common import load_level_config, resolved_settings
from ..legacy import level_09 as observational_context


def simulate_rentals(*, seed=42, effect=2.0, n=4000, baseline=4.0):
    if n < 100 or not all(math.isfinite(v) for v in (effect, baseline)):
        raise ValueError("Require n >= 100 and finite parameters")
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        demand = rng.randrange(2)
        treatment = int(rng.random() < (0.8 if demand else 0.2))
        outcome = baseline + 10 * demand + effect * treatment + rng.gauss(0, 1)
        rows.append((demand, treatment, outcome))
    return pl.DataFrame(rows, schema=["demand", "treatment", "outcome"], orient="row")


def estimate_effect(data):
    if data.is_empty() or set(data.columns) != {"demand", "treatment", "outcome"}:
        raise ValueError("Expected demand, treatment, outcome evidence")
    if (data.null_count().row(0) != (0, 0, 0)
            or not data["outcome"].is_finite().all()
            or set(data["treatment"].unique()) != {0, 1}
            or not set(data["demand"].unique()) <= {0, 1}):
        raise ValueError("Require finite outcomes, binary variables and treatment positivity")
    means = {t: data.filter(pl.col("treatment") == t)["outcome"].mean() for t in (0, 1)}
    adjusted, variance, strata = 0.0, 0.0, []
    for group in data.partition_by("demand"):
        cells = [group.filter(pl.col("treatment") == t)["outcome"] for t in (0, 1)]
        if min(len(cell) for cell in cells) < 2:
            raise ValueError("Treatment positivity requires both arms in every demand stratum")
        weight = group.height / data.height
        difference = cells[1].mean() - cells[0].mean()
        adjusted += weight * difference
        variance += weight**2 * sum(cell.var(ddof=1) / len(cell) for cell in cells)
        strata.append({"demand": group["demand"][0], "weight": weight,
                       "control_n": len(cells[0]), "treated_n": len(cells[1]), "difference": difference})
    margin = 1.96 * math.sqrt(variance)
    return {"naive": means[1] - means[0], "adjusted": adjusted,
            "ci95": [adjusted - margin, adjusted + margin], "strata": strata,
            "identification": "backdoor adjustment for demand, valid under the specified synthetic SCM",
            "assumptions": ["consistency", "no unmeasured confounding conditional on demand", "positivity"]}


def run(question, *, settings=None, offline=False):
    resolved = resolved_settings(settings)
    config = load_level_config(9)["experiment"]
    result = observational_context(question, settings=resolved, offline=True)
    baseline = float(SakilaDB(resolved).query("SELECT AVG(amount) AS amount FROM payment").item())
    data = simulate_rentals(seed=config["seed"], effect=config["known_effect"], n=config["samples"], baseline=baseline)
    estimate = estimate_effect(data)
    report = {"data_origin": "synthetic; Sakila average payment calibrates the intercept only",
              "seed": config["seed"], "samples": data.height, "known_effect": config["known_effect"],
              "estimates": estimate, "dag": [["demand", "treatment"], ["demand", "outcome"], ["treatment", "outcome"]]}
    directory = resolved.output_dir / "level09_causal"
    evidence = directory / "synthetic_experiment.json"
    evidence.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    wiki = directory / "wiki" / "Identified_Synthetic_Effect.md"
    explanation = (f"Synthetic experiment: naive={estimate['naive']:.3f}; adjusted={estimate['adjusted']:.3f}; "
                   f"known effect={config['known_effect']}. These are not causal estimates for real Sakila stores.")
    if not offline:
        explanation += "\n" + OllamaLLM(resolved).complete(json.dumps(report),
            system="Explain this synthetic causal experiment. Keep it separate from actual Sakila observations.")
    wiki.write_text(f"# Causal experiment / 因果实验\n\n{explanation}\n\n{json.dumps(report, indent=2)}", encoding="utf-8")
    # Generate a semantic vocabulary from the problem variables, explicitly scoped to the simulation.
    ontology = Graph()
    ns = Namespace("https://example.org/sakila/synthetic#")
    for variable in ("demand", "treatment", "outcome"):
        ontology.add((ns[variable], RDF.type, OWL.Class))
        ontology.add((ns[variable], RDFS.comment, Literal("Synthetic SCM variable; not an empirical causal discovery")))
    ontology_path = directory / "ontology" / "synthetic-causal.owl"
    ontology.serialize(str(ontology_path), format="xml")
    result.artifacts.extend(map(str, [evidence, wiki, ontology_path]))
    result.metadata["causal_experiment"] = report
    result.answer += "\n" + explanation
    return result
