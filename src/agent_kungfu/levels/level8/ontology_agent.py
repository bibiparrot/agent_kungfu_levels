"""OWL vocabulary and domain/range gates control publication of relational knowledge."""
import json

from rdflib import Graph, Namespace, OWL, RDF, RDFS

from ...database import SakilaDB
from ...graph_engineering import build_schema_graph, generate_sakila_ontology, generate_wiki
from ...llm import OllamaLLM
from ..common import resolved_settings
from ..level6.graph_agent import run as graph_run

SAKILA = Namespace("https://example.org/sakila#")


def vocabulary(ontology_path):
    graph = Graph().parse(str(ontology_path))
    claims = []
    for predicate in sorted(graph.subjects(RDF.type, OWL.ObjectProperty), key=str):
        domain, range_ = graph.value(predicate, RDFS.domain), graph.value(predicate, RDFS.range)
        if domain is not None and range_ is not None:
            claims.append({"subject_type": str(domain).split("#")[-1],
                           "predicate": str(predicate).split("#")[-1],
                           "object_type": str(range_).split("#")[-1]})
    return graph, claims


def publish_knowledge(db, ontology_path, claims, output_path):
    ontology, _ = vocabulary(ontology_path)
    schema = build_schema_graph(db)
    lines = ["# Ontology-governed knowledge / 本体约束知识", "",
             "Scope: explicit OWL classes and property domain/range; no complete OWL reasoner.", ""]
    if not isinstance(claims, list) or not claims:
        raise ValueError("At least one typed claim is required")
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"subject_type", "predicate", "object_type"}:
            raise ValueError("Claim requires exactly subject_type, predicate, object_type")
        subject, predicate, target = (SAKILA[claim[key]] for key in ("subject_type", "predicate", "object_type"))
        if ((subject, RDF.type, OWL.Class) not in ontology or (target, RDF.type, OWL.Class) not in ontology
                or (predicate, RDF.type, OWL.ObjectProperty) not in ontology
                or (predicate, RDFS.domain, subject) not in ontology
                or (predicate, RDFS.range, target) not in ontology):
            raise ValueError("Claim violates ontology classes or property domain/range")
        source_table = str(ontology.value(subject, RDFS.label))
        target_table = str(ontology.value(target, RDFS.label))
        if not schema.has_edge(source_table, target_table):
            raise ValueError("Ontology relationship lacks a database mapping")
        edge = schema.edges[source_table, target_table]
        if claim["predicate"] != f"{source_table}_{edge['source_column']}":
            raise ValueError("Property does not match the foreign-key mapping")
        sql = (f'SELECT COUNT(*) AS links FROM "{source_table}" s JOIN "{target_table}" t '
               f'ON s."{edge["source_column"]}" = t."{edge["target_column"]}"')
        count = db.query(sql).item(0, "links")
        lines.extend([f"## {claim['subject_type']} → {claim['predicate']} → {claim['object_type']}",
                      f"Observed links: {count}", f"```sql\n{sql}\n```", ""])
    # Validate every claim before publishing anything.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run(question, *, settings=None, offline=False, use_deepeval=False):
    resolved = resolved_settings(settings)
    db = SakilaDB(resolved)
    ontology = generate_sakila_ontology(db, resolved.output_dir / "level08_ontology")
    _, claims = vocabulary(ontology.owl)
    if not offline:
        response = OllamaLLM(resolved).complete(
            f"Question: {question}\nAllowed typed relations: {json.dumps(claims)}\n"
            "Return a nonempty JSON array selecting relevant relations without changing their fields.",
            system="Select knowledge claims from the supplied ontology vocabulary.")
        claims = json.loads(response)
    page = publish_knowledge(db, ontology.owl, claims, resolved.output_dir / "level08_wiki" / "SEMANTIC_KNOWLEDGE.md")
    pages = generate_wiki(db, page.parent)
    result = graph_run(question, settings=resolved, offline=offline, use_deepeval=use_deepeval)
    result.level, result.driver = 8, "Ontology Driven Agent"
    result.artifacts = [*ontology.paths(), *(str(path) for path in pages), str(page)]
    result.metadata["semantic_gate"] = {"accepted_claims": len(claims), "published": str(page),
                                         "scope": "OWL class/domain/range + FK mapping + query evidence"}
    result.answer += " 本体约束和数据库映射已验证；语义知识页仅包含通过的关系。"
    return result
