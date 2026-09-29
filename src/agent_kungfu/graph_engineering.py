from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef

from .database import SakilaDB


_FK = re.compile(
    r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+[\"`\[]?([\w]+)[\"`\]]?\s*\(([^)]+)\)",
    re.IGNORECASE,
)


def build_schema_graph(db: SakilaDB) -> nx.DiGraph:
    graph = nx.DiGraph(kind="sakila-schema")
    for row in db.objects().filter(type="table").to_dicts():
        name = str(row["name"])
        ddl = str(row.get("sql") or "")
        graph.add_node(name, kind="table", ddl=ddl)
        for source_column, target, target_column in _FK.findall(ddl):
            graph.add_edge(
                name,
                target,
                relation="foreign_key",
                source_column=source_column.strip(' "`[]'),
                target_column=target_column.strip(' "`[]'),
            )
    return graph


def graph_summary(graph: nx.DiGraph) -> dict[str, Any]:
    undirected = graph.to_undirected()
    return {
        "tables": graph.number_of_nodes(),
        "relationships": graph.number_of_edges(),
        "components": nx.number_connected_components(undirected),
        "most_connected": sorted(dict(graph.degree()).items(), key=lambda pair: pair[1], reverse=True)[:5],
        "cycles": [cycle for cycle in nx.simple_cycles(graph)][:10],
    }


def shortest_join_path(graph: nx.DiGraph, source: str, target: str) -> list[str]:
    return nx.shortest_path(graph.to_undirected(), source=source, target=target)


@dataclass(slots=True)
class OntologyArtifacts:
    owl: Path
    obda: Path
    annotation: Path
    annotation_v3: Path

    def paths(self) -> list[str]:
        return [str(self.owl), str(self.obda), str(self.annotation), str(self.annotation_v3)]


def generate_sakila_ontology(db: SakilaDB, output_dir: Path) -> OntologyArtifacts:
    """Generate OWL/OBDA/annotation assets from the live Sakila schema graph."""

    output_dir.mkdir(parents=True, exist_ok=True)
    schema = build_schema_graph(db)
    rdf = Graph()
    SAKILA = Namespace("https://example.org/sakila#")
    rdf.bind("sakila", SAKILA)
    rdf.bind("owl", OWL)

    ontology = URIRef("https://example.org/sakila")
    rdf.add((ontology, RDF.type, OWL.Ontology))
    rdf.add((ontology, RDFS.label, Literal("Sakila domain ontology")))

    for table in schema.nodes:
        cls = SAKILA[_camel(table)]
        rdf.add((cls, RDF.type, OWL.Class))
        rdf.add((cls, RDFS.label, Literal(table)))

    for source, target, data in schema.edges(data=True):
        prop = SAKILA[f"{source}_{data['source_column']}"]
        rdf.add((prop, RDF.type, OWL.ObjectProperty))
        rdf.add((prop, RDFS.domain, SAKILA[_camel(source)]))
        rdf.add((prop, RDFS.range, SAKILA[_camel(target)]))

    # Add core business concepts that are not literal database tables.
    for concept in ("RentalTransaction", "RevenueEvent", "CustomerValue", "FilmDemand"):
        rdf.add((SAKILA[concept], RDF.type, OWL.Class))
    rdf.add((SAKILA["amount"], RDF.type, OWL.DatatypeProperty))
    rdf.add((SAKILA["amount"], RDFS.range, XSD.decimal))

    owl_path = output_dir / "ontology-man-sakilas.owl"
    rdf.serialize(destination=str(owl_path), format="xml")

    obda_path = output_dir / "ontology-man-sakilas.obda"
    mappings = ["[PrefixDeclaration]", "sakila: https://example.org/sakila#", "", "[MappingDeclaration] @collection [["]
    for table in schema.nodes:
        mappings.extend(
            [
                f"mappingId {table}",
                f"target sakila:{_camel(table)}/{{rowid}} a sakila:{_camel(table)} .",
                f"source SELECT rowid, * FROM {table}",
                "",
            ]
        )
    mappings.append("]]\n")
    obda_path.write_text("\n".join(mappings), encoding="utf-8")

    annotation_path = output_dir / "anno_sakila.annotation"
    annotation_v3_path = output_dir / "anno_sakila_man3.annotation"
    annotation_path.write_text(_annotation_text(schema, detailed=False), encoding="utf-8")
    annotation_v3_path.write_text(_annotation_text(schema, detailed=True), encoding="utf-8")
    return OntologyArtifacts(owl_path, obda_path, annotation_path, annotation_v3_path)


def _camel(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_"))


def _annotation_text(graph: nx.DiGraph, *, detailed: bool) -> str:
    lines = ["# Sakila ontology annotations", "# generated from sqlite_master"]
    for table in sorted(graph.nodes):
        lines.append(f"table:{table}\tclass:https://example.org/sakila#{_camel(table)}")
    if detailed:
        for source, target, data in sorted(graph.edges(data=True)):
            lines.append(
                f"fk:{source}.{data['source_column']}\trelates:{target}.{data['target_column']}"
            )
    return "\n".join(lines) + "\n"


def generate_wiki(db: SakilaDB, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_schema_graph(db)
    profile = db.profile()
    pages: list[Path] = []
    index_lines = ["# Sakila Knowledge Wiki", "", "Generated from the live SQLite schema.", ""]
    for table in sorted(graph.nodes):
        neighbors = sorted(graph.to_undirected().neighbors(table))
        ddl = graph.nodes[table].get("ddl", "")
        page = output_dir / f"{table}.md"
        page.write_text(
            "\n".join(
                [
                    f"# {table}",
                    "",
                    f"Rows: {profile.get(table, 0)}",
                    "",
                    f"Related tables: {', '.join(neighbors) if neighbors else 'none'}",
                    "",
                    "## Schema",
                    "",
                    "```sql",
                    ddl,
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        pages.append(page)
        index_lines.append(f"- [{table}]({table}.md) — {profile.get(table, 0)} rows")
    index = output_dir / "README.md"
    index.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return [index, *pages]
