from __future__ import annotations

import re
from dataclasses import dataclass


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Zа-яА-Я0-9]+", "-", value.lower()).strip("-")
    return normalized or "concept"


@dataclass
class LearningOntology:
    subject: str
    concepts: dict[str, dict]

    @classmethod
    def from_seed_content(
        cls,
        topics: list[dict],
        tasks: list[dict],
        ontology_data: dict | None = None,
    ) -> "LearningOntology":
        ontology_data = ontology_data or {}
        subject = str(ontology_data.get("subject", "Общий курс"))
        concepts: dict[str, dict] = {}

        for concept in ontology_data.get("concepts", []):
            label = str(concept.get("label", "")).strip()
            if not label:
                continue
            concepts[label] = {
                "id": str(concept.get("id", _slug(label))),
                "label": label,
                "prerequisites": [
                    str(item).strip() for item in concept.get("prerequisites", []) if str(item).strip()
                ],
            }

        for topic in topics:
            title = str(topic.get("title", "")).strip()
            if title and title not in concepts:
                concepts[title] = {
                    "id": _slug(title),
                    "label": title,
                    "prerequisites": [],
                }

        for task in tasks:
            for question in task.get("questions", []):
                label = str(
                    question.get("concept")
                    or question.get("area")
                    or task.get("topic_title")
                    or "Общее"
                ).strip()
                if label and label not in concepts:
                    concepts[label] = {
                        "id": _slug(label),
                        "label": label,
                        "prerequisites": [],
                    }

        for concept in concepts.values():
            concept["prerequisites"] = [
                item for item in concept["prerequisites"] if item in concepts
            ]

        return cls(subject=subject, concepts=concepts)

    def concept_for_question(self, question: dict, fallback: str = "Общее") -> str:
        label = str(question.get("concept") or question.get("area") or fallback).strip()
        if not label:
            return fallback
        return label

    def infer_support_concepts(self, weak_concepts: list[str]) -> list[str]:
        support: list[str] = []
        seen = set(weak_concepts)
        queue = list(weak_concepts)
        while queue:
            current = queue.pop(0)
            node = self.concepts.get(current)
            if not node:
                continue
            for prerequisite in node.get("prerequisites", []):
                if prerequisite in seen:
                    continue
                seen.add(prerequisite)
                support.append(prerequisite)
                queue.append(prerequisite)
        return support

    def to_turtle(self) -> str:
        lines = [
            "@prefix lms: <https://example.org/lms#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "",
            f'lms:Subject rdfs:label "{self.subject}" .',
            "",
        ]
        for concept in self.concepts.values():
            uri = f"lms:{concept['id']}"
            lines.append(f"{uri} a lms:Concept ;")
            lines.append(f'    rdfs:label "{concept["label"]}" .')
            for prerequisite in concept.get("prerequisites", []):
                pre_id = self.concepts[prerequisite]["id"]
                lines.append(f"{uri} lms:requiresConcept lms:{pre_id} .")
            lines.append("")
        return "\n".join(lines)
