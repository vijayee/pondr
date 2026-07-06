"""GLiNER-based entity extraction for hippocampal memory.

Two models cooperate:
- GLiNER-Decoder (``knowledgator/gliner-decoder-base-v1.0``): open discovery —
  invents labels freely, surfacing entity types the stable schema does not yet
  know about. Discoveries are buffered; labels that recur across enough episodes
  become promotion candidates for the schema to absorb.
- GLiNER2 (``fastino/gliner2-base-v1``): stable extraction against the fixed
  schema below (entities / topics / tones / decisions).

Both models are heavy (GPU). They load on ``__init__``, so construct the
extractor on the RunPod GPU pod, not in the local editing loop. The class
itself is importable without ``gliner``/``gliner2`` installed — the imports are
deferred to construction time so the encoder orchestrator and tests can
reference ``GLiNERExtractor`` on a CPU box.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..config import config


# Stable schema GLiNER2 extracts against. GLiNER2's schema only processes
# `entities`, `relations`, `classifications`, and `structures` as top-level
# keys — any other top-level key is silently dropped. Span-typed things
# (person / project / technology) are `entities` (dict of label→prompt); the
# model returns them as `{label: [span, ...]}`. Whole-text labels (topics /
# tones / decisions) are `classifications` (list of task configs); the model
# returns each task as `{task_name: [predicted_label, ...]}`. The first RunPod
# measurement run scored 0.00 topic/tone recall because topics/tones/decisions
# were modeled as top-level dicts and dropped — this layout fixes that.
#
# Evolving this schema (promoting a discovered label into it) is what the
# discovery buffer feeds.
_STABLE_SCHEMA: dict[str, Any] = {
    "entities": {
        "person": "Full name of a person mentioned",
        "project": "Software project or system name",
        "technology": "Technical tool, framework, or concept",
    },
    "classifications": [
        {
            "task": "topics",
            "labels": [
                "database_design",
                "configuration",
                "graph_database",
                "performance",
                "decision_making",
                "ai_architecture",
                "api_design",
                "security",
            ],
            "multi_label": True,
        },
        {
            "task": "tones",
            "labels": ["frustrated", "excited", "curious", "neutral"],
            "multi_label": True,
        },
        {
            # GLiNER2 requires >=2 labels per classification task, so pair the
            # single "decision" label with a "none" sentinel and filter the
            # sentinel out in _extract_stable.
            "task": "decisions",
            "labels": ["decision", "none"],
            "multi_label": False,
        },
    ],
}


class GLiNERExtractor:
    """Extracts entities, topics, tones, and decisions from conversation text.

    GLiNER-Decoder discovers new entity types; GLiNER2 extracts against the
    stable schema. ``extract`` runs both and returns the merge.
    """

    def __init__(
        self,
        gliner2_model: str | None = None,
        gliner_decoder_model: str | None = None,
        threshold: float | None = None,
        promotion_threshold: int | None = None,
    ):
        # Defaults come from the central Config so local + RunPod runs use the
        # same model selection. Constructor args override for experiments/tests.
        gliner2_model = gliner2_model or config.gliner2_model
        gliner_decoder_model = gliner_decoder_model or config.gliner_decoder_model
        self.threshold = threshold if threshold is not None else config.extraction_threshold
        self.promotion_threshold = (
            promotion_threshold if promotion_threshold is not None
            else config.discovery_buffer_threshold
        )

        # Heavy imports are deferred so the module is importable on a CPU box
        # (the encoder orchestrator imports this class; only the RunPod pod
        # actually constructs it). A missing dep surfaces as a clear error
        # pointing at the right package rather than an ImportError at module
        # load.
        try:
            from gliner import GLiNER  # type: ignore
            from gliner2 import GLiNER2  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised only on RunPod
            raise ImportError(
                "GLiNERExtractor requires the 'gliner' and 'gliner2' packages. "
                "Install them on the RunPod GPU pod (pip install gliner gliner2)."
            ) from e

        # Model download happens here. Per the plan's process instruction, a
        # failed download must be reported verbatim — do not swallow HuggingFace
        # errors, let them propagate.
        self.discoverer = GLiNER.from_pretrained(gliner_decoder_model)
        self.extractor = GLiNER2.from_pretrained(gliner2_model)
        self.discovery_buffer: dict[str, list[str]] = defaultdict(list)

    def extract(self, text: str) -> dict:
        """Extract structured information from conversation text.

        Returns:
            {
                "entities": ["Alice", "WaveDB", "HBTrie"],
                "topics": ["database_design", "configuration"],
                "tones": ["frustrated", "curious"],
                "decisions": ["use_hbtrie"],
                "discovered": [
                    {"text": "WAL config", "label": "technical concept"},
                ],
            }
        """
        stable = self._extract_stable(text)
        discovered = self._extract_open(text)
        self._buffer_discoveries(discovered)

        return {**stable, "discovered": discovered}

    def _extract_stable(self, text: str) -> dict:
        """Extract using GLiNER2 against the stable schema.

        GLiNER2 returns `entities` as a dict of {category: [spans]} and each
        classification task as {task_name: [predicted_labels]}. We flatten the
        person/project/technology entity spans and pass the topic/tone/decision
        label lists through, dropping the "none" sentinel used to satisfy the
        >=2-label-per-classification requirement.
        """
        result: dict = self.extractor.extract(
            text, schema=_STABLE_SCHEMA, threshold=self.threshold
        )

        entities: list[str] = []
        for category in ("person", "project", "technology"):
            entities.extend(result.get("entities", {}).get(category, []))

        topics = [t for t in result.get("topics", []) if t and t != "none"]
        tones = [t for t in result.get("tones", []) if t and t != "none"]
        decisions = [d for d in result.get("decisions", []) if d and d != "none"]

        return {
            "entities": list(set(entities)),
            "topics": topics,
            "tones": tones,
            "decisions": decisions,
        }

    def _extract_open(self, text: str) -> list[dict]:
        """Open discovery using GLiNER-Decoder."""
        entities: list[dict] = self.discoverer.predict_entities(
            text, labels=["label"], threshold=self.threshold
        )
        return [{"text": e["text"], "label": e["label"]} for e in entities]

    def _buffer_discoveries(self, discovered: list[dict]) -> None:
        """Buffer discovered labels for potential promotion."""
        for item in discovered:
            self.discovery_buffer[item["label"]].append(item["text"])

    def get_promotion_candidates(self) -> list[str]:
        """Get labels that have crossed the promotion threshold."""
        return [
            label for label, examples in self.discovery_buffer.items()
            if len(examples) >= self.promotion_threshold
        ]