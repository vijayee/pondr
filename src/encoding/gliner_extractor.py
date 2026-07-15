"""GLiNER-based entity extraction for hippocampal memory.

Two models cooperate:
- GLiNER-Decoder (``knowledgator/gliner-decoder-base-v1.0``): open discovery —
  invents labels freely, surfacing entity types the stable schema does not yet
  know about. Discoveries are buffered; labels that recur across enough episodes
  become promotion candidates for the schema to absorb.
- GLiNER2 (``fastino/gliner2-base-v1``): stable extraction against the fixed
  schema below — entities / topics / decisions as text spans, tones as a
  fixed-label classification (see ``_STABLE_SCHEMA`` for the layout).

Both models are heavy (GPU). They load on ``__init__``, so construct the
extractor on the RunPod GPU pod, not in the local editing loop. The class
itself is importable without ``gliner``/``gliner2`` installed — the imports are
deferred to construction time so the encoder orchestrator and tests can
reference ``GLiNERExtractor`` on a CPU box.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import Any

import torch

from ..config import config


# Stable schema GLiNER2 extracts against. GLiNER2's schema only processes
# `entities`, `relations`, `classifications`, and `structures` as top-level
# keys — any other top-level key is silently dropped. Span-typed things
# (person / project / technology / decision / topic) are `entities` (dict of
# label→prompt); the model returns them as `{label: [span, ...]}`. Whole-text
# labels (tones) are `classifications` (list of task configs); the model
# returns each task as `{task_name: [predicted_label, ...]}`. The first RunPod
# measurement run scored 0.00 topic/tone recall because topics/tones/decisions
# were modeled as top-level dicts and dropped — the classifications layout
# fixed that. A later scale run on DialogSum then surfaced two more defects
# (see _extract_stable + _as_list) which moved decisions and topics out of
# `classifications` and into `entities` (span extraction).
#
# Decisions and topics are SPANS, not classifications:
# - Decisions: the first scale run modeled decisions as a binary
#   classification (labels ["decision","none"], multi_label:False). GLiNER2
#   returned a bare STRING "decision"/"none" for that task, which
#   _extract_stable iterated char-by-char → ["d","e","c","i","s","i","o","n"].
#   The "none" sentinel filter compared chars to the string "none" so it never
#   matched either. Even parsed correctly, the value was the useless label
#   NAME, not actual decision content. Span extraction returns real decision
#   phrases (e.g. "go with DEBOUNCED", "the 3pm appointment") as a list of
#   strings — char-split is impossible and the content is meaningful.
# - Topics: a fixed tech-domain label set (database_design/configuration/...)
#   collapsed on daily-life corpora like DialogSum, forcing every
#   conversation into api_design/decision_making. Span extraction yields
#   varied, corpus-derived topic phrases so topic-based retrieval
#   discriminates on any corpus, not just the project's own tech domain.
#
# Tones stay a fixed classification: AffectiveTone is a small bounded taxonomy
# (frustrated/excited/curious/neutral) that discriminates well in practice.
#
# Evolving this schema (promoting a discovered label into it) is what the
# discovery buffer feeds.
_STABLE_SCHEMA: dict[str, Any] = {
    "entities": {
        "person": "Full name of a person mentioned",
        "project": "Software project or system name",
        "technology": "Technical tool, framework, or concept",
        # Actual decision/choice text — spans, not a yes/no label.
        "decision": "A specific decision, choice, or conclusion someone reaches or agrees on in the conversation",
        # Free-form subject spans — the matter the conversation is about.
        "topic": "The main subject, activity, or matter the conversation is about",
    },
    "classifications": [
        {
            "task": "tones",
            "labels": ["frustrated", "excited", "curious", "neutral"],
            "multi_label": True,
        },
    ],
}

# A3: GLiNER2 entity category -> seed ontology class (Entity subclasses, see
# ontology.CONVERSATIONAL_CLASSES: Entity -> [Person, Project, Technology,
# Concept]). Drives the ``instanceOf`` typing edges the encoder emits.
_CATEGORY_TO_CLASS: dict[str, str] = {
    "person": "Person",
    "project": "Project",
    "technology": "Technology",
}


def _as_list(v) -> list:
    """Coerce a GLiNER2 classification result to a list of strings.

    GLiNER2 classification tasks can return a bare string instead of a list
    (observed for ``multi_label: False`` tasks, where the model emits the
    single chosen label as a string). Iterating a string yields one-character
    strings, which char-splits the label name (e.g. "decision" →
    ["d","e","c","i","s","i","o","n"]). Coerce first so downstream filters see
    the whole label.
    """
    if isinstance(v, str):
        return [v]
    if v is None:
        return []
    return list(v)


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
        device: str = "cpu",
        timing: bool = False,
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
        self._timing = timing

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

        # Resolve the device. "auto" -> CUDA when available, else CPU; "cpu"
        # (the default) keeps every existing caller on CPU byte-identically.
        # These are transformer models (the module docstring notes "both models
        # are heavy (GPU)"); running them on CPU is the documented ~20s/conv
        # ingestion bottleneck (memory gliner-gpu-config-task / wavedb-mvcc-
        # read-amplification-bottleneck). GPU is the intended path for the
        # SERVING entrypoint (which passes "auto"); CPU stays as the
        # always-available fallback (see the OOM guard below).
        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = device

        # Model download happens here. Per the plan's process instruction, a
        # failed download must be reported verbatim — do not swallow HuggingFace
        # errors, let them propagate.
        self.discoverer = GLiNER.from_pretrained(gliner_decoder_model)
        self.extractor = GLiNER2.from_pretrained(gliner2_model)

        # Move each model to the resolved device. The .to("cuda") is OOM-safe:
        # on this dev box the 8B Bonsai server already owns ~14.6GB of the
        # 5080's 16GB VRAM, so two GLiNER models may not fit the remainder and
        # the move can raise. A CUDA OOM must NOT crash the live-encode path --
        # fall back to CPU for that model and log it (the exchange still
        # persists, just slower). The fallback is per-model so a tight-VRAM box
        # can land one model on GPU and the other on CPU.
        self.discoverer, dev_d = self._move_model(self.discoverer, dev, "discoverer")
        self.extractor, dev_e = self._move_model(self.extractor, dev, "extractor")
        # Actual per-model devices (may differ from `dev` after an OOM fallback).
        self._device_d = dev_d
        self._device_e = dev_e
        print(f"[gliner-device] discoverer={dev_d} extractor={dev_e}", file=sys.stderr)
        self.discovery_buffer: dict[str, list[str]] = defaultdict(list)

    @staticmethod
    def _move_model(model, dev: str, name: str):
        """Move a GLiNER model to ``dev`` with an OOM-safe CPU fallback.

        Returns ``(model, resolved_device_str)``. A CUDA move that raises
        (typically out-of-memory when the 8B Bonsai server already fills the
        GPU) falls back to CPU and logs -- it never crashes the encoder, so a
        live exchange still persists even when VRAM is exhausted.
        """
        if dev == "cpu":
            return model, "cpu"
        try:
            model = model.to(dev)
            return model, dev
        except Exception as e:  # noqa: BLE001 - OOM / any device failure -> CPU
            print(f"[gliner-device] {name}: CUDA move failed ({e!r}); "
                  f"falling back to CPU", file=sys.stderr)
            return model, "cpu"

    def extract(self, text: str) -> dict:
        """Extract structured information from conversation text.

        Returns:
            {
                "entities": ["Alice", "WaveDB", "HBTrie"],
                "entity_classes": {"Alice": "Person", "WaveDB": "Project",
                                   "HBTrie": "Technology"},
                "topics": ["WAL config", "storage layer"],
                "tones": ["frustrated", "curious"],
                "decisions": ["go with DEBOUNCED"],
                "discovered": [
                    {"text": "WAL config", "label": "technical concept"},
                ],
            }

        ``topics`` and ``decisions`` are free-form text spans pulled from the
        conversation (GLiNER2 entity extraction), so they vary with the corpus;
        ``tones`` are fixed-label classifications from the bounded
        AffectiveTone taxonomy.
        """
        if self._timing:
            t0 = time.perf_counter()
            stable = self._extract_stable(text)
            t1 = time.perf_counter()
            discovered = self._extract_open(text)
            t2 = time.perf_counter()
            print(f"[gliner-timing] stable={t1 - t0:.3f}s open={t2 - t1:.3f}s "
                  f"total={t2 - t0:.3f}s device_d={self._device_d} "
                  f"device_e={self._device_e}", file=sys.stderr)
        else:
            stable = self._extract_stable(text)
            discovered = self._extract_open(text)
        self._buffer_discoveries(discovered)

        return {**stable, "discovered": discovered}

    def _extract_stable(self, text: str) -> dict:
        """Extract using GLiNER2 against the stable schema.

        GLiNER2 returns `entities` as a dict of {category: [spans]} and each
        classification task as {task_name: [predicted_labels]}. We flatten the
        person/project/technology spans into `entities`, pull `decision` and
        `topic` spans out into their own fields, and pass the `tones`
        classification through (coerced to a list, dropping the "none" sentinel
        if it ever appears). Span fields are lists of strings by construction,
        so there is no char-split path; `_as_list` guards the classification.
        """
        result: dict = self.extractor.extract(
            text, schema=_STABLE_SCHEMA, threshold=self.threshold
        )

        ents: dict = result.get("entities", {}) or {}
        # A3: preserve each entity's GLiNER2 category as a seed-class typing
        # (person->Person, project->Project, technology->Technology) so the
        # encoder can emit ``(E:entity, instanceOf, Class)`` edges. First
        # category wins if a span appears under two (rare); open-discovery
        # entities (GLiNER-Decoder, merged in ``extract``) stay untyped.
        entity_classes: dict[str, str] = {}
        for category in ("person", "project", "technology"):
            cls = _CATEGORY_TO_CLASS[category]
            for span in (ents.get(category, []) or []):
                if span and span not in entity_classes:
                    entity_classes[span] = cls
        entities = list(entity_classes.keys())

        decisions = [d for d in (ents.get("decision", []) or []) if d]
        topics = [t for t in (ents.get("topic", []) or []) if t]
        tones = [t for t in _as_list(result.get("tones")) if t and t != "none"]

        return {
            "entities": entities,
            "entity_classes": entity_classes,
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