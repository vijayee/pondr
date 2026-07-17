"""Bonsai-based relation extraction for hippocampal memory.

Bonsai (Prism-ML Ternary-Bonsai-8B) is an OpenAI-compatible small LLM served
by the Prism fork of llama.cpp's ``llama-server``. The server runs on the
RunPod GPU pod — there is no local llama-server — so this client talks to a
configurable HTTP endpoint (``config.bonsai_endpoint``, override via
``BONSAI_ENDPOINT`` or the constructor). When the encoding pipeline runs on the
same pod as ``llama-server`` the endpoint is ``http://localhost:8080/v1``;
when run remotely it is the pod's public URL.

The connection is opened lazily (on the first ``extract`` call), so the class is
constructible offline and the module imports without the server present. HTTP
and parse failures surface verbatim per the plan's process instruction.
"""

from __future__ import annotations

import json
import re

import requests

from ..config import config


BONSAI_RELATION_PROMPT = """Extract relationships from this conversation.
Return ONLY valid JSON, no other text.

Extract AT MOST 6 of the most important relations — prefer the salient few
over exhaustively listing every mention, or the response may truncate.

Relation types:
- explains(Person, Concept): someone explains something
- decides(Person, Decision): someone makes a decision
- expresses(Person, Tone): someone expresses an emotion
- questions(Person, Concept): someone asks about something
- suggests(Person, Concept): someone proposes an idea
- concerns(Episode, Topic): the conversation is about a topic
- involves(Episode, Entity): an entity participates in the conversation
- contradicts(Statement, Statement): one statement contradicts another
- has_state(Entity, Value): an ENTITY's current state/value/choice -- the
  subject is a tool/team/ticket/policy/project, NOT a person (a person making
  a choice is ``decides``). Use for explicit "the team chose X", "status: Y",
  "X is now Z", "switched to W"; Value is the literal value (a tool name, a
  status, a number).
- follows_up_on(Episode, Episode): this conversation continues from another

Conversation:
{text}

Return JSON:
{{"relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}"""


# Matches a ```json ... ``` (or bare ```) fenced block. The model is told to
# return ONLY JSON, but small models sometimes wrap output in fences anyway.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

# Cap on the number of relations kept — matches the prompt's "at most 6". The
# small Bonsai model over-extracts (30+ "explains" entries on a chatty
# conversation), which blows past the response token limit and truncates the
# JSON mid-stream. Capping the kept set bounds output size and favors salience.
_MAX_RELATIONS = 6

# Per-class cap for the isolated 10-pass extractor. The whole point of
# isolation is NO salience cap (the V1 "at most 6" is what loses has_state), so
# this is a generous pure-safety valve against a pathological over-extract, not
# a salience filter. Realistic input yields <10 relations per class
# (~4.9 rels/doc total across all 10 classes per the probe).
_MAX_RELATIONS_ISOLATED = 64


# The 10 canonical predicate classes, each with a focused single-predicate
# directive. Isolation removes the salience race: one pass per class, no
# competing predicates, no "at most 6" cap -> has_state no longer loses to
# decides/concerns/involves for the top slots. Proven 11/13 strict has_state
# zero-shot (vs V1's 0/13), all 10 classes emit, neg FP 0
# (scripts/_scratch/_probe_isolate_classes.py, uncommitted). The cost is 10 HTTP
# round-trips (~22.8 s/doc) -> only viable behind async_distill_enabled.
ISOLATION_CLASSES: list[tuple[str, str, str]] = [
    ("has_state", "has_state(Entity, Value)",
     "an ENTITY's current state/value/choice. The subject is a tool, team, "
     "ticket, policy, project, system, framework, database, or service -- "
     "NEVER a person, NEVER a topic. Use for explicit 'the team chose X', "
     "'status: Y', 'X is now Z', 'switched to W', 'we use X', 'the framework "
     "is X'. Value is the literal value (e.g. Postgres, React, red, v2), not "
     "a topic. Extract one has_state per distinct entity-value pair."),
    ("decides", "decides(Person, Decision)",
     "a person decides, chooses, picks, or commits to a course of action or "
     "option. The subject MUST be a person. The object is the decision/choice."),
    ("expresses", "expresses(Person, Tone)",
     "a person expresses a tone or emotion (e.g. frustrated, optimistic, "
     "concerned, enthusiastic). Subject is a person; object is the tone."),
    ("questions", "questions(Person, Concept)",
     "a person asks a question about a concept or topic. Subject is a person; "
     "object is the concept being asked about."),
    ("suggests", "suggests(Person, Concept)",
     "a person suggests, proposes, or recommends an idea or option. Subject "
     "is a person; object is the suggested idea/option."),
    ("explains", "explains(Person, Concept)",
     "a person explains a concept to someone. Subject is a person; object is "
     "the concept being explained."),
    ("concerns", "concerns(Episode, Topic)",
     "the conversation/episode is about a topic. Subject is the episode (use "
     "'episode'); object is the topic."),
    ("involves", "involves(Episode, Entity)",
     "the conversation/episode involves an entity (tool, team, service, "
     "person). Subject is the episode (use 'episode'); object is the entity."),
    ("contradicts", "contradicts(Statement, Statement)",
     "one statement in the conversation contradicts another. Subject and "
     "object are the two contradicting statements (short quotes or summaries)."),
    ("follows_up_on", "follows_up_on(Episode, Episode)",
     "the conversation follows up on a prior episode. Both subject and object "
     "are episodes."),
]

_ISO_TEMPLATE = """Extract ONLY __SIG__ relations from this conversation.
Return ONLY valid JSON, no other text.
A __PRED__ relation is: __DIRECTIVE__
Extract EVERY __PRED__ relation you can find in the conversation. Do NOT
extract any other relation type. Emit the predicate as the exact string
"__PRED__".
Conversation:
{text}
Return JSON:
{{"relations": [{{"subject": "...", "predicate": "__PRED__", "object": "..."}}]}}"""


def _iso_prompt(pred: str, sig: str, directive: str) -> str:
    """Build the isolated single-predicate prompt for one class.

    Uses ``__SIG__`` / ``__PRED__`` / ``__DIRECTIVE__`` placeholders (not
    ``.format`` fields) so the per-call ``{text}`` substitution in ``extract``
    is the only ``.format`` pass -- avoids a KeyError when the directive text
    contains braces.
    """
    return (
        _ISO_TEMPLATE
        .replace("__SIG__", sig)
        .replace("__PRED__", pred)
        .replace("__DIRECTIVE__", directive)
    )


def _scan_complete_relation_objects(body: str) -> list[dict]:
    """Salvage complete ``{...}`` JSON objects from a possibly-truncated body.

    Walks the body tracking brace depth + string state, extracting every
    *balanced* ``{...}`` substring and ``json.loads``-ing it. Keeps only dicts
    with the ``subject``/``predicate``/``object`` relation keys. An unbalanced
    (truncated) object is skipped — we advance past its opening brace and keep
    scanning, so complete objects nested earlier in an outer envelope that
    failed to close are still recovered. Used to recover partial relations
    when the model's JSON is truncated mid-stream by the ``max_tokens`` cap.
    """
    out: list[dict] = []
    n = len(body)
    i = 0
    while i < n:
        if body[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        end = -1
        while j < n:
            c = body[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
            j += 1
        if end == -1:
            # This object is unbalanced (truncated). Skip past its opening
            # brace and keep looking for complete objects further on — the
            # truncation only affects this object, not earlier complete ones
            # nested inside an outer envelope that itself failed to close.
            i += 1
            continue
        try:
            obj = json.loads(body[i : end + 1])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and {"subject", "predicate", "object"} <= obj.keys():
            out.append(obj)
        i = end + 1
    return out


class BonsaiRelationExtractor:
    """Extracts structured relations from conversation text.

    Talks to an OpenAI-compatible ``llama-server`` (Bonsai) over HTTP. The
    server is not assumed to be local — the endpoint is configurable and
    defaults to ``config.bonsai_endpoint`` (``BONSAI_ENDPOINT`` env var).
    """

    # Foreground-priority yielding hook (Phase 3c async-distill). Class-level
    # default so ``object.__new__``-built instances (test fixtures) carry the
    # attribute; ``extract_isolated`` reads it before each per-class HTTP call.
    # The distill worker sets an instance attribute (shadowing this) to a
    # blocking callable and clears it after the fill. ``None`` = no yielding,
    # byte-identical to the synchronous single-pass path.
    pause_gate: object | None = None

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        temperature: float | None = None,
        timeout: float = 60.0,
    ):
        self.model = model or config.bonsai_model
        self.endpoint = (endpoint or config.bonsai_endpoint).rstrip("/")
        self.temperature = temperature if temperature is not None else config.bonsai_temperature
        self.timeout = timeout

    def extract(self, text: str, *, isolated: bool | None = None) -> list[dict]:
        """Extract relations as (subject, predicate, object) triples.

        Dispatches on ``config.bonsai_isolation_extraction`` (overridable via
        the ``isolated`` kwarg): ``False`` (default) = the V1 single-pass
        ``BONSAI_RELATION_PROMPT`` (byte-identical to pre-async); ``True`` = the
        10-pass isolated per-class extractor (``extract_isolated``), which lifts
        strict ``has_state`` catch 0 -> 11/13 zero-shot at the cost of 10 HTTP
        round-trips. The isolation path is only viable behind
        ``async_distill_enabled`` (the ~22.8 s/doc runs on the background
        worker); enabling isolation without async would block the response.

        Raises ``RuntimeError`` with the exact server response if the request
        fails or the model returns non-JSON, so the caller can log the raw
        output rather than silently dropping relations.
        """
        if isolated is None:
            isolated = config.bonsai_isolation_extraction
        if isolated:
            return self.extract_isolated(text)
        return self._extract_single(text)

    def _post(self, prompt: str, text: str, *, max_tokens: int = 768) -> str:
        """One HTTP round-trip to the Bonsai chat endpoint -> raw content string.

        Shared by the V1 single pass and each isolated per-class pass. Raises
        ``RuntimeError`` verbatim on network / HTTP / shape failures so the
        caller's try/except (the encoder degrades a single failed pass to
        empty) sees the real error.
        """
        url = f"{self.endpoint}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt.format(text=text)}],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            # Bound the response so an over-extracting turn truncates the JSON
            # instead of running away. _parse_relations recovers complete
            # relation objects from a truncated stream.
            "max_tokens": max_tokens,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise RuntimeError(f"Bonsai request to {url} failed: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"Bonsai endpoint {url} returned HTTP {resp.status_code}: {resp.text}"
            )

        try:
            outer = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Bonsai returned non-JSON body: {resp.text}") from e

        try:
            return outer["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"Bonsai response missing choices[0].message.content: {outer}"
            ) from e

    def _extract_single(self, text: str) -> list[dict]:
        """The V1 single-pass extractor (the default). One HTTP call with the
        merged ``BONSAI_RELATION_PROMPT``; output capped at ``_MAX_RELATIONS``."""
        content = self._post(BONSAI_RELATION_PROMPT, text, max_tokens=768)
        return self._parse_relations(content)

    def extract_isolated(self, text: str) -> list[dict]:
        """The 10-pass isolated per-class extractor.

        One focused single-predicate pass per class in ``ISOLATION_CLASSES``,
        merged. Each pass's predicate is force-normalized to the exact class
        name (the prompt asks for it, but the ternary 8B sometimes paraphrases
        the predicate string). No salience cap: ``_MAX_RELATIONS_ISOLATED`` is a
        generous per-class safety valve, not the V1 "at most 6" filter. A
        failed pass degrades to empty for that class (the merged result keeps
        the other classes) -- one class's HTTP/parse hiccup does not drop the
        whole extraction, mirroring the V1 degrade-to-empty philosophy.
        """
        merged: list[dict] = []
        for pred, sig, directive in ISOLATION_CLASSES:
            # Yield to the foreground before each GPU-using HTTP call: a
            # foreground query() that lands while this pass is mid-flight would
            # queue behind it on the shared 8B. ``pause_gate`` blocks while the
            # foreground is busy; None (sync path) is a no-op.
            if self.pause_gate is not None:
                self.pause_gate()
            prompt = _iso_prompt(pred, sig, directive)
            try:
                content = self._post(prompt, text, max_tokens=768)
                rels = self._parse_relations(content, max_relations=_MAX_RELATIONS_ISOLATED)
            except Exception as e:  # noqa: BLE001 - one class fails, keep the rest
                # Degrade this class to empty rather than failing the whole
                # extraction; the caller (the async worker) logs via the
                # encoder's _extract_relations try/except if ALL classes fail.
                rels = []
            for r in rels:
                if isinstance(r, dict):
                    r["predicate"] = pred
            merged.extend(r for r in rels if isinstance(r, dict))
        return merged

    @staticmethod
    def _parse_relations(content: str, *, max_relations: int = _MAX_RELATIONS) -> list[dict]:
        """Parse the model's JSON content into a list of relation dicts.

        Strips accidental ``` fences and, failing that, falls back to the
        outermost ``{...}`` span. If that also fails (typically because the
        model over-extracted and the JSON was truncated mid-stream by the
        ``max_tokens`` cap), recover every *complete* ``{"subject":...,
        "predicate":..., "object":...}`` object from the truncated body rather
        than discarding the whole response. Raises with the raw content only
        when no complete relation object can be recovered at all — that is a
        real extraction failure, not an empty result, and the caller needs the
        raw text to debug it.
        """
        body = content.strip()
        fence = _FENCE_RE.match(body)
        if fence:
            body = fence.group(1).strip()

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            # Last resort: carve out the outermost object. Handles trailing
            # prose the model may have added despite "ONLY valid JSON".
            start, end = body.find("{"), body.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(body[start : end + 1])
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

        if data is None:
            # Truncated mid-stream (or otherwise malformed): salvage whatever
            # complete relation objects we can find in the body.
            salvaged = _scan_complete_relation_objects(body)
            if salvaged:
                return salvaged[:max_relations]
            raise RuntimeError(f"Bonsai returned unparseable JSON: {content!r}") from None

        relations = data.get("relations", []) if isinstance(data, dict) else []
        # Defend against the model returning a bare list instead of the
        # documented {"relations": [...]} envelope.
        if isinstance(data, list):
            relations = data

        out = [
            r for r in relations
            if isinstance(r, dict) and {"subject", "predicate", "object"} <= r.keys()
        ]
        return out[:max_relations]