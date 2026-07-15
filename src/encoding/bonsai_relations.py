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

    def extract(self, text: str) -> list[dict]:
        """Extract relations as (subject, predicate, object) triples.

        Raises ``RuntimeError`` with the exact server response if the request
        fails or the model returns non-JSON, so the caller can log the raw
        output rather than silently dropping relations.
        """
        url = f"{self.endpoint}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": BONSAI_RELATION_PROMPT.format(text=text)}],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            # Bound the response so an over-extracting turn truncates the JSON
            # instead of running away. _parse_relations recovers complete
            # relation objects from a truncated stream.
            "max_tokens": 768,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            # Network / connection error — surface verbatim. Common cause when
            # the llama-server isn't up or the endpoint is wrong.
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
            content = outer["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Bonsai response missing choices[0].message.content: {outer}") from e

        return self._parse_relations(content)

    @staticmethod
    def _parse_relations(content: str) -> list[dict]:
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
                return salvaged[:_MAX_RELATIONS]
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
        return out[:_MAX_RELATIONS]