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

Relation types:
- explains(Person, Concept): someone explains something
- decides(Person, Decision): someone makes a decision
- expresses(Person, Tone): someone expresses an emotion
- questions(Person, Concept): someone asks about something
- suggests(Person, Concept): someone proposes an idea
- concerns(Episode, Topic): the conversation is about a topic
- involves(Episode, Entity): an entity participates in the conversation
- contradicts(Statement, Statement): one statement contradicts another
- follows_up_on(Episode, Episode): this conversation continues from another

Conversation:
{text}

Return JSON:
{{"relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}"""


# Matches a ```json ... ``` (or bare ```) fenced block. The model is told to
# return ONLY JSON, but small models sometimes wrap output in fences anyway.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


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
        outermost ``{...}`` span. Raises with the raw content if no JSON object
        can be recovered — that is a real extraction failure, not an empty
        result, and the caller needs the raw text to debug it.
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
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Bonsai returned unparseable JSON: {content!r}") from e
            else:
                raise RuntimeError(f"Bonsai returned unparseable JSON: {content!r}") from None

        relations = data.get("relations", []) if isinstance(data, dict) else []
        # Defend against the model returning a bare list instead of the
        # documented {"relations": [...]} envelope.
        if isinstance(data, list):
            relations = data

        return [
            r for r in relations
            if isinstance(r, dict) and {"subject", "predicate", "object"} <= r.keys()
        ]