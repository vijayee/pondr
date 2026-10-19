"""Write-time intent keying -- judge-built keys stored beside the memory unit.

WHY (addenda 15/16, docs/ICM_intent_store_spec.md sec 2-4): the measured route
is judge-built intent keys at INDEX-BUILD time, with deterministic, local work
at query time. This module is the build-time half: one judge reading per intent
that yields the requirement LIST (the measured 4.63 requirements per real
issue, not one summary triple) plus the intent's epoch reference.

PRE-REGISTERED BEHAVIOUR (written before implementation; no fitting on runs):
  (1) CACHE BY TEXT HASH: the cache key is
      sha1(prompt_version | declared_qualifier | normalised intent text), so
      re-writing the same intent costs zero calls. One call per unique intent.
  (2) A key always carries `requirements` (a LIST; 3-5 expected, fewer is
      legitimate and is never padded) and `epoch_ref` {kind, value}.
  (3) THE JUDGE NEVER BLOCKS A WRITE. Four statuses are returned, always:
        judged    -- fresh valid answer
        cached    -- cache hit (no call made)
        abstained -- valid answer that carries no requirement and no epoch
        error     -- empty/unparseable transport result; key is None
      A caller records the status; nothing is silently dropped.
  (4) Keys are CANONICALISATIONS, not labels. The caller declares the intent
      TEXT (and may declare a qualifier); the judge decomposes it. A declared
      qualifier is stored separately as `declared_qualifier` and never
      overwrites the judge's own reading of the qualifier.
  (5) Opposition is pairwise by construction (addendum 16's SAME / DIFFERENT /
      OPPOSED prompt). It is available through `relation_between()` and is
      cached independently of per-intent keys. It is NOT run implicitly on
      every write: the cost per write / epochs per memory question is open in
      the spec, so the caller chooses when to pay it.

FAILURE DISCIPLINE: an empty response or an unparseable one is `error`, never a
placeholder key. The query path's stated fallback is dense@1; this module gives
the caller what it needs to detect that case.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

KEY_VERSION = "icm-intent-key/1"
PROMPT_VERSION = "intent-key/1"
DEFAULT_MODEL = "deepseek-v4.1-flash:cloud"
DEFAULT_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_CACHE = os.path.join("data", "_scratch", "icm", "intent_keys_cache.jsonl")
ORDINALS = ("EARLIEST", "MIDDLE", "LATEST")
# judge synonyms mapped onto the fixed ordinal vocabulary (the prompt asks for
# the three words, but models paraphrase; mapping is deterministic, not fitted)
ORDINAL_SYNONYMS = {
    "EARLIEST": ("EARLIEST", "FIRST", "INITIAL", "OLDEST", "ORIGINAL"),
    "MIDDLE": ("MIDDLE", "SECOND", "INTERMEDIATE", "INTERIM"),
    "LATEST": ("LATEST", "LAST", "NEWEST", "MOST RECENT", "RECENT", "FINAL"),
}
REQUIREMENT_FIELDS = ("subject", "attitude", "relation", "object", "qualifier")

EXTRACT_PROMPT = """Read the intent statement below and return a JSON object only.

Rules:
- requirements: a LIST of 3 to 5 triples that TOGETHER capture everything the
  intent requires of the memory/artifact. Each triple:
  {"subject": "<who/what the requirement is about>",
   "attitude": "<stance verb, e.g. report/verify/update/avoid>",
   "relation": "<what is to be done or true>",
   "object": "<the thing acted on, few words>"}
  Do not collapse a multi-part intent into one summary triple. If the text
  genuinely warrants fewer than 3, return fewer; never pad.
- qualifier: the state/version marker the intent itself carries (e.g.
  archived, current, before migration), or "" if it carries none.
- epoch_ref: the time/run reference the intent uses, as
  {"kind": "ordinal"|"absolute"|"none", "value": "..."}.
  ordinal value must be one of EARLIEST, MIDDLE, LATEST (relative references:
  first/earliest/initial, middle, latest/most recent/newest).
  absolute value is the stated date, run number or version.
  Use kind "none" when the intent makes no temporal reference.

Reply with exactly: {"requirements": [...], "qualifier": "...",
"epoch_ref": {"kind": "...", "value": "..."}}"""

RELATION_PROMPT = """Compare the two intents below, both about the same artifact.
Do they require the SAME thing of it, DIFFERENT things, or OPPOSED things
(acting on one would defeat the other)? Answer exactly one word:
SAME, DIFFERENT or OPPOSED.

A: {a}

B: {b}"""


@dataclass(frozen=True)
class KeyResult:
    """Outcome of a write-time key lookup."""
    key: dict | None
    status: str  # judged | cached | abstained | error

    @property
    def ok(self) -> bool:
        return self.key is not None and self.status in ("judged", "cached")


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def _hash(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _first_json(raw: str, opener: str):
    """First JSON object/array in a model response, or None."""
    if not raw:
        return None
    start = raw.find(opener)
    if start < 0:
        return None
    close = "]" if opener == "[" else "}"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == close:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except Exception:
                    return None
    return None


def _norm_epoch_ref(raw) -> dict:
    """Normalise whatever the judge returned into {kind, value}."""
    if isinstance(raw, str):
        raw = {"kind": "ordinal", "value": raw}
    if not isinstance(raw, dict):
        return {"kind": "none", "value": ""}
    kind = str(raw.get("kind", "")).strip().lower()
    value = str(raw.get("value", "")).strip()
    if kind == "ordinal":
        up = value.upper()
        for word in ORDINALS:
            if any(syn in up for syn in ORDINAL_SYNONYMS[word]):
                return {"kind": "ordinal", "value": word}
        return {"kind": "none", "value": ""}
    if kind == "absolute" and value:
        return {"kind": "absolute", "value": value}
    return {"kind": "none", "value": ""}


def _norm_requirements(raw) -> list[dict]:
    """Keep only well-formed triples; a single dict is accepted as one item."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        relation = str(item.get("relation", "")).strip()
        obj = str(item.get("object", "")).strip()
        if not relation or not obj:
            continue
        out.append({
            "subject": str(item.get("subject", "")).strip() or "SELF",
            "attitude": str(item.get("attitude", "")).strip(),
            "relation": relation,
            "object": obj,
            "qualifier": str(item.get("qualifier", "")).strip(),
        })
    return out


def parse_key(raw: str, model: str, text_hash: str,
              declared_qualifier: str = "") -> dict | None:
    """Parse a judge response into a key record. None means unparseable."""
    obj = _first_json(raw, "{")
    if not isinstance(obj, dict):
        return None
    requirements = _norm_requirements(obj.get("requirements"))
    epoch_ref = _norm_epoch_ref(obj.get("epoch_ref"))
    qualifier = str(obj.get("qualifier", "")).strip()
    return {
        "key_version": KEY_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "text_hash": text_hash,
        "requirements": requirements,
        "requirement_count": len(requirements),
        "qualifier": qualifier,
        "declared_qualifier": str(declared_qualifier or "").strip(),
        "epoch_ref": epoch_ref,
        "abstained": not requirements and epoch_ref["kind"] == "none",
    }


def ollama_ask(prompt: str, model: str = DEFAULT_MODEL,
               endpoint: str = DEFAULT_ENDPOINT, timeout: int = 240,
               tries: int = 4) -> str:
    """Default judge transport (localhost ollama; cloud relay allowed)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0.0, "num_ctx": 8192}}
    backoff = 8.0
    for _ in range(tries):
        try:
            req = urllib.request.Request(
                endpoint, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            raw = urllib.request.urlopen(req, timeout=timeout).read()
            return str(json.loads(raw)["message"]["content"]).strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(backoff, 60))
                backoff *= 1.8
                continue
            time.sleep(3.0)
        except Exception:
            time.sleep(3.0)
    return ""


class IntentKeyer:
    """Judge-backed key extraction with a persistent hash cache.

    `ask` is injectable so tests never touch the network. Cache writes append
    one JSON line per answer; a corrupt line is skipped on load (the cache is
    an optimisation, never the source of truth).
    """

    def __init__(self, cache_path: str | None = DEFAULT_CACHE,
                 ask=None, model: str = DEFAULT_MODEL,
                 endpoint: str = DEFAULT_ENDPOINT):
        self.model = model
        self._ask = ask or (lambda p: ollama_ask(p, model=model, endpoint=endpoint))
        self.cache_path = cache_path
        self._cache: dict[str, object] = {}
        self._load()

    # -- cache ---------------------------------------------------------
    def _load(self) -> None:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        for line in io.open(self.cache_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                self._cache[row["hash"]] = row.get("val")
            except Exception:
                continue

    def _save(self, key_hash: str, val) -> None:
        self._cache[key_hash] = val
        if not self.cache_path:
            return
        parent = os.path.dirname(self.cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with io.open(self.cache_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"hash": key_hash, "val": val},
                               ensure_ascii=True, sort_keys=True) + "\n")

    # -- keys ----------------------------------------------------------
    def key_for(self, text: str, declared_qualifier: str = "") -> KeyResult:
        """One judge call per unique intent text; cache hit is free."""
        text = _norm(text)
        if not text:
            return KeyResult(None, "error")
        key_hash = _hash("KEY", PROMPT_VERSION, self.model,
                         str(declared_qualifier or ""), text)
        if key_hash in self._cache:
            val = self._cache[key_hash]
            return KeyResult(val if isinstance(val, dict) else None, "cached")
        raw = self._ask(EXTRACT_PROMPT + "\n\nIntent:\n" + text[:4000])
        key = parse_key(raw, self.model, key_hash, declared_qualifier) if raw else None
        if key is None:
            return KeyResult(None, "error")
        self._save(key_hash, key)
        return KeyResult(key, "abstained" if key["abstained"] else "judged")

    # -- pairwise opposition -------------------------------------------
    def relation_between(self, text_a: str, text_b: str) -> str | None:
        """SAME / DIFFERENT / OPPOSED between two intents; None on failure.

        Deliberately explicit: the caller decides when the pairwise cost is
        worth paying (the spec's open cost question).
        """
        a, b = _norm(text_a), _norm(text_b)
        if not a or not b:
            return None
        rel_hash = _hash("REL", PROMPT_VERSION, self.model, a, b)
        if rel_hash in self._cache:
            val = self._cache[rel_hash]
            return val if val in ("same", "different", "opposed") else None
        raw = self._ask(RELATION_PROMPT.format(a=a[:900], b=b[:900])).upper()
        for word in ("OPPOSED", "DIFFERENT", "SAME"):
            if word in raw:
                label = word.lower()
                self._save(rel_hash, label)
                return label
        return None
