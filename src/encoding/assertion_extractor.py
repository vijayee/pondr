"""Deterministic state-assertion extractor (Phase 3c, D1/D6).

A *state assertion* is an explicit ``entity -> value`` claim extracted from
text -- the production writer of ``(E:entity, state, literal)`` edges that
unblocks the dormant A2 ``contradictory_state`` anomaly resolver. The chat's
conflict-aware cognitive mode needs a *fact-level* signal: "Policy A said X,
Policy B updated it to Y" must become two ``(E:policy, state, X)`` /
``(E:policy, state, Y)`` edges so the detector sees two distinct live values
for one entity (``_detect_contradictory_state``, anomaly_rules.py).

This module is the **deterministic, no-model half** of the assertion path
(plan D1): explicit, zero-shot, free, and risk-free. It catches the
high-precision, contradiction-meaningful cases -- structured ``key: value`` /
``key = value`` / ``key is value`` / ``key is now value`` field lines (the
shape Jira/Linear/Confluence status fields, config snippets, and spec tables
take), plus explicit change-verb patterns (``chose X`` / ``switched to X`` /
``selected X``). The **paraphrase** half is Bonsai's job (``has_state``
relations, ``bonsai_relations.py``): a conflict the deterministic normalizer
cannot see is an *honest miss* here, not a failure to fix -- the
EnterpriseRAG-Bench eval (D8) sets its recall threshold to this deterministic
ceiling and documents it.

Why line-anchored: scanning the whole prose for ``is``/``are``/``was`` would
turn every sentence ("Postgres handles concurrent readers") into a spurious
``(handles, "Postgres concurrent readers")`` field. Field-style assertions are
*line-shaped* (one ``key: value`` per line), so matching per-line over
structured document text keeps precision high. Conversation ``full_text`` is
``"User: ...\\nAssistant: ..."`` -- only the role prefixes are line-anchored,
and they are rejected (see ``_REJECT_KEYS``), so a plain conversation yields
near-zero deterministic assertions (Bonsai carries conversations); a
structured doc yields its real fields. This is the cold-start no-op property
(D6): a corpus with no field-style state claims produces zero ``state`` edges
-> the detector never fires -> byte-identical to today.

Pure: no IO, no model, no store. Unit-testable in isolation. The encoder
merges these with Bonsai ``has_state`` relations (dedup; Bonsai wins on
overlap, deterministic fills when Bonsai returns none).
"""

from __future__ import annotations

import re
from typing import Iterable

__all__ = ["extract_state_assertions"]


# Keys that are structural / role / prose-prefix labels, not stable attribute
# entities. ``User:``/``Assistant:`` are the conversation role prefixes (must
# never become ``(E:user, state, ...)`` -- every conversation would contradict
# every other). Email headers (``from``/``to``/``date``/``subject``) vary per
# thread. Markdown prose prefixes (``note``/``example``/``warning``) are
# documentation, not facts. Rejecting them keeps the deterministic path honest.
_REJECT_KEYS = frozenset({
    # Conversation role prefixes (the encoder's ``full_text`` shape).
    "user", "assistant", "system", "developer", "tool",
    # Email headers (vary per thread; not stable attributes).
    "from", "to", "cc", "bcc", "date", "subject", "re", "fwd",
    # Prose / documentation prefixes (not facts).
    "note", "notes", "todo", "example", "examples", "warning", "caveat",
    "see", "see also", "source", "ref", "reference", "references",
    "figure", "table", "image", "caption", "listing", "code",
    # Pronoun sentence-starters (``this is``, ``it is``, ``there is``).
    "this", "it", "that", "there", "here", "he", "she", "they", "we", "i", "you",
    "these", "those", "all", "some", "most", "both", "each", "every",
    # Generic connectives that match ``key:`` too often.
    "and", "or", "but", "if", "then", "when", "while", "with", "without",
})

# Stopword boundaries that end a change-verb object ("chose Postgres FOR the
# JSONB support" -> value "Postgres"). Cutting here keeps the value to the
# object noun phrase rather than the rest of the clause.
_VALUE_STOPWORDS = frozenset({
    "for", "because", "since", "due", "as", "after", "before", "but",
    "though", "although", "while", "whereas", "so", "therefore", "thus",
    "instead", "rather", "in", "on", "at", "to", "from", "with", "by",
    "over", "under", "against", "than", "and", "or",
})

_ARTICLES = frozenset({"the", "a", "an"})

# A field key: 1-6 words of letters/digits/dash/underscore/space, starting with
# a letter. Capped at 6 words so prose ("one decision up front about the
# migration plan") is not read as a field.
_KEY = r"[A-Za-z][A-Za-z0-9 _\-]{0,60}?"

# ``key: value`` or ``key = value`` (field style). Anchored at line start;
# value is the rest of the line (non-empty). The colon form also matches
# definition lists; the reject set + key-length cap bound the noise.
_FIELD_KV_RE = re.compile(
    rf"^\s*({_KEY})\s*[:=]\s*(.+?)\s*$"
)

# ``key is/are/was [now] value`` (field style), line-anchored. The optional
# ``now`` marks an explicit update ("the deployment target is now production"),
# which is exactly the contradiction-inducing shape.
_FIELD_IS_RE = re.compile(
    rf"^\s*({_KEY})\s+(?:is|are|was)\s+(?:now\s+)?(.+?)\s*$"
)

# Change-verb patterns: ``<subject> chose/selected/adopted/switched to/moved
# to/is now using/now uses <value>``. Subject is captured (1-4 words) so the
# entity is the decider, not a generic "decision". Subject stability across
# documents is the honest ceiling: "We chose Postgres" vs "The team chose
# MySQL" -> different entities -> no deterministic contradiction (Bonsai's
# job). When subjects match, the contradiction IS caught.
_VERB_RE = re.compile(
    r"^(.{1,40}?)\s+"
    r"(?:chose|selected|adopted|picked|switched\s+to|moved\s+to|"
    r"is\s+now\s+using|now\s+uses|replaced\s+\S+\s+with)\s+"
    r"(.+?)$"
)


def _norm_key(raw: str) -> str:
    """Normalize a field key to a stable attribute entity id.

    Lowercase, collapse whitespace, strip leading articles + trailing
    punctuation. ``"The deployment target"`` -> ``"deployment target"``.
    Two documents writing the same attribute use the same entity string, so
    ``(E:deployment target, state, X)`` collides across docs and the detector
    can fire.
    """
    k = raw.strip().lower()
    k = re.sub(r"\s+", " ", k).strip(" .,:;!?\"'()[]")
    # Strip a leading ``e:`` entity-id prefix (a Bonsai ``has_state`` subject may
    # already be ``E:db``; the store writes ``E:{entity}`` so the assertion's
    # ``entity`` is the bare name).
    if k.startswith("e:"):
        k = k[2:].strip()
    # Strip a single leading article ("the database" -> "database").
    parts = k.split(" ")
    if len(parts) > 1 and parts[0] in _ARTICLES:
        parts = parts[1:]
    return " ".join(parts).strip()


def _trim_value(raw: str) -> str:
    """Normalize + trim a value to its object phrase.

    Strips wrapping quotes, trailing sentence punctuation, and cuts the
    change-verb object at the first stopword boundary ("Postgres for the
    JSONB support" -> "Postgres"). Field-style values (``key: value``) are
    not cut at stopwords -- the rest of the line IS the value.
    """
    v = raw.strip().strip("\"'").strip()
    v = v.rstrip(".,;:!?")
    return v


def _trim_verb_value(raw: str) -> str:
    """Cut a change-verb object at the first stopword boundary."""
    v = raw.strip().strip("\"'").strip()
    # Take tokens until a stopword; keep up to the first stopword boundary.
    out: list[str] = []
    for tok in v.split():
        clean = tok.rstrip(".,;:!?")
        if clean.lower() in _VALUE_STOPWORDS:
            break
        out.append(tok)
    # If everything was a stopword (degenerate), keep the first token.
    if not out and v:
        out.append(v.split()[0])
    return _trim_value(" ".join(out))


def _valid_key(k: str) -> bool:
    """A key is usable iff it is non-empty, not rejected, and <= 6 words."""
    if not k:
        return False
    if k in _REJECT_KEYS:
        return False
    # Reject keys that contain only digits / are too long (prose).
    if len(k.split(" ")) > 6:
        return False
    if not re.search(r"[A-Za-z]", k):
        return False
    return True


def _valid_value(v: str) -> bool:
    """A value is usable iff it is non-empty, not a bare URL, and bounded."""
    if not v:
        return False
    # URLs as values are not state assertions ("see http://...").
    if v.lower().startswith(("http://", "https://", "ftp://")):
        return False
    if len(v) > 120:
        return False
    return True


def _add(out: list[dict], seen: set, entity: str, value: str) -> None:
    """Append a deduped assertion (entity lowercased-key, value trimmed)."""
    e = _norm_key(entity)
    if not _valid_key(e):
        return
    v = value.strip()
    if not _valid_value(v):
        return
    key = (e, v.lower())
    if key in seen:
        return
    seen.add(key)
    out.append({"entity": e, "value": v})


def _scan_text(text: str, out: list[dict], seen: set) -> None:
    """Line-anchored scan of ``text`` for field + verb state assertions."""
    for line in text.splitlines():
        if not line.strip():
            continue
        # Field key:value / key=value (highest precision; try first).
        m = _FIELD_KV_RE.match(line)
        if m:
            _add(out, seen, m.group(1), _trim_value(m.group(2)))
            continue  # a line is one assertion (avoid double-counting)
        # Field "key is/are/was [now] value".
        m = _FIELD_IS_RE.match(line)
        if m:
            _add(out, seen, m.group(1), _trim_value(m.group(2)))
            continue
        # Change-verb "subject chose/switched to/... value".
        m = _VERB_RE.match(line)
        if m:
            _add(out, seen, m.group(1), _trim_verb_value(m.group(2)))


def extract_state_assertions(
    text: str,
    decisions: "Iterable[str] | None" = None,
    relations: "Iterable[dict] | None" = None,
) -> list[dict]:
    """Extract explicit ``entity -> value`` state assertions, deterministically.

    Scans ``text`` line-by-line for field-style assertions (``key: value``,
    ``key = value``, ``key is [now] value``) and change-verb patterns
    (``subject chose/switched to/selected X``). Each ``decision`` string
    (a GLiNER-extracted decision span) is scanned the same way, so a
    decision span that is itself a field ("database: Postgres") yields an
    assertion while a bare "chose Postgres" yields none from this path
    (Bonsai's ``has_state`` carries the subject-aware case via ``relations``).

    ``relations`` is the Bonsai relation list; ``has_state`` (and the alias
    ``state``) triples are lifted here so the deterministic + Bonsai halves
    are one union (Bonsai subject+value, no regex). Other relation predicates
    are ignored. Dedup is by ``(entity, value)`` (case-insensitive); Bonsai
    and deterministic agree on overlap -> one assertion.

    Returns ``[{"entity": "<normalized key>", "value": "<value>"}, ...]`` in
    first-seen order. Pure: no IO, no model, no store -- unit-testable in
    isolation. Empty for text with no explicit state claims (cold-start no-op).
    """
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    if text:
        _scan_text(text, out, seen)
    if decisions:
        for d in decisions:
            if isinstance(d, str) and d.strip():
                _scan_text(d, out, seen)
    if relations:
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            pred = str(rel.get("predicate", "")).lower().strip()
            if pred not in ("has_state", "state"):
                continue
            subj = rel.get("subject")
            obj = rel.get("object")
            if isinstance(subj, str) and isinstance(obj, str):
                _add(out, seen, subj, obj)

    return out