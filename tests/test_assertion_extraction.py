"""Deterministic state-assertion extractor -- Phase 4 (D1/D6).

Pure/unit: no IO, no model, no store. Covers the deterministic normalizer's
high-precision paths (field ``key: value`` / ``key = value`` / ``key is [now]
value`` / change-verb) and the cold-start no-op (a plain conversation yields
near-zero assertions -- Bonsai carries conversations; the deterministic path is
for structured doc fields). Also covers the Bonsai ``has_state`` merge: Bonsai
wins on overlap, deterministic fills when Bonsai returns none.
"""

from __future__ import annotations

from src.encoding.assertion_extractor import extract_state_assertions


# ── field style ──

def test_field_colon():
    """``Status: open`` -> (status, open)."""
    out = extract_state_assertions("Status: open\n")
    assert {"entity": "status", "value": "open"} in out


def test_field_equals():
    """``Database = Postgres`` -> (database, Postgres)."""
    out = extract_state_assertions("Database = Postgres\n")
    assert {"entity": "database", "value": "Postgres"} in out


def test_field_is_now_marks_update():
    """``The deployment target is now production`` -> the update shape.

    ``is now`` is exactly the contradiction-inducing pattern (a newer doc
    updating an older value); the entity normalizes to ``deployment target``.
    """
    out = extract_state_assertions("The deployment target is now production\n")
    assert {"entity": "deployment target", "value": "production"} in out


def test_field_is_basic():
    out = extract_state_assertions("Priority is high\n")
    assert {"entity": "priority", "value": "high"} in out


# ── change-verb ──

def test_change_verb_chose():
    """``The team chose Redis for caching`` -> (team, Redis)."""
    out = extract_state_assertions("The team chose Redis for caching\n")
    assert {"entity": "team", "value": "Redis"} in out


def test_change_verb_switched_to():
    out = extract_state_assertions("Acme switched to Postgres.\n")
    assert {"entity": "acme", "value": "Postgres"} in out


def test_change_verb_value_cut_at_stopword():
    """``chose Postgres for the JSONB support`` -> value ``Postgres`` only."""
    out = extract_state_assertions("The team chose Postgres for the JSONB support\n")
    assert {"entity": "team", "value": "Postgres"} in out


# ── cold-start no-op (D6) ──

def test_plain_conversation_yields_no_role_assertions():
    """``User:``/``Assistant:`` role prefixes are rejected -- a plain
    conversation yields NO deterministic assertions (Bonsai carries them).

    This is the cold-start byte-identical property: a corpus with no
    field-style state claims -> zero ``state`` edges -> detector never fires.
    """
    text = "User: I think we should use Postgres.\nAssistant: That sounds great."
    out = extract_state_assertions(text)
    # No assertion whose entity is a role prefix.
    assert not any(a["entity"] in ("user", "assistant") for a in out)


def test_prose_yields_no_spurious_fields():
    """Free prose (no field shape) -> empty (line-anchored precision)."""
    text = (
        "Postgres handles concurrent readers well. "
        "We discussed the migration plan over lunch."
    )
    out = extract_state_assertions(text)
    assert out == []


def test_reject_pronoun_starts():
    """``This is a test`` -> rejected (``this`` in reject set)."""
    out = extract_state_assertions("This is a test\n")
    assert not any(a["entity"] == "this" for a in out)


def test_reject_email_headers():
    """``From: alice@example.com`` -> rejected (``from`` is an email header)."""
    out = extract_state_assertions("From: alice@example.com\nTo: bob@example.com\n")
    assert not any(a["entity"] in ("from", "to") for a in out)


# ── Bonsai merge (D1) ──

def test_bonsai_has_state_lifted():
    """A Bonsai ``has_state`` relation is lifted into the assertion set."""
    out = extract_state_assertions(
        "", decisions=None,
        relations=[{"subject": "team", "predicate": "has_state",
                    "object": "MySQL"}],
    )
    assert {"entity": "team", "value": "MySQL"} in out


def test_bonsai_state_alias_lifted():
    """The ``state`` predicate alias is also lifted."""
    out = extract_state_assertions(
        "", relations=[{"subject": "E:db", "predicate": "state", "object": "v4"}],
    )
    assert {"entity": "db", "value": "v4"} in out


def test_bonsai_other_predicates_ignored():
    """``decides``/``explains`` relations are NOT state assertions."""
    out = extract_state_assertions(
        "", relations=[
            {"subject": "User", "predicate": "decides", "object": "Postgres"},
            {"subject": "Alice", "predicate": "explains", "object": "WaveDB"},
        ],
    )
    assert out == []


def test_dedup_entity_value_case_insensitive():
    """Deterministic + Bonsai agreeing on the same (entity, value) -> one."""
    out = extract_state_assertions(
        "Database: Postgres\n",
        relations=[{"subject": "database", "predicate": "has_state",
                    "object": "Postgres"}],
    )
    # Exactly one (database, Postgres) assertion.
    matches = [a for a in out if a["entity"] == "database"
               and a["value"].lower() == "postgres"]
    assert len(matches) == 1


def test_deterministic_fills_when_bonsai_none():
    """When Bonsai returns no ``has_state``, deterministic still yields its own."""
    out = extract_state_assertions(
        "Status: open\n",
        relations=[{"subject": "User", "predicate": "decides", "object": "X"}],
    )
    assert {"entity": "status", "value": "open"} in out


def test_url_value_rejected():
    """A value that is a bare URL is not a state assertion."""
    out = extract_state_assertions("Docs: https://example.com/spec\n")
    assert out == []


def test_decisions_scanned_as_fields():
    """A GLiNER decision span that is a field ("database: Postgres") yields
    an assertion via the ``decisions`` path."""
    out = extract_state_assertions("", decisions=["database: Postgres"])
    assert {"entity": "database", "value": "Postgres"} in out


def test_empty_input_returns_empty():
    assert extract_state_assertions("") == []
    assert extract_state_assertions(None) == []  # type: ignore[arg-type]


def test_article_stripped_from_key():
    """``The database: Postgres`` -> entity ``database`` (article stripped)."""
    out = extract_state_assertions("The database: Postgres\n")
    assert {"entity": "database", "value": "Postgres"} in out