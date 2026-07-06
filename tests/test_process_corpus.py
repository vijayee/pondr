"""Offline tests for scripts/download_corpora.py + process_corpus.py helpers.

The full ingestion pipeline (GLiNER + Bonsai → WaveDB) runs on the pod; these
tests cover the locally-testable surface: the corpus parsers (DialogSum /
SAMSum → [user, assistant] turn pairs) and the resume checkpoint helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``scripts.*`` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_corpora import parse_dialogsum, parse_samsum  # noqa: E402
from scripts.process_corpus import _load_checkpoint, _save_checkpoint  # noqa: E402


# ── parsers ──


def test_parse_dialogsum_pairs_speakers():
    d = "#Person1#: I need a database.#Person2#: Try HBTrie.#Person1#: How does it compare?#Person2#: It is a B+tree of B+trees."
    assert parse_dialogsum(d) == [
        ["I need a database.", "Try HBTrie."],
        ["How does it compare?", "It is a B+tree of B+trees."],
    ]


def test_parse_dialogsum_merges_consecutive_same_speaker():
    d = "#Person1#: Wait.#Person1#: Let me think.#Person2#: OK."
    # Two Person1 utterances merge into one before pairing with Person2.
    assert parse_dialogsum(d) == [["Wait. Let me think.", "OK."]]


def test_parse_dialogsum_trailing_unpaired():
    d = "#Person1#: hello#Person2#: hi#Person1#: one more"
    assert parse_dialogsum(d) == [["hello", "hi"], ["one more", ""]]


def test_parse_dialogsum_empty():
    assert parse_dialogsum("") == []
    assert parse_dialogsum("no markers here") == []


def test_parse_samsum_line_per_utterance():
    s = "Alex: hey\nBob: yeah\nAlex: last play was insane"
    assert parse_samsum(s) == [
        ["hey", "yeah"],
        ["last play was insane", ""],
    ]


def test_parse_samsum_merges_consecutive_same_speaker():
    s = "Alex: a\nAlex: b\nBob: c"
    assert parse_samsum(s) == [["a b", "c"]]


def test_parse_samsum_continuation_line_joins_previous():
    # A line with no speaker prefix continues the previous utterance.
    s = "Alex: hey there\nhow are you\nBob: good"
    assert parse_samsum(s) == [["hey there how are you", "good"]]


def test_parse_samsum_blank_lines_ignored():
    s = "Alex: a\n\n\nBob: b"
    assert parse_samsum(s) == [["a", "b"]]


# ── checkpoint helpers ──


def test_checkpoint_round_trip(tmp_path):
    db = str(tmp_path / "db")
    Path(db).mkdir()
    cp = {"processed_ids": ["conv_001", "conv_002"], "episodes": 7, "failures": ["x"]}
    _save_checkpoint(db, cp)
    loaded = _load_checkpoint(db)
    assert loaded["processed_ids"] == ["conv_001", "conv_002"]
    assert loaded["episodes"] == 7
    assert loaded["failures"] == ["x"]


def test_load_checkpoint_missing_returns_empty(tmp_path):
    db = str(tmp_path / "db")
    Path(db).mkdir()
    cp = _load_checkpoint(db)
    assert cp["processed_ids"] == []
    assert cp["episodes"] == 0
    assert cp["failures"] == []


def test_load_checkpoint_corrupt_falls_back(tmp_path, capsys):
    db = str(tmp_path / "db")
    Path(db).mkdir()
    (Path(db) / ".checkpoint.json").write_text("{not valid json", encoding="utf-8")
    cp = _load_checkpoint(db)
    assert cp["processed_ids"] == []  # fresh fallback, no crash
    assert "unreadable" in capsys.readouterr().err


def test_save_checkpoint_is_atomic_overwrite(tmp_path):
    db = str(tmp_path / "db")
    Path(db).mkdir()
    _save_checkpoint(db, {"processed_ids": ["a"], "episodes": 1, "failures": []})
    _save_checkpoint(db, {"processed_ids": ["a", "b"], "episodes": 2, "failures": []})
    # No stale .tmp left behind; final state reflects the second write.
    assert not (Path(db) / ".checkpoint.json.tmp").exists()
    cp = json.loads((Path(db) / ".checkpoint.json").read_text(encoding="utf-8"))
    assert cp["processed_ids"] == ["a", "b"]
    assert cp["episodes"] == 2