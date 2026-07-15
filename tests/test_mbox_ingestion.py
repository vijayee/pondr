"""Phase 3c (D9): ``.mbox`` single-file thread parsing.

``mailbox.mbox`` yields stdlib ``email.message.Message`` objects, so the email
thread-reconstruction core (``EmailParser.parse_messages``) needs NO changes --
only two wiring touches (``_TYPE_BY_EXT`` routes ``.mbox`` -> ``"email"``;
``EmailParser.parse`` feeds ``list(mailbox.mbox(path))`` into the core). These
tests write a real ``.mbox`` on disk (via stdlib ``mailbox``) and parse it end-
to-end, including the ``in_reply_to``/``references`` graph edges through
``encode_document`` (D9 + D5 together). No network.
"""

from __future__ import annotations

import mailbox
import os

from src.ingestion.email_parser import EmailParser
from src.ingestion.parsers import detect_type, get_parser
from src.memory.document import Document
from src.memory.store import HippocampalStore


def _msg(mid, frm, subject, date, body, in_reply_to=None, references=None):
    """Build a stdlib ``email.message.Message`` with the given headers."""
    m = mailbox.mboxMessage()
    m["Message-ID"] = f"<{mid}>"
    m["From"] = frm
    m["Subject"] = subject
    m["Date"] = date
    if in_reply_to:
        m["In-Reply-To"] = in_reply_to
    if references:
        m["References"] = references
    m.set_payload(body)
    return m


def _write_mbox(path, messages):
    """Write a real ``.mbox`` file from a list of ``Message`` objects."""
    mb = mailbox.mbox(str(path))
    for m in messages:
        mb.add(m)
    mb.flush()
    mb.close()


# ── routing ──

def test_mbox_ext_routes_to_email_parser():
    """``_TYPE_BY_EXT`` maps ``.mbox`` -> ``"email"`` (not PlainTextParser)."""
    assert detect_type("thread.mbox") == "email"
    assert detect_type("ARCHIVE.MBOX") == "email"  # case-insensitive ext
    assert isinstance(get_parser("email"), EmailParser)


# ── thread reconstruction from a .mbox ──

def test_mbox_three_message_thread(tmp_path):
    """A 3-message .mbox (root + reply + reply-to-reply) parses to 3 sections
    in DFS order with correct levels; metadata carries the reply edges."""
    path = tmp_path / "thread.mbox"
    msgs = [
        _msg("a@h", "alice@h", "Lunch", "Tue, 01 Jul 2025 10:00:00 +0000",
             "Lets do Tuesday for lunch.\n"),
        _msg("b@h", "bob@h", "Re: Lunch", "Tue, 01 Jul 2025 11:00:00 +0000",
             "Tuesday works for me.\n",
             in_reply_to="<a@h>", references="<a@h>"),
        _msg("c@h", "carol@h", "Re: Lunch", "Tue, 01 Jul 2025 12:00:00 +0000",
             "Can I join too?\n",
             in_reply_to="<b@h>", references="<a@h> <b@h>"),
    ]
    _write_mbox(path, msgs)

    pd = EmailParser().parse(str(path))
    assert len(pd.sections) == 3
    # Root (1) -> reply (2) -> reply-to-reply (3).
    assert [s.level for s in pd.sections] == [1, 2, 3], \
        [s.level for s in pd.sections]
    # DFS order: root, then reply, then reply-to-reply.
    assert "Tuesday for lunch" in pd.sections[0].content
    assert "Tuesday works for me" in pd.sections[1].content
    assert "Can I join too" in pd.sections[2].content
    # metadata: message_ids stripped of brackets, in_reply_to edges present.
    assert pd.metadata["message_ids"] == ["a@h", "b@h", "c@h"]
    assert pd.metadata["in_reply_to"] == {"b@h": "a@h", "c@h": "b@h"}


def test_mbox_parse_matches_parse_messages(tmp_path):
    """``.parse`` on a .mbox produces the SAME result as feeding the materialized
    message list directly into ``parse_messages`` (the core is unchanged)."""
    path = tmp_path / "t.mbox"
    msgs = [
        _msg("a@h", "alice@h", "X", "Tue, 01 Jul 2025 10:00:00 +0000", "root\n"),
        _msg("b@h", "bob@h", "Re: X", "Tue, 01 Jul 2025 11:00:00 +0000",
             "reply\n", in_reply_to="<a@h>"),
    ]
    _write_mbox(path, msgs)

    via_file = EmailParser().parse(str(path))
    via_core = EmailParser().parse_messages(list(mailbox.mbox(str(path))),
                                             source_path=str(path))
    assert [s.level for s in via_file.sections] == \
        [s.level for s in via_core.sections]
    assert via_file.metadata["in_reply_to"] == via_core.metadata["in_reply_to"]


# ── end-to-end: .mbox -> encode_document -> in_reply_to/references edges ──

def test_mbox_end_to_end_provenance_edges(tmp_path):
    """A .mbox thread ingested through the pipeline emits ``in_reply_to`` +
    ``references`` graph edges (D9 + D5 together)."""
    # Build the .mbox.
    mbox_path = tmp_path / "thread.mbox"
    msgs = [
        _msg("a@h", "alice@h", "Lunch", "Tue, 01 Jul 2025 10:00:00 +0000",
             "root body here\n"),
        _msg("b@h", "bob@h", "Re: Lunch", "Tue, 01 Jul 2025 11:00:00 +0000",
             "reply body here\n",
             in_reply_to="<a@h>", references="<a@h>"),
    ]
    _write_mbox(mbox_path, msgs)

    # Parse (the routing + .mbox wiring).
    pd = EmailParser().parse(str(mbox_path))
    assert pd.metadata.get("message_ids")

    # Encode into the store directly (mirrors the pipeline's encode_document).
    store = HippocampalStore(str(tmp_path / "db"))
    try:
        from src.memory.document import DocumentSection
        secs = []
        for i, s in enumerate(pd.sections):
            secs.append(DocumentSection(
                id=f"doc_000001_sec_{i:03d}", heading=s.heading,
                level=s.level, content=s.content,
            ))
        doc = Document(
            id="doc_000001", source_type="email", source_path=str(mbox_path),
            title=pd.title or "thread", ingested_at="2026-07-15T00:00:00",
            sections=secs, metadata=pd.metadata,
        )
        store.encode_document(doc)

        # reply section (sec_001) -> in_reply_to -> root section (sec_000).
        r = store.graph.query().vertex("doc_000001_sec_001").out(
            "in_reply_to").execute_sync()
        try:
            targets = sorted(r.vertices)
        finally:
            r.close()
        assert targets == ["doc_000001_sec_000"], targets

        r = store.graph.query().vertex("doc_000001_sec_001").out(
            "references").execute_sync()
        try:
            targets = sorted(r.vertices)
        finally:
            r.close()
        assert targets == ["doc_000001_sec_000"], targets
    finally:
        store.close()


# ── edge cases ──

def test_empty_mbox_no_crash(tmp_path):
    """An empty .mbox -> an empty ParsedDocument (no crash)."""
    path = tmp_path / "empty.mbox"
    _write_mbox(path, [])
    pd = EmailParser().parse(str(path))
    assert pd.sections == []


def test_single_message_mbox(tmp_path):
    """A one-message .mbox -> a one-section thread at level 1."""
    path = tmp_path / "one.mbox"
    _write_mbox(path, [
        _msg("a@h", "alice@h", "Solo", "Tue, 01 Jul 2025 10:00:00 +0000",
             "just one message\n"),
    ])
    pd = EmailParser().parse(str(path))
    assert len(pd.sections) == 1
    assert pd.sections[0].level == 1