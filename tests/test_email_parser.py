"""Tests for the email parser with thread/conversation reconstruction.

Email is the one source format where the structure IS a conversation; the parser
reconstructs the reply tree (``In-Reply-To``/``References``) and emits one
``RawSection`` per message in thread-DFS order. Zero deps (stdlib ``email``),
so NO ``importorskip`` -- these tests run dep-free.
"""

from __future__ import annotations

import email
import os

from src.ingestion.email_parser import EmailParser
from src.ingestion.parsers import detect_type, get_parser


_ROOT = (
    "Message-ID: <a@host>\n"
    "From: alice@host\n"
    "To: bob@host\n"
    "Subject: Re: Re: Lunch next week\n"
    "Date: Tue, 01 Jul 2025 10:00:00 +0000\n"
    "\n"
    "Lets do Tuesday for lunch.\n"
)
_REPLY1 = (
    "Message-ID: <b@host>\n"
    "From: bob@host\n"
    "To: alice@host\n"
    "Subject: Re: Lunch next week\n"
    "Date: Tue, 01 Jul 2025 11:00:00 +0000\n"
    "In-Reply-To: <a@host>\n"
    "\n"
    "Tuesday works for me.\n"
)
_REPLY2 = (
    "Message-ID: <c@host>\n"
    "From: carol@host\n"
    "To: alice@host\n"
    "Subject: Re: Lunch next week\n"
    "Date: Tue, 01 Jul 2025 12:00:00 +0000\n"
    "In-Reply-To: <a@host>\n"
    "\n"
    "Can I join too?\n"
)
_HTML_ONLY = (
    "Message-ID: <h@host>\n"
    "From: dave@host\n"
    "Subject: Announcement\n"
    "Date: Wed, 02 Jul 2025 09:00:00 +0000\n"
    "Content-Type: text/html; charset=utf-8\n"
    "\n"
    "<html><body><p>Hello <b>world</b></p><p>Second line</p></body></html>\n"
)


def _msgs(*texts):
    return [email.message_from_string(t) for t in texts]


def test_detect_type_and_registry():
    assert detect_type("thread.eml") == "email"
    assert detect_type("folder/some.eml") == "email"
    # get_parser returns an EmailParser instance.
    p = get_parser("email")
    assert isinstance(p, EmailParser)
    assert p.source_type == "email"


def test_thread_reconstruction_order_and_levels():
    # Pass messages out of chronological + out of reply order; the parser must
    # reconstruct root-first, replies in chronological order under their parent.
    pd = EmailParser().parse_messages(
        _msgs(_REPLY2, _ROOT, _REPLY1), source_path="/tmp/thread"
    )
    assert len(pd.sections) == 3
    levels = [s.level for s in pd.sections]
    assert levels == [1, 2, 2], f"expected root(1) + two replies(2,2), got {levels}"
    # Root body first (the message with no In-Reply-To in the set).
    assert "Tuesday for lunch" in pd.sections[0].content
    # Replies in chronological order: b@host (11:00) before c@host (12:00).
    senders_in_order = [s.heading.split(" (")[0] for s in pd.sections]
    assert senders_in_order == ["alice@host", "bob@host", "carol@host"], senders_in_order


def test_authors_and_created_at_and_metadata():
    pd = EmailParser().parse_messages(
        _msgs(_ROOT, _REPLY1, _REPLY2), source_path="/tmp/thread"
    )
    # Distinct senders across the thread.
    assert set(pd.authors) == {"alice@host", "bob@host", "carol@host"}
    # created_at is the root message's Date.
    assert pd.created_at == "Tue, 01 Jul 2025 10:00:00 +0000"
    # metadata carries the raw thread graph.
    assert pd.metadata["message_ids"] == ["a@host", "b@host", "c@host"]
    assert pd.metadata["in_reply_to"] == {"b@host": "a@host", "c@host": "a@host"}


def test_title_collapses_re_prefix_run():
    # The root subject "Re: Re: Lunch next week" collapses to a single "Re:".
    pd = EmailParser().parse_messages(_msgs(_ROOT), source_path="/tmp/x.eml")
    assert pd.title == "Re: Lunch next week", pd.title


def test_single_eml_is_one_message_thread():
    pd = EmailParser().parse_text(_ROOT, source_path="/tmp/one.eml")
    assert len(pd.sections) == 1
    assert pd.sections[0].level == 1
    assert "Tuesday for lunch" in pd.sections[0].content


def test_directory_of_emls(tmp_path):
    # A directory of .eml files is the thread; write them out and parse the dir.
    d = tmp_path / "thread"
    d.mkdir()
    (d / "0.eml").write_text(_ROOT, encoding="utf-8")
    (d / "1.eml").write_text(_REPLY1, encoding="utf-8")
    (d / "2.eml").write_text(_REPLY2, encoding="utf-8")
    pd = EmailParser().parse(str(d))
    assert len(pd.sections) == 3
    assert [s.level for s in pd.sections] == [1, 2, 2]


def test_html_only_body_extracted():
    pd = EmailParser().parse_text(_HTML_ONLY, source_path="/tmp/h.eml")
    assert len(pd.sections) == 1
    body = pd.sections[0].content
    assert "Hello world" in body
    assert "Second line" in body
    assert "<" not in body  # tags stripped


def test_missing_headers_chronological_fallback():
    # No In-Reply-To/References -> chronological by Date; all root at level 1.
    a = (
        "Message-ID: <x@h>\nFrom: a@h\nSubject: s\n"
        "Date: Tue, 01 Jul 2025 08:00:00 +0000\n\nfirst\n"
    )
    b = (
        "Message-ID: <y@h>\nFrom: b@h\nSubject: s\n"
        "Date: Tue, 01 Jul 2025 09:00:00 +0000\n\nsecond\n"
    )
    pd = EmailParser().parse_messages(_msgs(b, a), source_path="/tmp/n")
    assert [s.level for s in pd.sections] == [1, 1]
    # Chronological: a (08:00) before b (09:00).
    assert "first" in pd.sections[0].content
    assert "second" in pd.sections[1].content


def test_orphan_reply_roots_at_level_one():
    # In-Reply-To points at a Message-ID NOT in the set -> roots at level 1.
    orphan = (
        "Message-ID: <o@h>\nFrom: z@h\nSubject: s\n"
        "Date: Tue, 01 Jul 2025 10:00:00 +0000\n"
        "In-Reply-To: <missing@h>\n\nreply to a parent not in the folder\n"
    )
    pd = EmailParser().parse_messages(_msgs(orphan), source_path="/tmp/o")
    assert len(pd.sections) == 1
    assert pd.sections[0].level == 1
    # The raw edge is still recorded (provenance) even though the parent isn't
    # in the set -- the message just roots the section tree at level 1.
    assert pd.metadata["in_reply_to"] == {"o@h": "missing@h"}


def test_non_ascii_sender_is_ascii_sanitized():
    m = (
        "Message-ID: <u@h>\nFrom: =?utf-8?Q?Ren=C3=A9?= <r@h>\n"
        "Subject: Hi\nDate: Tue, 01 Jul 2025 10:00:00 +0000\n\nbody\n"
    )
    pd = EmailParser().parse_text(m, source_path="/tmp/u.eml")
    # ASCII-sanitized: the non-ASCII byte becomes '?', never a cp1252 crash.
    assert pd.sections[0].heading.startswith("Ren")
    assert pd.sections[0].heading.encode("ascii")  # must be ASCII-safe


def test_non_ascii_subject_title_is_ascii_sanitized():
    # A non-ASCII Subject reaches the thread TITLE; it must be ASCII-sanitized
    # too (a non-ASCII title would crash a cp1252 console / argparse help).
    m = (
        "Message-ID: <s@h>\nFrom: a@h\n"
        "Subject: =?utf-8?Q?R=C3=A9union?=\n"
        "Date: Tue, 01 Jul 2025 10:00:00 +0000\n\nbody\n"
    )
    pd = EmailParser().parse_text(m, source_path="/tmp/s.eml")
    assert pd.title.encode("ascii")  # ASCII-safe (the accent -> '?')
    assert "R" in pd.title


def test_empty_input_no_crash():
    pd = EmailParser().parse_messages([], source_path="/tmp/empty")
    assert pd.sections == []


def test_references_used_when_no_in_reply_to():
    # No In-Reply-To, but References chain -> the last id is the parent.
    m = (
        "Message-ID: <r2@h>\nFrom: q@h\nSubject: s\n"
        "Date: Tue, 01 Jul 2025 12:00:00 +0000\n"
        "References: <r0@h> <r1@h>\n\nreply\n"
    )
    root = (
        "Message-ID: <r1@h>\nFrom: p@h\nSubject: s\n"
        "Date: Tue, 01 Jul 2025 10:00:00 +0000\n\nroot\n"
    )
    pd = EmailParser().parse_messages(_msgs(m, root), source_path="/tmp/ref")
    # r1 is the parent (last in References); root message r1 is level 1, r2 is 2.
    levels = {s.content: s.level for s in pd.sections}
    assert levels["root"] == 1
    assert levels["reply"] == 2