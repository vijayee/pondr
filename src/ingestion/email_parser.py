"""Email parser -- stdlib ``email`` -> ParsedDocument with thread reconstruction.

Email is the one source format where the structure IS a conversation: replies
chain via the ``In-Reply-To`` / ``References`` headers. So this parser does not
dump one section per header -- it RECONSTRUCTS the thread and emits one
``RawSection`` per message in thread order (root-first, replies under their
parent in chronological order -- a flattened DFS of the reply tree, the order a
reader reconstructs the conversation). ``level`` records the reply depth (root
message = 1, a direct reply = 2, ...) so the chunker's parent wiring recovers
the tree the parser saw.

A ``.eml`` is ONE message; the common ingest case is a DIRECTORY of ``.eml``
files (a mailbox export, a saved folder of replies) -- the parser treats the
directory as a thread. A single ``.eml`` is a one-message thread.

Zero external deps (stdlib ``email`` parses RFC 822/MIME), so -- unlike the
PDF/DOCX/web/code parsers -- this parser has NO ``RuntimeError`` dep path and
NO ``importorskip``; its tests run dep-free. ``parse_messages(messages,
source_path)`` is the no-IO mirror (takes a list of ``email.message.Message``
objects); ``parse_text(eml_text, source_path)`` is the single-message mirror.

ASCII-only constraint (cp1252 crashes on non-ASCII in print/argparse): ``From``
/``Subject`` strings are ASCII-sanitized (``.encode("ascii","replace").decode``)
before they reach ``heading`` / ``title`` so a non-ASCII sender never breaks
the store. The message BODY is kept verbatim (UTF-8 decoded with errors replaced)
-- it is content, not a graph-key component.
"""

from __future__ import annotations

import email
import email.header
import email.utils
import html as _html
import os
import re
from email.message import Message
from typing import Optional

from .parsers import ParsedDocument, RawSection


def _ascii(s: str) -> str:
    """ASCII-sanitize a header string (graph-key-safe component)."""
    if not s:
        return ""
    return s.encode("ascii", "replace").decode("ascii")


def _strip_angle(msgid: Optional[str]) -> str:
    """``<id@host>`` -> ``id@host`` (normalize a Message-ID to its bare form)."""
    if not msgid:
        return ""
    return msgid.strip().strip("<>").strip()


def _header_value(msg: Message, name: str) -> str:
    """A header value decoded to a str (email.header decode + ASCII fallback)."""
    raw = msg.get(name)
    if raw is None:
        return ""
    if isinstance(raw, email.header.Header):
        try:
            raw = str(raw)
        except Exception:  # noqa: BLE001 - header decode is best-effort
            return ""
    # ``email.header.decode_header`` + ``make_header`` handles RFC 2047 encoded
    # words (``=?utf-8?Q?...?=``). Fall back to the raw str on any decode failure.
    try:
        from email.header import decode_header, make_header
        return str(make_header(decode_header(raw)))
    except Exception:  # noqa: BLE001 - never crash on a malformed header
        return str(raw)


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _html_to_text(fragment: str) -> str:
    """Lossy HTML body -> plain text (no bs4 dep for email)."""
    # Drop <script>/<style> blocks entirely (their text is not content).
    fragment = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", fragment)
    # Turn block-level tags into newlines so paragraphs survive the tag strip.
    fragment = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"(?i)</(p|div|li|tr|h[1-6])>", "\n", fragment)
    fragment = _TAG_RE.sub("", fragment)
    fragment = _html.unescape(fragment)
    return _WS_RE.sub(" ", fragment).strip()


def _payload_text(msg: Message) -> str:
    """The message body as plain text, preferring ``text/plain``.

    Walks the MIME parts; if a ``text/plain`` part exists, use it. Otherwise
    fall back to the first ``text/html`` part (tags stripped). Quoted-reply
    lines (``>``-prefixed) are preserved as-is so the conversation context is
    intact. Returns "" when no readable body is found.
    """
    plain: Optional[str] = None
    html_body: Optional[str] = None
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        try:
            payload = part.get_payload(decode=True)
        except Exception:  # noqa: BLE001 - malformed payloads are best-effort
            payload = None
        if payload is None:
            continue
        charset = part.get_content_charset() or "utf-8"
        try:
            text = payload.decode(charset, errors="replace")
        except (LookupError, TypeError):
            text = payload.decode("utf-8", errors="replace")
        if ctype == "text/plain" and plain is None:
            plain = text
        elif ctype == "text/html" and html_body is None:
            html_body = text
    if plain is not None:
        return plain.strip()
    if html_body is not None:
        return _html_to_text(html_body)
    return ""


class EmailParser:
    """``parse(source_path) -> ParsedDocument`` via stdlib ``email`` (thread)."""

    source_type: str = "email"

    def parse(self, source_path: str) -> ParsedDocument:
        """Parse a directory of ``.eml`` files (a thread) or a single ``.eml``.

        A directory -> every ``*.eml`` in it is one message in the thread; a
        single ``.eml`` file -> a one-message thread. Non-``.eml`` files in a
        directory are skipped (honest: only ``.eml`` is wired; ``.mbox`` is
        deferred).
        """
        messages: list[Message] = []
        if os.path.isdir(source_path):
            for name in sorted(os.listdir(source_path)):
                if not name.lower().endswith(".eml"):
                    continue
                path = os.path.join(source_path, name)
                with open(path, "rb") as fh:
                    messages.append(email.message_from_binary_file(fh))
        else:
            with open(source_path, "rb") as fh:
                messages.append(email.message_from_binary_file(fh))
        return self.parse_messages(messages, source_path)

    def parse_text(self, eml_text: str, source_path: str = "") -> ParsedDocument:
        """Single-message mirror (no file IO); tests use this."""
        return self.parse_messages([email.message_from_string(eml_text)], source_path)

    def parse_messages(
        self, messages: list[Message], source_path: str = ""
    ) -> ParsedDocument:
        """Build the thread ``ParsedDocument`` from parsed ``Message`` objects.

        The testable core: no file IO, just the reply-tree reconstruction +
        section emission. Empty input -> an empty ``ParsedDocument`` (no crash).
        """
        # One node per message keyed by its bare Message-ID. A message with no
        # Message-ID gets a synthetic id so it still joins the thread (it will
        # root unless something references it, which nothing can).
        nodes: list[dict] = []
        seen_ids: set[str] = set()
        for i, msg in enumerate(messages):
            mid = _strip_angle(_header_value(msg, "Message-ID")) or f"_synthetic_{i}"
            if mid in seen_ids:
                # Duplicate Message-ID (a forwarded copy in the folder) -- keep
                # the first; the duplicate is data noise, not a second voice.
                continue
            seen_ids.add(mid)
            in_reply_to_raw = _header_value(msg, "In-Reply-To")
            parent = _strip_angle(in_reply_to_raw)
            if not parent:
                # Fall back to the LAST Message-ID in References (RFC 5322: the
                # immediate parent is the last entry in the References chain).
                refs = _header_value(msg, "References")
                ref_ids = [
                    _strip_angle(r) for r in re.findall(r"<[^>]+>", refs) if r
                ]
                parent = ref_ids[-1] if ref_ids else ""
            nodes.append({
                "msgid": mid,
                "parent": parent,
                "from": _ascii(_header_value(msg, "From")),
                "subject": _header_value(msg, "Subject"),
                "date": _header_value(msg, "Date"),
                "references": _header_value(msg, "References"),
                "body": _payload_text(msg),
            })

        if not nodes:
            return ParsedDocument(
                source_type=self.source_type, source_path=source_path,
                sections=[], title=os.path.splitext(os.path.basename(source_path))[0],
            )

        by_id = {n["msgid"]: n for n in nodes}
        # Children index: parent msgid -> list of child nodes (chronological by Date).
        children: dict[str, list[dict]] = {}
        roots: list[dict] = []
        for n in nodes:
            p = n["parent"]
            if p and p in by_id and p != n["msgid"]:
                children.setdefault(p, []).append(n)
            else:
                # Orphan reply (parent not in the set) -> roots at level 1, not lost.
                roots.append(n)

        def _date_key(n: dict) -> tuple:
            try:
                dt = email.utils.parsedate_to_datetime(n["date"])
                return (dt.timestamp(),) if dt is not None else (float("inf"),)
            except (TypeError, ValueError):
                return (float("inf"),)

        for lst in children.values():
            lst.sort(key=_date_key)
        roots.sort(key=_date_key)

        # Flattened DFS: root-first, then each child subtree in chronological order.
        ordered: list[dict] = []

        def _visit(n: dict, level: int) -> None:
            n["level"] = level
            ordered.append(n)
            for child in children.get(n["msgid"], []):
                _visit(child, level + 1)

        for root in roots:
            _visit(root, 1)

        # Thread subject: the common subject with Re:/Fwd: prefix runs collapsed
        # to one. Fall back to the first message's subject, then the file/dir stem.
        base_subject = ""
        for n in nodes:
            if n["subject"]:
                base_subject = n["subject"]
                break
        title = _ascii(_collapse_subject(base_subject)) or os.path.splitext(
            os.path.basename(source_path)
        )[0]

        sections: list[RawSection] = []
        authors: list[str] = []
        seen_authors: set[str] = set()
        root_date = ""
        message_ids: list[str] = []
        in_reply_to_map: dict[str, str] = {}
        references_map: dict[str, str] = {}

        for n in ordered:
            date_short = _short_date(n["date"])
            heading = f"{n['from']} ({date_short})".strip()
            sections.append(RawSection(
                heading=heading, level=n["level"], content=n["body"],
            ))
            if n["from"] and n["from"] not in seen_authors:
                seen_authors.add(n["from"])
                authors.append(n["from"])
            message_ids.append(n["msgid"])
            if n["parent"]:
                in_reply_to_map[n["msgid"]] = n["parent"]
            if n["references"]:
                references_map[n["msgid"]] = n["references"]

        # created_at = the root message's raw Date header string (the
        # ParsedDocument field keeps the original RFC date; downstream code that
        # needs a timestamp parses it via email.utils.parsedate_to_datetime).
        if roots:
            root_date = roots[0]["date"]

        return ParsedDocument(
            source_type=self.source_type,
            source_path=source_path,
            sections=sections,
            title=title,
            authors=authors,
            created_at=root_date or None,
            metadata={
                "message_ids": message_ids,
                "in_reply_to": in_reply_to_map,
                "references": references_map,
            },
        )


def _collapse_subject(subject: str) -> str:
    """``Re: Re: Fwd: subject`` -> ``Re: subject`` (collapse the prefix run)."""
    if not subject:
        return ""
    s = subject.strip()
    m = re.match(r"^\s*((?:Re|Fwd|Fw):\s*)+(.*)$", s, re.IGNORECASE)
    if not m:
        return s
    # The LAST prefix token's kind is the surviving prefix (Re over Fwd over Fw
    # when mixed, but a run is usually one kind). Keep it simple: reuse the last
    # prefix word seen.
    prefixes = re.findall(r"(Re|Fwd|Fw):", s, re.IGNORECASE)
    last = prefixes[-1] if prefixes else "Re"
    body = m.group(2).strip()
    return f"{last.capitalize()}: {body}" if body else f"{last.capitalize()}:"


def _short_date(date_str: str) -> str:
    """A short, ASCII-safe date string for the section heading (``2026-07-14``)."""
    if not date_str:
        return ""
    try:
        dt = email.utils.parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        return ""
    if dt is None:
        return ""
    try:
        return dt.strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001 - never crash on a weird date
        return ""