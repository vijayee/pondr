"""Download + normalize conversation corpora for the encoding pipeline.

Pulls public HuggingFace datasets (no API key) and normalizes each
conversation to the JSONL shape ``scripts/process_corpus.py`` expects:

    {"id": "...", "turns": [[user, assistant], ...], "summary": "..."}

Two corpora are supported (selectable via ``--corpus``; default both):

- **DialogSum** (`knorng/DialogSum`): dialogues between ``#Person1#`` /
  ``#Person2#`` with a reference summary + topic.
- **SAMSum** (`Samsung/samsum`): chat-style dialogues with speaker names and a
  reference summary.

Normalization parses each dialogue into ``(speaker, text)`` utterances, merges
consecutive same-speaker utterances, and pairs them as ``[speaker_A,
speaker_B]`` turns so ``encode_conversation`` sees ``[user, assistant]`` pairs.
A trailing unpaired utterance becomes ``[text, ""]``.

Run on the pod (so the corpus is local to the ingestion run) or locally:

    python scripts/download_corpora.py --out data/corpora --limit 50
    python scripts/download_corpora.py --corpus dialogsum --out data/corpora
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _merge_consecutive(utts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge runs of the same speaker into one utterance."""
    merged: list[tuple[str, str]] = []
    for spk, text in utts:
        if merged and merged[-1][0] == spk:
            prev_spk, prev_text = merged[-1]
            merged[-1] = (prev_spk, f"{prev_text} {text}".strip())
        else:
            merged.append((spk, text))
    return merged


def _pair_turns(utts: list[tuple[str, str]]) -> list[list[str]]:
    """Pair merged utterances as ``[speaker_A, speaker_B]`` turns."""
    turns: list[list[str]] = []
    for i in range(0, len(utts), 2):
        a = utts[i][1]
        b = utts[i + 1][1] if i + 1 < len(utts) else ""
        turns.append([a, b])
    return turns


_DIALOGSUM_SPLIT = re.compile(r"#(Person\d+)#:\s*")


def parse_dialogsum(dialogue: str) -> list[list[str]]:
    """Parse a DialogSum dialogue string into [user, assistant] turn pairs."""
    # Split on `#PersonN#:` markers; the first element is empty (text before
    # the first marker) so drop it.
    parts = _DIALOGSUM_SPLIT.split(dialogue)
    utts: list[tuple[str, str]] = []
    # parts looks like ['', 'Person1', 'text', 'Person2', 'text', ...]
    for i in range(1, len(parts), 2):
        speaker = parts[i]
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if text:
            utts.append((speaker, text))
    return _pair_turns(_merge_consecutive(utts))


_SAMSUM_LINE = re.compile(r"^([^:]+):\s*(.*)$")


def parse_samsum(dialogue: str) -> list[list[str]]:
    """Parse a SAMSum dialogue (one utterance per line, `Name: text`) into pairs."""
    utts: list[tuple[str, str]] = []
    for line in dialogue.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SAMSUM_LINE.match(line)
        if m:
            utts.append((m.group(1).strip(), m.group(2).strip()))
        else:
            # Continuation of the previous utterance (no speaker prefix).
            if utts:
                spk, prev = utts[-1]
                utts[-1] = (spk, f"{prev} {line}".strip())
    return _pair_turns(_merge_consecutive(utts))


def _write_jsonl(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def fetch_dialogsum(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("knorng/DialogSum", split="train")
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        if limit and len(rows) >= limit:
            break
        turns = parse_dialogsum(ex.get("dialogue", ""))
        if not turns:
            continue
        rows.append({
            "id": f"dialogsum_{i:05d}",
            "turns": turns,
            "summary": (ex.get("summary") or "").strip(),
            "topic": (ex.get("topic") or "").strip(),
        })
    return rows


def fetch_samsum(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Samsung/samsum", split="train")
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        if limit and len(rows) >= limit:
            break
        turns = parse_samsum(ex.get("dialogue", ""))
        if not turns:
            continue
        rows.append({
            "id": ex.get("id") or f"samsum_{i:05d}",
            "turns": turns,
            "summary": (ex.get("summary") or "").strip(),
        })
    return rows


_CORPORA = {
    "dialogsum": (fetch_dialogsum, "dialogsum.jsonl"),
    "samsum": (fetch_samsum, "samsum.jsonl"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download + normalize conversation corpora.")
    parser.add_argument("--corpus", choices=list(_CORPORA) + ["all"], default="all",
                        help="Which corpus to fetch (default: all).")
    parser.add_argument("--out", default="data/corpora", help="Output directory for JSONL files.")
    parser.add_argument("--limit", type=int, help="Max conversations per corpus.")
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("ERROR: `datasets` not installed. Install with: pip install datasets",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    names = list(_CORPORA) if args.corpus == "all" else [args.corpus]
    total = 0
    for name in names:
        fetcher, fname = _CORPORA[name]
        print(f"Fetching {name}...")
        rows = fetcher(args.limit)
        n = _write_jsonl(rows, out_dir / fname)
        total += n
        print(f"  wrote {n} conversations → {out_dir / fname}")
    print(f"\nDone. {total} conversations across {len(names)} corpus/corpora.")
    return 0


if __name__ == "__main__":
    sys.exit(main())