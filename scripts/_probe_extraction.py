"""Probe: run the fixed GLiNER extractor on a small DialogSum sample.

Validates the Phase 1a extraction fix on real daily-life conversations:
- decisions must be whole content spans (not char-split ['d','e','c',...]).
- topics must be varied free-form spans (not collapsed to one tech label).
- tones should still come from the bounded taxonomy.

Prints per-episode entities/topics/tones/decisions + an aggregate summary.
"""
from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, "/root/hippo")

from datasets import load_dataset

from src.encoding.gliner_extractor import GLiNERExtractor


def dialogsum_text(row) -> str:
    """Render a DialogSum dialogue as User/Assistant-style turns for the extractor."""
    dialogue = row["dialogue"]
    # DialogSum format: "#Person1#: ... #Person2#: ..." — keep it as raw text;
    # the extractor is domain-agnostic. Add the topic field as context too.
    return f"Topic: {row.get('topic','')}\n{dialogue}"


def main(n: int = 12) -> None:
    ds = load_dataset("knkarthick/dialogsum", split="train")
    ext = GLiNERExtractor()
    topic_c = Counter()
    tone_c = Counter()
    dec_total = 0
    char_split = 0
    n_eps = 0
    for i in range(min(n, len(ds))):
        row = ds[i]
        text = dialogsum_text(row)
        r = ext.extract(text)
        n_eps += 1
        topics = r["topics"]
        tones = r["tones"]
        decisions = r["decisions"]
        for t in topics:
            topic_c[t] += 1
        for t in tones:
            tone_c[t] += 1
        dec_total += len(decisions)
        # char-split regression: any single-char decision = the old bug
        if any(isinstance(d, str) and len(d) <= 1 for d in decisions):
            char_split += 1
        print(f"--- row {i} (topic_field={row.get('topic','')!r}) ---")
        print(f"  entities : {r['entities']}")
        print(f"  topics   : {topics}")
        print(f"  tones    : {tones}")
        print(f"  decisions: {decisions}")
    print("\n=== AGGREGATE ===")
    print(f"episodes          : {n_eps}")
    print(f"distinct topics   : {len(topic_c)} -> {dict(topic_c.most_common(10))}")
    print(f"distinct tones    : {len(tone_c)} -> {dict(tone_c)}")
    print(f"total decisions   : {dec_total}")
    print(f"char-split episodes: {char_split}  (must be 0)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(n)