"""Measure per-span GLiNER2 confidence on CPU for the 2 failing-test inputs.

Prints every span the counting layer proposed for the `topic`/`decision`
categories with its sigmoid confidence, sorted descending. Shows how far below
0.3 the matching spans sit on CPU -- decides whether a CPU threshold-tune can
recover them (spans at 0.2-0.29 -> maybe; spans at <0.1 -> GPU/finetune only).
"""
import json
import sys
sys.path.insert(0, ".")
from gliner2 import GLiNER2
from src.encoding.gliner_extractor import _STABLE_SCHEMA
from src.config import config

ext = GLiNER2.from_pretrained(config.gliner2_model)


def show(label, text, expected_keywords=None):
    print(f"\n=== {label} ===")
    if expected_keywords:
        print("expected keywords:", expected_keywords)
    r = ext.extract(text, schema=_STABLE_SCHEMA, threshold=0.0, include_confidence=True)
    ents = r.get("entities", {}) or {}
    if isinstance(ents, list) and ents:
        ents = ents[0]
    for cat in ("decision", "topic"):
        spans = ents.get(cat, []) or []
        # spans may be list[str] (no confidence) or list[{text,confidence}]
        pairs = []
        for s in spans:
            if isinstance(s, dict):
                pairs.append((s.get("text", ""), float(s.get("confidence", 0.0))))
            else:
                pairs.append((str(s), None))
        pairs.sort(key=lambda p: (p[1] is None, -(p[1] or 0)))
        print(f"  {cat} ({len(pairs)} spans, sorted by conf desc):")
        for txt, conf in pairs[:12]:
            mark = ""
            if expected_keywords and conf is not None:
                tl = txt.lower()
                if any(k in tl for k in expected_keywords):
                    mark = "  <== matches expected keyword"
            print(f"    {conf!s:>6}  {txt!r}{mark}")


show("decisions text",
     "I've decided to go with DEBOUNCED for the WAL sync mode.",
     expected_keywords=["debounc", "wal", "sync", "go with", "decid"])

with open("data/sample_conversations.jsonl", encoding="utf-8") as f:
    conv = json.loads(f.readline())
full = " ".join(f"User: {u} Assistant: {a}" for u, a in conv["turns"])
kw = [k for k in conv.get("expected_topics", ["database_design"])[0].lower().split("_") if k]
show(f"conv {conv.get('id')} (expected_topics={conv.get('expected_topics')})", full, expected_keywords=kw)