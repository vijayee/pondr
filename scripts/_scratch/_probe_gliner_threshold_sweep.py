"""Threshold sweep on CPU: for each threshold, run all 20 sample conversations
and the decisions text, report mean topic recall + whether decisions is non-empty
+ how many garbage (len<=2 or pure-punctuation) spans leak through.

Tells us if a CPU-tuned threshold recovers the 2 failing tests WITHOUT GPU or
training, and how much noise it lets in.
"""
import json
import re
import sys
sys.path.insert(0, ".")
from gliner2 import GLiNER2
from src.encoding.gliner_extractor import _STABLE_SCHEMA
from src.config import config

ext = GLiNER2.from_pretrained(config.gliner2_model)

with open("data/sample_conversations.jsonl", encoding="utf-8") as f:
    convs = [json.loads(line) for line in f]


def topic_recall(expected_labels, spans):
    if not expected_labels:
        return 1.0
    spans_low = [s.lower() for s in spans if isinstance(s, str)]
    hits = 0
    for label in expected_labels:
        kws = [k for k in label.lower().split("_") if k]
        if kws and any(kw in sp for kw in kws for sp in spans_low):
            hits += 1
    return hits / len(expected_labels)


def is_garbage(s):
    s = s.strip()
    if len(s) <= 2:
        return True
    if re.fullmatch(r"[\W\d]+", s):  # pure punctuation/digits
        return True
    return False


print("\nthreshold | mean_topic_recall | convs_w_garbage_leak | mean_garbage_per_conv | decisions_nonempty")
for thr in (0.3, 0.1, 0.05, 0.03, 0.02, 0.01):
    recalls = []
    garbage_counts = []
    convs_with_garbage = 0
    for conv in convs:
        full = " ".join(f"User: {u} Assistant: {a}" for u, a in conv["turns"])
        r = ext.extract(full, schema=_STABLE_SCHEMA, threshold=thr)
        ents = r.get("entities", {}) or {}
        if isinstance(ents, list) and ents:
            ents = ents[0]
        topics = ents.get("topic", []) or []
        recalls.append(topic_recall(conv.get("expected_topics", []), topics))
        g = sum(1 for t in topics if is_garbage(t))
        garbage_counts.append(g)
        if g:
            convs_with_garbage += 1
    # decisions text
    dtext = "I've decided to go with DEBOUNCED for the WAL sync mode."
    dr = ext.extract(dtext, schema=_STABLE_SCHEMA, threshold=thr)
    dents = dr.get("entities", {}) or {}
    if isinstance(dents, list) and dents:
        dents = dents[0]
    decisions = dents.get("decision", []) or []
    dec_nonempty = len(decisions) > 0
    print(f"  {thr:.2f}    |     {sum(recalls)/len(recalls):.2f}          |        "
          f"{convs_with_garbage:2d}/20            |        "
          f"{sum(garbage_counts)/len(garbage_counts):.1f}            |    {dec_nonempty}")
    if thr in (0.05, 0.03):
        print(f"         decisions spans @ {thr}: {decisions}")