"""Diagnostic: does GLiNER2 produce topic/decision spans on CPU, and at what
threshold? If spans appear at threshold 0.0 but not 0.3, the local test failures
are a CPU-confidence/threshold issue, not a transformers/gliner2-version issue.
"""
import json
import sys
sys.path.insert(0, ".")
from gliner2 import GLiNER2
from src.encoding.gliner_extractor import _STABLE_SCHEMA
from src.config import config

ext = GLiNER2.from_pretrained(config.gliner2_model)

text1 = "I've decided to go with DEBOUNCED for the WAL sync mode."
print("\n=== decisions text ===", repr(text1))
for thr in (0.3, 0.1, 0.0):
    r = ext.extract(text1, schema=_STABLE_SCHEMA, threshold=thr)
    ents = r.get("entities", {}) or {}
    print(f"-- threshold={thr} --")
    print("  decision :", ents.get("decision"))
    print("  topic    :", ents.get("topic"))
    print("  person   :", ents.get("person"), "project:", ents.get("project"),
          "tech:", ents.get("technology"))

# One sample conversation for topics.
data_path = "data/sample_conversations.jsonl"
with open(data_path, encoding="utf-8") as f:
    conv = json.loads(f.readline())
full = " ".join(f"User: {u} Assistant: {a}" for u, a in conv["turns"])
print("\n=== conv", conv.get("id"), "===")
print("expected_topics:", conv.get("expected_topics"))
for thr in (0.3, 0.1, 0.0):
    r = ext.extract(full, schema=_STABLE_SCHEMA, threshold=thr)
    ents = r.get("entities", {}) or {}
    print(f"-- threshold={thr} --  topic:", ents.get("topic"),
          "| decision:", ents.get("decision"),
          "| entities:", ents.get("person"), ents.get("project"), ents.get("technology"))