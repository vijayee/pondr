"""Diagnostic probe: show what GLiNER2 returns at several thresholds for the
0.00-entity conversations. Run on the pod (needs GPU + gliner2)."""
import json
from src.encoding.gliner_extractor import GLiNERExtractor, _STABLE_SCHEMA

convs = [json.loads(l) for l in open(
    "data/sample_conversations.jsonl", encoding="utf-8").read().splitlines() if l.strip()]
by_id = {c["id"]: c for c in convs}
ex = GLiNERExtractor()
for cid in ["conv_004", "conv_013", "conv_018", "conv_020"]:
    c = by_id[cid]
    text = " ".join("User: " + u + " Assistant: " + a for u, a in c["turns"])
    print("\n=== " + cid + " ===")
    print("expected_entities:", c.get("expected_entities"))
    print("expected_tones:", c.get("expected_tones"))
    for th in [0.3, 0.2, 0.1]:
        r = ex.extractor.extract(text, schema=_STABLE_SCHEMA, threshold=th)
        ents = {k: v for k, v in r.get("entities", {}).items() if v}
        print("  th=" + str(th) + ": entities=" + str(ents) +
              " topics=" + str(r.get("topics")) + " tones=" + str(r.get("tones")))
    print("  text[:240]:", text[:240])