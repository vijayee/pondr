"""Step 3 smoke test: HippocampalStore end-to-end through WaveDB.

Exercises the path that was blocked by the WaveDB HBTrie scan-corruption bug:
ontology seed (many ``graph.insert_sync`` with empty values -> empty-value
graph-index keys) + ``encode_episode`` (atomic ``batch_sync`` of content puts
+ ``expand_triple`` graph ops) + a graph-index scan and ``get_episode``.

Run:  python -m tests.test_store_smoke
"""
from __future__ import annotations

import os
import sys
import tempfile

from src.memory.episode import Episode
from src.memory.store import HippocampalStore


def _make_episode(eid: str, follows: str | None = None) -> Episode:
    return Episode(
        id=eid,
        timestamp=f"2026-07-04T12:00:0{eid[-1]}",
        summary=f"Summary for {eid}",
        full_text=f"User: hello {eid}\nAssistant: hi back from {eid}",
        entities=["acme", "project_alpha"],
        topics=["onboarding"],
        tones=["neutral"],
        decisions=["start_onboarding"],
        relations=[{"subject": "acme", "predicate": "has_project", "object": "project_alpha"}],
        follows=follows,
    )


def main() -> int:
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "hippo")
    failures: list[str] = []

    store = HippocampalStore(db_path)

    # 1. Ontology seed ran in __init__; scan the graph index for corrupt keys.
    graph_keys = [k for k, _ in store.db.create_read_stream(start="memory/", end=None)]
    corrupt = [k for k in graph_keys if "\x00" in k]
    print(f"[seed] graph index keys: {len(graph_keys)}  corrupt: {len(corrupt)}")
    if corrupt:
        failures.append(f"seed: {len(corrupt)} corrupt graph keys (first: {corrupt[0]!r})")
    if not graph_keys:
        failures.append("seed: no ontology triples written")

    # 2. Encode two episodes.
    store.encode_episode(_make_episode("ep_0001"))
    store.encode_episode(_make_episode("ep_0002", follows="ep_0001"))

    # 3. Graph index scan across BOTH seed + episodes (the >38-entry threshold).
    all_graph = [k for k, _ in store.db.create_read_stream(start="memory/", end=None)]
    corrupt2 = [k for k in all_graph if "\x00" in k]
    print(f"[eps]  graph index keys: {len(all_graph)}  corrupt: {len(corrupt2)}")
    if corrupt2:
        failures.append(f"eps: {len(corrupt2)} corrupt graph keys (first: {corrupt2[0]!r})")

    # 4. A graph query: episodes that mention entity 'acme' (scan the in_episode index).
    # expand_triple writes E:acme in_episode ep_xxxx as memory/spo/E:acme/in_episode/ep_xxxx.
    # start= is a lexicographic lower bound, so prefix-scan by also bounding end at
    # the next delimiter to avoid pulling in unrelated spo/ entries.
    hit_keys = [k for k, _ in store.db.create_read_stream(
        start="memory/spo/E:acme/in_episode/",
        end="memory/spo/E:acme/in_episode/\x7f")]
    print(f"[query] E:acme in_episode hits: {hit_keys}")
    expected = "memory/spo/E:acme/in_episode/ep_0001"
    if any("\x00" in k for k in hit_keys):
        failures.append(f"query: corrupt keys in E:acme/in_episode scan: {[k for k in hit_keys if chr(0) in k]}")
    if expected not in hit_keys:
        failures.append(f"query: exact key {expected!r} not in hits ({hit_keys})")
    if "memory/spo/E:acme/in_episode/ep_0002" not in hit_keys:
        failures.append(f"query: ep_0002 not in hits ({hit_keys})")

    # 5. get_episode round-trips content.
    ep1 = store.get_episode("ep_0001")
    print(f"[get]  ep_0001 summary={ep1.summary!r}" if ep1 else "[get]  ep_0001 -> None")
    if ep1 is None or ep1.summary != "Summary for ep_0001":
        failures.append(f"get: ep_0001 content mismatch (got {ep1})")
    if ep1 and ep1.follows is not None:
        # get_episode is content-only; follows lives in the graph, not content.
        pass

    # 6. Ontology triple present + clean (subClassOf index).
    sub_keys = [k for k, _ in store.db.create_read_stream(
        start="memory/spo/", end="memory/spo/\x7f")]
    sample = [k for k in sub_keys if "subClassOf" in k][:3]
    print(f"[onto] subClassOf samples: {sample}")
    if not sample:
        failures.append("onto: no subClassOf triples found in graph scan")

    store.close()

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: all smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())