"""E0-v2 -- the decisive regime: memory consult vs stuffing when the corpus
EXCEEDS the prompt, with confuser twins making relevance gradeable.

Corpus: 300 work-log sessions x 4 fact-lines each, values drawn per session
(so every "probe latency" twin elsewhere differs -> a wrong-memory answer
carries the wrong token and FAILS grading).

Arms (all retries=2 on relay):
    M  :: consult(memory, q, k=3) -> answer from recalled episodes
    C2 :: stuffing: as many MOST-RECENT episode lines as fit a ~5k-token window
    C1 :: no memory, fresh context

Regimes:
    recent (inside stuffed window)   -> tie by construction (sanity)
    old    (outside stuffed window)  -> THE decisive cell: C2 must fail, M must not

Also reports relevance readouts from the M arm: consult top-1 correctness
(is the gold episode the FIRST recalled) per regime.

Run:  python scripts/icm/e0_v2_corpus.py  (corpus gen only; runners import it)
"""

from __future__ import annotations

import random

FACTS = [
    ("deploy", "probe returned {lat}ms", ("lat", list(range(8, 96, 4)))),
    ("storage", "disk at {pct} percent", ("pct", list(range(62, 97, 2)))),
    ("backup", "backup completed at {ts}", ("ts", [f"{h:02d}:{m:02d}" for h in range(0, 6) for m in (5, 25, 45)])),
    ("images", "pruned {n} stale images, freed {gb} GB", ("n", list(range(12, 480, 6)))),
]
# questions ask one fact per session; confuser twins = the other 299 sessions
# with the same template but different values.

N_SESSIONS = 300
EPISODES_PER_SESSION = 4
STUFF_RECENT = 18000  # chars; recent-window stuffing budget (~4.5k tokens)


def corpus_seed(seed: int = 7) -> dict[int, dict]:
    """ts = the session ANCHOR: unique by construction (shuffled 1440 slots).
    Everything else intentionally repeats across the corpus (confuser twins)."""
    rng = random.Random(seed)
    ts_slots = [f"{h:02d}:{m:02d}" for h in range(0, 24) for m in range(0, 60)]
    ts_shuffled = rng.sample(ts_slots, N_SESSIONS)
    values: dict[int, dict] = {}
    for si in range(N_SESSIONS):
        values[si] = {
            "lat": rng.choice(range(8, 96, 4)),
            "pct": rng.choice(range(62, 97, 2)),
            "ts": ts_shuffled[si],
            "n": rng.choice(range(12, 300, 6)),
            "gb": rng.choice(range(4, 62, 2)),
        }
    return values


def session_texts(si: int, values: dict[int, dict]) -> list[str]:
    vals = values[si]
    lines = []
    for tag, tmpl, (key, _pool) in FACTS:
        lines.append(f"Checkpoint [{tag}]: {tmpl.format_map(vals)}")
    return lines


def episode_text(si: int, values: dict[int, dict], idx: int) -> str:
    lines = session_texts(si, values)
    return f"[session {si:04d}] " + lines[idx]


# THE question: anchored by a distinctive OTHER fact of that session
# (so recall must find the right twin; the answer's value is NOT in the query)
QUESTIONS = {
    # anchor (the unique ts) names the session; the answer token belongs to a
    # fact whose VALUE appears in hundreds of other sessions (wrong-memory
    # answers carry a twin's token and fail grading).
    "lat": ("In the run whose backup completed at {ts}, what latency did the probe return?", "lat"),
    "pct": ("In the run whose backup completed at {ts}, what was the disk percent?", "pct"),
    "n": ("In the run whose backup completed at {ts}, how many stale images were pruned?", "n"),
}


def question_for(si: int, values: dict[int, dict]) -> tuple[str, str]:
    """Returns (question, gold_answer_token); question type cycles per session."""
    tmpl, key = list(QUESTIONS.values())[si % len(QUESTIONS)]
    return tmpl.format_map(values[si]), str(values[si][key])


if __name__ == "__main__":
    import io, json, os, time
    values = corpus_seed(7)
    # sanity: uniqueness of anchors + gold tokens
    anchors = [values[si]["ts"] for si in range(N_SESSIONS)]
    assert len(set(anchors)) == N_SESSIONS, "ts anchors must be unique"
    golds = [question_for(si, values)[1] for si in range(0, N_SESSIONS)]
    twin_rate = round(1 - len(set(golds)) / N_SESSIONS, 3)
    print(f"confuser twin rate (gold tokens repeating across corpus): {twin_rate:.2%}")
    print("corpus: 300 sessions x 4 fact lines; anchors unique")
    print("sample pair:")
    q, g = question_for(7, values)
    print("  q:", q, "| gold:", g)
    print("  its episode lines:")
    for line in session_texts(7, values):
        print("   ", line)