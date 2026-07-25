"""One-off: stream-inspect lmsys-chat-1m to design the serve-trace mapping.
Looks at column names, turn-count distribution (the multi-turn tail is what's
usable for the per-source z_r gap), and prints a couple multi-turn examples.
"""
import sys
from collections import Counter
from datasets import load_dataset

ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
turn_counts = Counter()
langs = Counter()
n_seen = 0
examples_long = []
first_keys = None
for row in ds:
    if first_keys is None:
        first_keys = list(row.keys())
    conv = row.get("conversation") or row.get("conversations") or row.get("messages")
    lang = row.get("language") or row.get("detected_language") or row.get("language_tag") or "?"
    langs[lang] += 1
    n_turns = len(conv) if conv is not None else 0
    turn_counts[n_turns] += 1
    # capture a few English multi-turn examples (>=6 messages => >=3 user turns)
    if conv is not None and len(conv) >= 6 and len(examples_long) < 2:
        is_en = isinstance(lang, str) and lang.lower().startswith("en")
        if is_en:
            examples_long.append((row.get("conversation_id"), lang, conv))
    n_seen += 1
    if n_seen >= 20000:
        break

print("columns:", first_keys)
print(f"seen {n_seen} rows")
print("turn-count distribution (n_messages in conversation):")
for k in sorted(turn_counts):
    bar = "#" * min(60, int(turn_counts[k] * 60 / max(turn_counts.values())))
    print(f"  {k:>3}: {turn_counts[k]:>6}  {bar}")
print("top languages:", langs.most_common(5))
print()
for cid, lang, conv in examples_long:
    print(f"--- example conv_id={cid} lang={lang} n_msgs={len(conv)} ---")
    for i, m in enumerate(conv):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        content = str(content).replace("\n", " ")
        print(f"  [{i}] {role}: {content[:140]}")
    print()
