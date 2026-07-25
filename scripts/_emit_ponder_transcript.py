"""Emit clean artifacts from The_Ponder_Engine_Chat.json.

Outputs (UTF-8, overwriting):
  scripts/_scratch/ponder_transcript.md   -- main-thread transcript (177 msgs)
  scripts/_scratch/ponder_user_msgs.md    -- all user messages verbatim, in thread order

Main thread = follow latest_child_message from the single root. Branch points
(alternate generations) are flattened to the latest_child line; alternates are
NOT included (preferred_response_id is unused in this export, so latest_child is
the only canonical selector).
"""
import json
from pathlib import Path

SRC = Path('docs/The_Ponder_Engine_Chat.json')
OUT_DIR = Path('scripts/_scratch')
OUT_DIR.mkdir(parents=True, exist_ok=True)

with SRC.open('r', encoding='utf-8') as f:
    d = json.load(f)

m = d['messages']
by_id = {x['message_id']: x for x in m}
roots = [x for x in m if x.get('parent_message') is None]

# Walk main thread via latest_child_message.
thread = []
cur = roots[0]
seen = set()
while cur is not None and cur['message_id'] not in seen:
    seen.add(cur['message_id'])
    thread.append(cur)
    nxt_id = cur.get('latest_child_message')
    cur = by_id.get(nxt_id) if nxt_id is not None else None

# Transcript with turn indices.
tx_lines = [
    f"# The Ponder Engine chat — main thread transcript",
    f"Source: docs/The_Ponder_Engine_Chat.json  ({len(thread)} messages on the latest_child line)",
    f"chat_session_id: {d.get('chat_session_id')}",
    f"time_created: {d.get('time_created')}",
    "",
]
for i, x in enumerate(thread):
    role = x.get('message_type', '?').upper()
    ts = x.get('time_sent', '')
    mid = x.get('message_id')
    body = x.get('message') or ''
    tx_lines.append(f"\n## [{i:03d}] {role}  (msg_id={mid}, {ts})\n")
    tx_lines.append(body)

(OUT_DIR / 'ponder_transcript.md').write_text(''.join(tx_lines), encoding='utf-8')

# User messages verbatim, in thread order.
u_lines = [
    f"# The Ponder Engine chat — USER messages (verbatim, thread order)",
    f"{sum(1 for x in thread if x.get('message_type')=='user')} user messages on the main thread",
    "",
]
for i, x in enumerate(thread):
    if x.get('message_type') != 'user':
        continue
    ts = x.get('time_sent', '')
    mid = x.get('message_id')
    body = x.get('message') or ''
    u_lines.append(f"\n## [{i:03d}] USER  (msg_id={mid}, {ts})\n")
    u_lines.append(body)

(OUT_DIR / 'ponder_user_msgs.md').write_text(''.join(u_lines), encoding='utf-8')

# ASCII-safe summary to stdout.
print(f'thread msgs: {len(thread)}')
print(f'transcript chars: {sum(len(x.get("message") or "") for x in thread)}')
print(f'user msgs on thread: {sum(1 for x in thread if x.get("message_type")=="user")}')
print(f'user msgs total (all branches): {sum(1 for x in m if x.get("message_type")=="user")}')