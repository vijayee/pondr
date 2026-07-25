"""One-off: analyze The_Ponder_Engine_Chat.json structure to plan extraction.

Not a committed tool — a scratch probe. Writes ASCII-safe summaries to stdout.
"""
import json
from collections import Counter

with open('docs/The_Ponder_Engine_Chat.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

m = d['messages']
by_id = {x['message_id']: x for x in m}

# Tree: parent_message points up; latest_child_message + preferred_response_id point down.
roots = [x for x in m if x.get('parent_message') is None]
print(f'roots: {len(roots)}  | total msgs: {len(m)}')

# How many parents have multiple children (branching)?
children_by_parent = {}
for x in m:
    p = x.get('parent_message')
    children_by_parent.setdefault(p, []).append(x['message_id'])
branch_points = {p: c for p, c in children_by_parent.items() if len(c) > 1}
print(f'branch points (parent with >1 child): {len(branch_points)}')
for p, c in list(branch_points.items())[:10]:
    print(f'  parent={p} children={c}')

# preferred_response_id presence
pref = [x for x in m if x.get('preferred_response_id') is not None]
print(f'msgs with preferred_response_id: {len(pref)}')

# size split
user_chars = sum(len(x.get('message') or '') for x in m if x.get('message_type') == 'user')
asst_chars = sum(len(x.get('message') or '') for x in m if x.get('message_type') == 'assistant')
sys_chars = sum(len(x.get('message') or '') for x in m if x.get('message_type') == 'system')
print(f'user chars: {user_chars} (~{user_chars//4} tok)')
print(f'assistant chars: {asst_chars} (~{asst_chars//4} tok)')
print(f'system chars: {sys_chars}')

# Build main thread: follow latest_child_message from root, prefer preferred_response_id at branches.
# Find the system/root
root = roots[0]
thread = []
cur = root
while cur is not None:
    thread.append(cur)
    nxt = cur.get('latest_child_message')
    if nxt is None:
        break
    nxt_node = by_id.get(nxt)
    if nxt_node is None:
        break
    # if this node has siblings (branch), pick preferred_response_id if set on the PARENT
    parent_id = nxt_node.get('parent_message')
    sibs = children_by_parent.get(parent_id, [nxt])
    if len(sibs) > 1:
        # does the parent (cur) have a preferred_response_id?
        pr = cur.get('preferred_response_id')
        if pr is not None and pr in by_id:
            nxt_node = by_id[pr]
    cur = nxt_node

print(f'main-thread length (following latest_child + preferred): {len(thread)}')
types = Counter(x.get('message_type') for x in thread)
print(f'main-thread types: {types}')
mt_chars = sum(len(x.get('message') or '') for x in thread)
print(f'main-thread chars: {mt_chars} (~{mt_chars//4} tok)')