"""Pull the assistant passages most relevant to Phase 2c intent.

Phase 2c = Working Memory + SSM Chunking + Presentation Gate (+ persistence).
The relevant user prompts and their assistant responses, by thread index:
  [002]  WM = activated pointers + attention (neuroscience root)
  [008]  SSM needs LLM query-plan + embed; JEPA gates "when enough embedding"
  [058]  EXPAND when not confident; maybe an LLM process to plan how to expand
  [092]  SSM losing the past over time; re-encode requested info the SSM forgot
  [094]  saturation/feedback: don't overweight importance indefinitely
  [128]  chunking: divide graph results by context size, SSM compresses prior chunks
  [130]  JEPA to handle chunking/compression; size SSM to fit results or standard
  [132]  whole request life cycle; large chat history / docs / ingestion
  [142]  ssm+jepa to compress large prompts/docs AND query results
  [144]  not always LLM-process results; sometimes just formatting context; consumers differ
  [146]  explicit API to decide how results returned; hard to train JEPA (no feedback loop)
  [174]  chatbot vs database vs agent/harness; procedural memory; what 2c system is capable of
"""
import json
from pathlib import Path

with open('docs/The_Ponder_Engine_Chat.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
m = d['messages']
by_id = {x['message_id']: x for x in m}
thread = []
cur = [x for x in m if x.get('parent_message') is None][0]
seen = set()
while cur and cur['message_id'] not in seen:
    seen.add(cur['message_id'])
    thread.append(cur)
    nid = cur.get('latest_child_message')
    cur = by_id.get(nid) if nid else None

WANT = [2, 8, 58, 92, 94, 128, 130, 132, 142, 144, 146, 174]
out = []
for i in WANT:
    if i >= len(thread):
        continue
    x = thread[i]
    role = x.get('message_type', '?').upper()
    body = x.get('message') or ''
    out.append(f"\n{'='*78}\n## THREAD[{i:03d}] {role}  (msg_id={x.get('message_id')}, len={len(body)})\n{'='*78}\n")
    out.append(body)

Path('scripts/_scratch/ponder_2c_passages.md').write_text(''.join(out), encoding='utf-8')
print('wrote scripts/_scratch/ponder_2c_passages.md')
print('total chars:', sum(len(thread[i].get('message') or '') for i in WANT if i < len(thread)))