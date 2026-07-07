#!/bin/bash
# Full Phase 1b scale run on the pod: ingest 1000 DialogSum + 500 SAMSum,
# build a FAISS index over the DialogSum DB, run the WaveDB reopen + NUL-free
# scan scale gate. Writes a runner.log line per stage + an ALL_DONE marker.
set -u
cd /root/hippo
LOG=/root/runner.log
echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

# 1. DialogSum 1000 -> /root/ingest_db_dialogsum
python scripts/process_corpus.py \
  --input /root/corpora/dialogsum.jsonl/dialogsum.jsonl \
  --user poduser --db /root/ingest_db_dialogsum --limit 1000 \
  --extractions /root/dialogsum_extractions.jsonl \
  --report /root/dialogsum_ingestion_report.json \
  --resume --progress-every 50 > /root/ingest_dialogsum.log 2>&1
echo "DIALOGSUM_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

# 2. SAMSum 500 -> /root/ingest_db_samsum
python scripts/process_corpus.py \
  --input /root/corpora/samsum.jsonl/samsum.jsonl \
  --user poduser --db /root/ingest_db_samsum --limit 500 \
  --extractions /root/samsum_extractions.jsonl \
  --report /root/samsum_ingestion_report.json \
  --resume --progress-every 50 > /root/ingest_samsum.log 2>&1
echo "SAMSUM_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

# 3. FAISS index over the larger DialogSum DB
python scripts/build_vector_index.py --db /root/ingest_db_dialogsum > /root/faiss.log 2>&1
echo "FAISS_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

# 4. WaveDB scale gate: reopen the DialogSum DB, scan content/ep/, assert NUL-free.
python - > /root/gate.log 2>&1 <<'PY'
import sys
sys.path.insert(0, "/root/hippo")
import wavedb
db = wavedb.WaveDB("/root/ingest_db_dialogsum")
keys = []
for k, _ in db.create_read_stream(start="content/ep/", end="content/ep/\x7f"):
    keys.append(k)
nul = [k for k in keys if "\x00" in k]
print(f"scan_keys={len(keys)} nul_keys={len(nul)}")
assert not nul, f"NUL bytes in {len(nul)} keys"
# Count episodes
eps = set()
for k in keys:
    parts = k.split("/", 3)
    if len(parts) >= 3 and parts[2]:
        eps.add(parts[2])
print(f"episodes={len(eps)}")
db.close()
print("GATE_PASS")
PY
echo "GATE_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

touch /root/ALL_DONE
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"