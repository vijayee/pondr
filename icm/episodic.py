"""Episodic memory store on WaveDB -- ICM Phase 1 (Job A baseline).

Layer shape (plan 5.2): this is L2, the addressed episodic store.

    KV subtree "icm":
        episodes/<id>   episode record (text, fidelity, links)
        index/ids       ledger-owned registry; "episode" is a first-class kind
    VectorLayer "icm-episodes" (FLAT/COSINE, dim = embedder dim):
        insert at capture, re-encode on blur, delete at drop

ENCODING NOTE (honest): the plan writes "chunk -> OCR-compress -> store". The
DeepSeek-OCR compressor (steal R10) is a heavy, GPU-gated external model, so
v1 stores VERBATIM text at full fidelity with a bge embedding for recall, and
degradation happens in the FIDELITY cascade instead: full -> gist (re-encoded
at lower information) -> dropped. The OCR encoder slots into exactly one seam
later: it would BE the gist re-encoder. Until then this is an
architecture-compatible superset of the plan's L2, not an OCR memory -- do
not describe it as the compressed store.

FORGETTING (plan 5.3 step 4, first principled schedule, rule version
"icm-forget/1"): surprise = any expectation linked by the episode resolved
VIOLATED (congruence axis -- a surprise keeps memory vivid; a merely-bad-but-
expected outcome does not):
    violated-linked  -> vivid: fidelity stays "full" past all horizons
    otherwise        -> age > blur_horizon_days: fidelity "gist" (text
                        re-encoded at lower fidelity and re-embedded)
                        age > drop_horizon_days: dropped (KV + index)
Links with pending resolution neither protect nor blur early (age applies).

The re-encoder is pluggable: gist_backend="heuristic" (default, deterministic
truncation) or "ocr" (DeepSeek-OCR re-encode daemon -- write-path only,
~1s/page warm, ~0.94 precision at Small mode; probe data in
scripts/icm/ocr_gist_probe.py / data/_scratch/icm/ocr_gist_probe.json).
"""

from __future__ import annotations

import time

from icm.ledger import ICMLedger

FORGET_RULE_VERSION = "icm-forget/1"
FIDELITY_FULL = "full"
FIDELITY_GIST = "gist"
DEFAULTS = {"blur_horizon_days": 7.0, "drop_horizon_days": 30.0}


def gist_of(text: str, max_chars: int = 220) -> str:
    """Deterministic first-drop fidelity: leading sentences up to max_chars.

    The OCR re-encoder replaces this when it arrives; until then this is a
    fixed, versioned degradation, not a semantic compressor.
    """
    marker = "[gist] "
    text = " ".join(text.split())
    if len(marker) + len(text) <= max_chars:
        return marker + text
    cut = max_chars - len(marker)
    head = text[:cut]
    if "." in head:
        cut = head.rindex(".") + 1
        head = text[:cut]
    ellipsis = "" if head.rstrip().endswith((".", "!", "?")) else "..."
    return marker + head.rstrip(" ,;") + ellipsis


class EpisodicStore:
    """Job A baseline store over an ICMLedger's WaveDB instance."""

    def __init__(self, ledger: ICMLedger, embedder, dim: int = 384,
                 blur_horizon_days: float = DEFAULTS["blur_horizon_days"],
                 drop_horizon_days: float = DEFAULTS["drop_horizon_days"],
                 gist_backend: str = "heuristic"):
        self.ledger = ledger
        self.embedder = embedder  # callable: [text] -> [[float] * dim]
        self.dim = dim
        self.blur_horizon_days = blur_horizon_days
        self.drop_horizon_days = drop_horizon_days
        # blur backend: "heuristic" (deterministic first-sentences gist,
        # default, zero deps) or a DeepSeek-OCR re-encode via the daemon
        # (icm.ocr_encoder.OcrGistBackend -- a flag, default OFF in this repo)
        self._ocr_backend = None
        if gist_backend == "ocr":
            from icm.ocr_encoder import OcrGistBackend

            self._ocr_backend = OcrGistBackend()
        from wavedb import VectorLayer, Format, IndexType, Distance, Runtime

        self._vector = VectorLayer.open(
            "icm-episodes", ledger._db,
            fmt=Format(IndexType.FLAT, dim=dim, distance=Distance.COSINE),
            rt=Runtime(top_k=10, sync_only=1),
        )
        # persistence must outlive the vector layer: run this closer before
        # the ledger's own close (ledger.register_close contract).
        ledger.register_close(self._vector.close)

    # ---- capture ----

    def capture(self, session_id: str, intent_id: str | None, text: str,
                expectation_ids: list[str] | None = None,
                episode_id: str | None = None, created_ts: float | None = None) -> str:
        episode_id = episode_id or self.ledger.new_id("ep")
        rec = {
            "episode_id": episode_id, "session_id": session_id,
            "intent_id": intent_id, "text": text, "fidelity": FIDELITY_FULL,
            "expectation_ids": expectation_ids or [],
            "created_ts": created_ts if created_ts is not None else time.time(),
            "forget_version": FORGET_RULE_VERSION,
        }
        self.ledger.put("episode", rec)
        self._vector.insert_sync(episode_id, [float(x) for x in self.embedder([text])[0]],
                                 metadata=text.encode("utf-8")[:512])
        return episode_id

    # ---- recall ----

    def recall(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        """ANN recall over the index; returns [(episode_record, similarity)]."""
        q = self.embedder([query])[0]
        results = self._vector.search_sync([float(x) for x in q], k)
        out = []
        for r in results:
            rec = self.ledger.get("episode", r.id_str)
            if rec is None or rec.get("fidelity") == "dropped":
                continue
            out.append((rec, 1.0 - float(r.distance)))
        return out

    # ---- forgetting (maintenance pass) ----

    def _violated_exp_ids(self) -> set[str]:
        violated: set[str] = set()
        for o in self.ledger.records("outcome"):
            if o["congruence"]["congruence"] == "violated":
                violated.add(o["exp_id"])
        return violated

    def maintain(self, now_ts: float | None = None) -> dict:
        """Run the icm-forget/1 pass; returns an accounting summary."""
        now_ts = now_ts if now_ts is not None else time.time()
        violated = self._violated_exp_ids()
        blurred = kept_vivid = dropped = 0
        # SNAPSHOT: delete_record mutates the registry list this iterates over;
        # iterating the live generator skips a record per drop (halving bug).
        records = list(self.ledger.records("episode"))
        for rec in records:
            eid = rec["episode_id"]
            if rec.get("fidelity") == "dropped":
                continue
            if violated & set(rec.get("expectation_ids", [])):
                kept_vivid += 1  # vivid: no blur, no drop
                continue
            age_days = (now_ts - rec["created_ts"]) / 86400.0
            if age_days <= self.blur_horizon_days:
                continue
            if rec.get("fidelity") == FIDELITY_FULL:
                gist = (self._ocr_backend.gist(rec["text"]) if self._ocr_backend
                        else gist_of(rec["text"]))
                rec["text_before_blur"] = rec["text"]
                rec["text"] = gist
                rec["fidelity"] = FIDELITY_GIST
                self.ledger.put("episode", rec)
                self._vector.insert_sync(eid, [float(x) for x in self.embedder([gist])[0]],
                                         metadata=gist.encode("utf-8")[:512])
                blurred += 1
            if age_days > self.drop_horizon_days:
                self.ledger.delete_record("episode", eid)
                self._vector.delete_sync(eid)
                dropped += 1
        return {"rule": FORGET_RULE_VERSION, "blurred": blurred,
                "kept_vivid": kept_vivid, "dropped": dropped}

    # ---- accounting ----

    def counts(self) -> dict[str, int]:
        by_fid: dict[str, int] = {}
        for rec in self.ledger.records("episode"):
            fid = rec.get("fidelity", FIDELITY_FULL)
            by_fid[fid] = by_fid.get(fid, 0) + 1
        return by_fid