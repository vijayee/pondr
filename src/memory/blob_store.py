"""Cold, content-addressed blob store for document section bodies.

A SECOND WaveDB instance (``data/document_db``) that holds the bulky section
bodies a document ingests, kept separate from the hot memory store
(``data/memory_db``, 100MB LRU) so large chunk bodies never flush the hot
cache. The split is the user's load-bearing concern (ingestion plan
``mellow-jumping-token.md`` decision 4): the memory store keeps only graph
pointers + small render-time metadata (incl. a ``blob_hash`` ref per
section); the body itself is pulled from here only when a matched section
renders into context -- cold, disk-bound, behind its own tiny LRU.

Content-addressing (coding-chat IDX 8/10): each blob is keyed by
``sha256[:16]`` of its body, so:

* **Dedup across documents.** Two documents sharing an identical section
  share ONE blob; ``put_blob`` is a no-op when the hash already exists.
* **Incremental re-ingest (Phase 2).** A hash-diff of the new section set
  against the old re-embeds only changed chunks (unchanged reuse the blob +
  its eventual embedding).
* **Shared-blob delete safety.** A blob may back more than one document, so
  ``delete_document`` does NOT physically delete a blob on a single delete --
  it decrements a refcount and leaves zero-ref blobs as orphans for
  ``gc_zero_ref`` to sweep. Physically deleting on a single delete would
  corrupt a surviving document that still references the blob.

Key layout (separate sub-namespaces so gc scans are unambiguous):

* ``blob/body/{hash}``  -- the section body (UTF-8 text).
* ``blob/refs/{hash}``  -- the refcount (integer string); PRESENCE of this key
  means "at least one document references this blob." ``decr_ref`` deletes the
  key when the count reaches zero, so ``gc_zero_ref`` scans ``blob/body/`` and
  removes any body whose ``blob/refs/{hash}`` is absent.

The store is opened lazily by ``HippocampalStore`` (mirroring the graph
layer's deferred init) and closed BEFORE the memory db in ``HippocampalStore.
close()`` (the binding's ``close()`` refuses to destroy a db while child
handles are open -- same ordering rule).
"""

import hashlib
from typing import Optional

from wavedb import WaveDB, WaveDBConfig


def blob_hash(body: str) -> str:
    """Content-addressed key for a section body (``sha256[:16]``)."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


class BlobStore:
    """Content-addressed cold store for document section bodies."""

    def __init__(self, db_path: str, lru_memory_mb: int = 16):
        cfg = WaveDBConfig(
            lru_memory_mb=lru_memory_mb,
            wal_sync_mode="debounced",
        )
        self.db = WaveDB(db_path, config=cfg)
        self.db_path = db_path

    # ---- bodies ----

    def has_blob(self, hash_: str) -> bool:
        return bool(self.db.get_sync(f"blob/body/{hash_}"))

    def put_blob(self, hash_: str, body: str) -> bool:
        """Write a body keyed by ``hash_``. Idempotent: a no-op if it exists.

        Returns ``True`` if a NEW blob was written, ``False`` if the blob was
        already present (dedup hit -- the body is shared with another document
        or a re-ingest). Does NOT touch the refcount: the caller increments the
        ref for each document that references this blob, so a dedup hit still
        gets a ref bump from the caller.
        """
        if self.has_blob(hash_):
            return False
        self.db.put_sync(f"blob/body/{hash_}", body)
        return True

    def get_blob(self, hash_: str) -> Optional[str]:
        """Return the body for ``hash_``, or ``None`` if absent."""
        raw = self.db.get_sync(f"blob/body/{hash_}")
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode("utf-8", "replace")
        return str(raw)

    # ---- refcounts ----

    def incr_ref(self, hash_: str) -> int:
        """Increment the refcount for ``hash_``; return the new count.

        A ref = "one more document references this blob." Always paired with a
        ``put_blob`` at encode time (the caller increments once per section
        that references the blob, including dedup hits).
        """
        cur = self._refcount(hash_)
        nxt = cur + 1
        self.db.put_sync(f"blob/refs/{hash_}", str(nxt))
        return nxt

    def decr_ref(self, hash_: str) -> int:
        """Decrement the refcount for ``hash_``; return the new count.

        Deletes the refcount key when it reaches zero, so ``gc_zero_ref`` can
        treat "no ref key" as "orphan." Never deletes the body -- shared-blob
        safety: a blob may back more than one document, so a single document's
        delete leaves the blob in place for any surviving referencer (or as an
        orphan for gc if none). Never returns below zero.
        """
        cur = self._refcount(hash_)
        nxt = max(0, cur - 1)
        if nxt <= 0:
            self.db.batch_sync([{"type": "del", "key": f"blob/refs/{hash_}"}])
        else:
            self.db.put_sync(f"blob/refs/{hash_}", str(nxt))
        return nxt

    def refcount(self, hash_: str) -> int:
        """Current refcount for ``hash_`` (0 if the key is absent)."""
        return self._refcount(hash_)

    def _refcount(self, hash_: str) -> int:
        raw = self.db.get_sync(f"blob/refs/{hash_}")
        if raw is None:
            return 0
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    # ---- gc ----

    def gc_zero_ref(self) -> int:
        """Physically remove bodies with no refcount key (orphans).

        Scans ``blob/body/``; any body whose ``blob/refs/{hash}`` is absent is
        an orphan (no live document references it) and is deleted along with
        any stale ref key. Returns the count of bodies removed. Safe to run
        after a batch of deletes or periodically.
        """
        removed = 0
        ops: list[dict] = []
        for k, _ in self.db.create_read_stream(
            start="blob/body/", end="blob/body/\x7f"
        ):
            # k = blob/body/{hash}
            parts = k.split("/", 2)
            if len(parts) < 3 or not parts[2]:
                continue
            hash_ = parts[2]
            if self._refcount(hash_) > 0:
                continue
            ops.append({"type": "del", "key": f"blob/body/{hash_}"})
            ops.append({"type": "del", "key": f"blob/refs/{hash_}"})
            removed += 1
        if ops:
            self.db.batch_sync(ops)
        return removed

    # ---- lifecycle ----

    def close(self) -> None:
        self.db.close()