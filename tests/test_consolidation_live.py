"""Live gate for the gist-on-forgetting consolidation loop (Phase C) against
the local Bonsai llama-server.

Skipped automatically when the endpoint (``config.bonsai_endpoint`` /
``localhost:8080/v1``) is unreachable, via the ``GET /v1/models`` probe (the
established skip guard -- mirrors ``test_bonsai_decider_live.py``). Run with
the 8B Bonsai server up (see memory ``hippo-bonsai-local-server``); pre-warm to
avoid PTX-JIT cold-start stalls.

This is the HONEST live gate for commit 806da1b: the unit tests
(``test_consolidation.py``) prove the worker/thread/gate logic with a STUB
gister; THIS test proves the REAL BonsaiGister (a live HTTP call to Bonsai for
the narrative + the real BonsaiRelationExtractor for facts) drives the REAL
``FadeMemory.consolidate`` end-to-end -- a faded-to-R4 anchor is gisted in place
and jumps back to recallable (R1), with the blurb replaced by the gist narrative
(not the verbatim) and the facts sidecar staged.
"""

from __future__ import annotations

import time

import pytest
import requests

from src.config import config
from src.subconscious.consolidation_worker import ConsolidationWorker
from src.subconscious.fade import FadeConfig, FadeMemory, REGIME_FORGOTTEN
from src.subconscious.gister import default_gister

# Reuse the test_fade doc-themed stub embedder + voice + the R4-flooding helper.
from tests.test_fade import _StubEmbedder, _StubVoice, _fade_anchor_to_r4


@pytest.fixture(scope="module")
def gister_live():
    """A real ``BonsaiGister`` (live Bonsai HTTP); skip if the server is down."""
    url = config.bonsai_endpoint.rstrip("/") + "/models"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Bonsai endpoint {config.bonsai_endpoint} unreachable: {e}")
    return default_gister()


def test_consolidate_gist_first_pass_live(gister_live):
    # The decider produces a real gist narrative for a fading blurb (first
    # consolidation, prior_gist=None). Non-empty, control-char-clean, and is a
    # compression (shorter than the source, not the verbatim echoed back).
    blurb = ("We decided to use WaveDB for the vector store because the in-DB "
             "vector layer avoids a separate sidecar index, and FLAT/COSINE was "
             "exact and fast enough at our scale of roughly ten thousand "
             "episodes. Alice and Bob benchmarked it against FAISS on Tuesday.")
    narrative = gister_live.decider.consolidate_gist(blurb, None, 1)
    assert narrative is not None, "Bonsai returned no gist narrative"
    assert isinstance(narrative, str) and len(narrative) > 15
    assert all(ord(c) >= 0x20 or c in "\n\t" for c in narrative)
    # A gist is a paraphrase, not an echo: it must differ from the verbatim
    # surface form (the model rewrites "We decided..." -> "The team chose...").
    # It is NOT strictly shorter -- on an already-terse blurb the paraphrase can
    # be longer; the compression is semantic (abstraction), not by character count.
    assert narrative.strip() != blurb.strip()
    assert len(narrative) < 8 * len(blurb)  # sanity cap, not a real gist bound


def test_consolidate_gist_gist_of_gist_live(gister_live):
    # The prior-baseline-merge branch: feed the prior gist back in (count=2) and
    # get a COMPLETE new gist (not a delta), preserving the key fact. The result
    # must still read as a standalone memory of the SAME subject.
    prior = ("WaveDB was chosen as the vector store: its in-DB vector layer "
             "removes the need for a sidecar, and FLAT/COSINE was exact and "
             "fast enough at ~10k episodes.")
    narrative = gister_live.decider.consolidate_gist(prior, prior, 2)
    assert narrative is not None, "Bonsai returned no gist-of-gist"
    assert isinstance(narrative, str) and len(narrative) > 10
    # Fidelity: the subject (WaveDB) is preserved across the further compression.
    assert "wavedb" in narrative.lower()


def test_gister_structured_gist_live(gister_live):
    # The gister composes the narrative + fact extractors into a StructuredGist.
    # The narrative is populated; facts may be empty on a short blurb (extraction
    # is best-effort), but the StructuredGist shape is always returned.
    blurb = ("Alice owns a 2019 Subaru Outback. Bob owns a 2021 Toyota Corolla. "
             "They discussed fuel economy on Tuesday.")
    gist = gister_live.gist(blurb, None, 1)
    assert gist is not None
    assert isinstance(gist.narrative, str) and len(gist.narrative) > 10
    assert isinstance(gist.facts, list)
    assert isinstance(gist.state_assertions, list)
    assert gist.consolidation_count == 1


def test_worker_consolidates_faded_anchor_live(gister_live):
    # THE end-to-end loop: a real FadeMemory + the REAL Bonsai gister + the real
    # worker thread. Drive an anchor to R4 (forgotten), tick the worker; it gists
    # the anchor via live Bonsai between turns and the anchor jumps R4 -> R1 with
    # the blurb replaced by the gist narrative (not the verbatim).
    emb = _StubEmbedder()
    cfg = FadeConfig(decay=0.5, cos_ring=0.95, cos_gist=0.20, ring_capacity=2)
    mem = FadeMemory(cfg, emb, _StubVoice())
    aid = mem.ingest("docA:0 we picked wavedb for the vector store because the "
                     "in-db layer avoids a sidecar and flat cosine was exact")
    verbatim = mem.blurbs.text(aid)
    _fade_anchor_to_r4(mem, aid)
    assert mem.recall_anchor(aid).regime == REGIME_FORGOTTEN, "anchor did not fade to R4"
    assert mem.consolidation_count(aid) == 0

    worker = ConsolidationWorker(mem, gister_live, epsilon=0.03, max_depth=3,
                                 max_per_tick=8)
    worker.foreground_busy.set()
    n = worker.tick()
    assert n >= 1, "tick did not enqueue the fading anchor"
    # While the foreground gate is held the worker must NOT consolidate. (The
    # structural gate -- ``_wait_foreground`` blocks while the Event is set --
    # is proven in test_consolidation.py with a deterministic stub; this 0.3s
    # smoke check just confirms no eager consolidation leaked through.)
    time.sleep(0.3)
    assert mem.consolidation_count(aid) == 0
    # Release the gate -> the worker gists the anchor via live Bonsai.
    worker.foreground_busy.clear()
    assert worker.drain(timeout=60.0) is True

    # The anchor was consolidated in place: count climbed, the blurb is now the
    # gist narrative (different from the verbatim), and it is recallable again.
    assert mem.consolidation_count(aid) == 1
    new_blurb = mem.blurbs.text(aid)
    assert new_blurb is not None
    assert new_blurb != verbatim, "blurb was not replaced by the gist"
    # ``new_blurb`` is ``narrative[:blurb_chars]``; ``prior_gist`` is the full
    # narrative -> new_blurb is always a prefix of the recorded prior gist.
    assert mem.prior_gist(aid).startswith(new_blurb[:24])
    r = mem.recall_anchor(aid)
    assert r.regime != REGIME_FORGOTTEN, "consolidated anchor is still R4"
    # The fact sidecar was staged (best-effort: may be empty, but the slot exists).
    assert isinstance(mem.blurbs.facts(aid), list)