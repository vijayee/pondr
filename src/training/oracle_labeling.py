"""Oracle labeling infrastructure for Phase 1d GNN training (infra only).

Phase 1b Phase G builds the *plumbing* for oracle-labeled subgraph training
examples — no live oracle (Bonsai) calls happen here. The pipeline:

1. ``extract_subgraph(center_id, radius=3)`` — BFS over the memory graph from a
   center node (typically an episode id), following node-to-node predicates
   (``has_entity`` / ``has_topic`` / ``has_tone`` / ``has_decision`` /
   ``in_episode`` / ``has_session`` / ``in_session`` / ``follows`` /
   ``follows_session``) in BOTH directions, up to ``radius`` hops. Returns
   ``{"nodes": [...], "edges": [...]}`` with node types inferred from the id
   prefix (``E:``/``T:``/``A:``/``D:``/``S:``/``U:``/``ep_``).
2. The 5 labeling prompts (salience / cluster / link_prediction / anomaly /
   ontology) live in ``src/training/prompts.py`` and are called by the runner;
   they are NOT defined here. (A single-label ``ORACLE_GNN_LABELING_PROMPT`` used
   to live here but was dead — it contradicted the 5-prompt library the generator
   actually uses — and was removed in Phase 3a Task 3.)

``scripts/generate_gnn_training_data.py`` is the runner that extracts subgraphs
for a sample of episodes and labels them with that 5-prompt library (Phase 1d).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import requests

from ..config import config as _config
from ..memory.store import HippocampalStore

# Node-to-node predicates traversed by the BFS. Literal-valued predicates
# (at_time, state, validity_start, ended_at) are excluded — their objects are
# timestamps/strings, not graph nodes, so they'd inject non-node "neighbors".
_NODE_PREDICATES = (
    "has_entity",
    "has_topic",
    "has_tone",
    "has_decision",
    "in_episode",
    "has_session",
    "in_session",
    "follows",
    "follows_session",
)

# Prompt for the oracle (local Bonsai) to label a subgraph for GNN training.
# REMOVED in Phase 3a Task 3: the single-label ``ORACLE_GNN_LABELING_PROMPT``
# that lived here was dead — it contradicted the 5-prompt library in
# ``src/training/prompts.py`` (gnn_salience/cluster/link_prediction/anomaly/
# ontology) that the generator actually uses. The 5-prompt library is the
# source of truth. See ``build_labeling_prompt`` removal below too.


def _node_type(node_id: str) -> str:
    """Infer the node type from its id prefix."""
    for prefix, kind in (
        ("E:", "entity"), ("T:", "topic"), ("A:", "tone"),
        ("D:", "decision"), ("S:", "session"), ("U:", "user"),
    ):
        if node_id.startswith(prefix):
            return kind
    if node_id.startswith("ep_"):
        return "episode"
    return "unknown"


class OracleLabelingPipeline:
    """Extracts subgraphs for oracle labeling (Phase 1d GNN training prep)."""

    def __init__(self, store: HippocampalStore) -> None:
        self.store = store
        self.graph = store.graph

    def _get_neighbors(self, node_id: str) -> list[tuple[str, str, str]]:
        """Return ``(neighbor_id, predicate, direction)`` for all node-to-node
        edges incident to ``node_id``. direction is "out" (node→neighbor) or
        "in" (neighbor→node) — recorded so the caller can orient the edge.
        """
        out: list[tuple[str, str, str]] = []
        for pred in _NODE_PREDICATES:
            for direction, query in (
                ("out", self.graph.query().vertex(node_id).out(pred)),
                ("in", self.graph.query().vertex(node_id).in_(pred)),
            ):
                result = query.execute_sync()
                try:
                    neighbors = list(result.vertices)
                finally:
                    result.close()
                for nb in neighbors:
                    out.append((nb, pred, direction))
        return out

    def extract_subgraph(self, center_id: str, radius: int = 3) -> dict:
        """BFS from ``center_id`` up to ``radius`` hops over node-to-node edges.

        Returns ``{"center": center_id, "radius": radius, "nodes": [...],
        "edges": [...]}`` where each node is ``{"id", "type", "depth"}`` (depth
        = hop distance from the center) and each edge is ``{"subject",
        "predicate", "object"}`` with directions normalized so subject→object
        matches the stored triple orientation.
        """
        nodes: dict[str, dict] = {
            center_id: {"id": center_id, "type": _node_type(center_id), "depth": 0}
        }
        edges: dict[tuple[str, str, str], None] = {}  # ordered dedup set
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(center_id, 0)])

        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            if depth >= radius:
                continue
            for nb, pred, direction in self._get_neighbors(node_id):
                # Orient the edge so subject→object reflects stored direction:
                # "out" means node_id → nb; "in" means nb → node_id.
                if direction == "out":
                    edges[(node_id, pred, nb)] = None
                else:
                    edges[(nb, pred, node_id)] = None
                if nb not in nodes:
                    nodes[nb] = {"id": nb, "type": _node_type(nb), "depth": depth + 1}
                if nb not in visited:
                    queue.append((nb, depth + 1))

        return {
            "center": center_id,
            "radius": radius,
            "nodes": list(nodes.values()),
            "edges": [{"subject": s, "predicate": p, "object": o}
                      for (s, p, o) in edges],
        }

    # NOTE: ``build_labeling_prompt`` was removed in Phase 3a Task 3 along with
    # the dead ``ORACLE_GNN_LABELING_PROMPT``. The live labeling prompts are the
    # 5 functions in ``src/training/prompts.py``; the generator calls those
    # directly with the subgraph JSON from ``extract_subgraph``.


def sample_episode_centers(store: HippocampalStore, n: Optional[int] = None) -> list[str]:
    """Return up to ``n`` episode ids to use as subgraph centers (all if None)."""
    ids: set[str] = set()
    for k, _ in store.db.create_read_stream(start="content/ep/", end="content/ep/\x7f"):
        parts = k.split("/", 3)
        if len(parts) >= 3 and parts[2]:
            ids.add(parts[2])
    centers = sorted(ids)
    return centers if n is None else centers[:n]


# ═══════════════════════════════════════════════════════════════
# Phase 1d: Oracle API client (DeepSeek via local Ollama)
# ═══════════════════════════════════════════════════════════════

# Matches a ```json ... ``` (or bare ```) fenced block. The Oracle is told to
# return ONLY JSON, but cloud models sometimes wrap output in fences anyway.
# Same pattern as src/encoding/bonsai_relations.py:_FENCE_RE.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass
class OracleConfig:
    """Configuration for Oracle API calls.

    Defaults are pulled from the Hippo ``Config`` (``oracle_*`` fields, env-
    overridden) when the client is constructed with ``config=None``; the
    dataclass is also instantiable directly for tests/scale runs. ``$`` cost is
    left at 0 for the local/:cloud Ollama backend — set ``cost_per_1k_*`` to
    meter Ollama-cloud credits if desired.
    """
    model: str = _config.oracle_model
    endpoint: str = _config.oracle_endpoint
    temperature: float = _config.oracle_temperature
    max_tokens: int = _config.oracle_max_tokens
    max_retries: int = _config.oracle_max_retries
    retry_delay: float = _config.oracle_retry_delay
    batch_delay: float = _config.oracle_batch_delay
    timeout: float = _config.oracle_timeout
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    cache_path: Optional[Path] = None  # on-disk prompt-hash cache for resume


@dataclass
class OracleResult:
    """Result of one Oracle API call (or a cache hit)."""
    prompt: str
    response: dict
    input_tokens: int
    output_tokens: int
    cost: float
    latency_seconds: float
    retries: int
    cached: bool = False


class OracleClient:
    """Calls the Oracle (DeepSeek via local Ollama) with caching + retries.

    Mirrors ``src/encoding/bonsai_relations.py: BonsaiRelationExtractor``:
    ``requests``-based (no ``openai`` SDK dependency), OpenAI-compatible
    ``/chat/completions`` at ``config.oracle_endpoint``, ``response_format:
    json_object``, fence-strip + outermost-span + balanced-object salvage
    parse. Adds: retry with exponential backoff, a prompt-hash on-disk cache
    (so resuming a generation run doesn't re-pay for identical prompts), and
    token/cost tracking.

    The HTTP layer is isolated as ``_post`` so tests stub it without
    monkeypatching ``requests``. The connection is lazy — the class is
    constructible offline and the module imports without the server present.
    """

    def __init__(self, config: Optional[OracleConfig] = None) -> None:
        self.config = config or OracleConfig()
        self.endpoint = self.config.endpoint.rstrip("/")
        # In-memory prompt-hash -> OracleResult. Hydrated from cache_path on
        # init so a resumed run skips already-labeled prompts.
        self.cache: dict[str, OracleResult] = {}
        self.total_cost = 0.0
        self.total_calls = 0
        self.total_tokens = 0
        self._cache_dirty = False
        # Guards the in-memory cache dict and the total_* stat counters, which
        # are mutated from inside ``generate``. Concurrent batch generation
        # (``generate_batch(max_workers>1)``) calls ``generate`` from several
        # worker threads; the HTTP call itself runs OUTSIDE this lock so the
        # calls actually parallelize — only the cache lookup/insert and the
        # counter bumps are serialized. Default ``max_workers=1`` leaves the
        # original sequential behavior byte-identical (the lock is uncontended).
        self._lock = threading.Lock()
        if self.config.cache_path:
            self._load_cache()

    # ── on-disk cache ──

    def _load_cache(self) -> None:
        path = self.config.cache_path
        if not path or not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        for prompt_hash, rec in raw.items():
            try:
                self.cache[prompt_hash] = OracleResult(
                    prompt=rec["prompt"],
                    response=rec["response"],
                    input_tokens=rec["input_tokens"],
                    output_tokens=rec["output_tokens"],
                    cost=rec["cost"],
                    latency_seconds=rec["latency_seconds"],
                    retries=rec["retries"],
                    cached=True,
                )
            except (KeyError, TypeError, ValueError):
                continue

    def flush_cache(self) -> None:
        """Persist the in-memory cache to ``cache_path`` (atomic temp write).

        Called by generators at each checkpoint and at the end of a batch.
        No-op when ``cache_path`` is unset or the cache is clean.
        """
        path = self.config.cache_path
        if not path or not self._cache_dirty:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # Persist EVERY record (not just non-cached ones): ``tmp.replace`` swaps
        # the whole file, so filtering out cache-loaded records would drop them
        # from disk and defeat cross-run resume. The ``cached`` flag is an
        # in-memory marker only; ``_load_cache`` resets it on the next run.
        # Snapshot the payload under the lock so concurrent ``generate`` calls
        # can't mutate the dict mid-serialization; the disk write itself runs
        # outside the lock (brief, and holds no in-memory state).
        with self._lock:
            payload = {
                h: {
                    "prompt": r.prompt,
                    "response": r.response,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost": r.cost,
                    "latency_seconds": r.latency_seconds,
                    "retries": r.retries,
                }
                for h, r in self.cache.items()
            }
            self._cache_dirty = False
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp.replace(path)

    # ── HTTP layer (isolated for tests) ──

    def _post(self, payload: dict) -> tuple[int, dict]:
        """POST ``payload`` to ``{endpoint}/chat/completions``; return ``(status, outer)``.

        Raises ``requests.RequestException`` on connection failure (retryable).
        Raises ``ValueError`` if the response body is not JSON (retryable as a
        transient server hiccup). Override in tests to stub the HTTP layer.
        """
        url = f"{self.endpoint}/chat/completions"
        resp = requests.post(url, json=payload, timeout=self.config.timeout)
        try:
            outer = resp.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"Oracle returned non-JSON body: {resp.text}") from e
        return resp.status_code, outer

    # ── public API ──

    def generate(self, prompt: str,
                 response_format: str = "json_object") -> OracleResult:
        """Call the Oracle with caching + retry; return an ``OracleResult``.

        Caching: identical prompts return the cached result with
        ``cached=True`` (no HTTP call). Retries: up to ``max_retries`` with
        exponential backoff on connection errors, non-200 HTTP, or non-JSON
        bodies. A 200 with an unparseable JSON payload is NOT retried — the
        model returned a response, just not valid JSON, and the caller needs
        the raw content to debug it (mirrors BonsaiRelationExtractor).
        """
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        with self._lock:
            cached = self.cache.get(prompt_hash)
        if cached is not None:
            cached.cached = True
            return cached

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": response_format},
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        last_error: Optional[str] = None
        for attempt in range(self.config.max_retries):
            try:
                start = time.time()
                status, outer = self._post(payload)
                elapsed = time.time() - start

                if status != 200:
                    body_text = json.dumps(outer) if isinstance(outer, dict) else str(outer)
                    last_error = f"HTTP {status}: {body_text}"
                    # 4xx (except 429) is a bad request — retrying won't fix it.
                    if 400 <= status < 500 and status != 429:
                        raise RuntimeError(
                            f"Oracle endpoint {self.endpoint}/chat/completions "
                            f"returned {last_error}"
                        )
                    raise _Retryable(last_error)

                content, input_tokens, output_tokens = self._extract(outer)
                try:
                    data = self._parse_json(content)
                except RuntimeError:
                    # A parse failure with output at the max_tokens cap means the
                    # response was truncated mid-JSON — not a malformed response
                    # the caller needs to debug. Surface that clearly so the fix
                    # (raise --oracle-max-tokens) is obvious.
                    if output_tokens >= self.config.max_tokens:
                        raise RuntimeError(
                            f"Oracle response truncated at max_tokens="
                            f"{self.config.max_tokens} (output_tokens="
                            f"{output_tokens}); raise --oracle-max-tokens. "
                            f"Raw tail: ...{content[-200:]!r}"
                        ) from None
                    raise
                cost = (
                    input_tokens / 1000 * self.config.cost_per_1k_input
                    + output_tokens / 1000 * self.config.cost_per_1k_output
                )
                result = OracleResult(
                    prompt=prompt,
                    response=data,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    latency_seconds=elapsed,
                    retries=attempt,
                )
                with self._lock:
                    # Cache insert + stat bumps are serialized; the HTTP call
                    # above ran outside the lock so concurrent workers
                    # parallelize on the network, not on this critical section.
                    self.cache[prompt_hash] = result
                    self._cache_dirty = True
                    self.total_cost += cost
                    self.total_calls += 1
                    self.total_tokens += input_tokens + output_tokens
                return result

            except RuntimeError:
                # Non-retryable (4xx, unparseable 200, malformed response) —
                # surface verbatim so the caller can debug the raw output.
                raise
            except (requests.RequestException, ValueError, _Retryable) as e:
                # Retryable: connection error, non-JSON body, 5xx, or 429.
                last_error = last_error or f"{type(e).__name__}: {e}"
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                break  # exhausted retries — fall through to the final RuntimeError

        raise RuntimeError(
            f"Oracle call failed after {self.config.max_retries} retries: {last_error}"
        )

    def generate_batch(
        self,
        prompts: list[str],
        response_format: str = "json_object",
        progress_callback: Optional[Callable[[int, int, OracleResult], None]] = None,
        max_workers: int = 1,
    ) -> list[OracleResult]:
        """Generate results for a batch of prompts, preserving input order.

        ``max_workers=1`` (default) is the original sequential, rate-limited path
        — one ``generate`` call at a time, sleeping ``config.batch_delay`` between
        calls. ``max_workers>1`` dispatches the calls across a
        ``ThreadPoolExecutor`` so the network-bound Oracle calls overlap; results
        are returned in the SAME order as ``prompts``. The cache dict and the
        ``total_*`` counters are guarded by an internal lock (see ``generate``),
        so concurrent workers only parallelize the HTTP waits, not the shared
        state. ``progress_callback`` fires as each call completes (in completion
        order, not input order); the cache is flushed once at the end.

        ``batch_delay`` is only honored on the sequential path — with concurrent
        workers there is no single "between calls" gap to sleep.
        """
        if max_workers <= 1:
            results: list[OracleResult] = []
            try:
                for i, prompt in enumerate(prompts):
                    result = self.generate(prompt, response_format)
                    results.append(result)
                    if progress_callback:
                        progress_callback(i + 1, len(prompts), result)
                    if i < len(prompts) - 1 and self.config.batch_delay > 0:
                        time.sleep(self.config.batch_delay)
            finally:
                self.flush_cache()
            return results

        # Concurrent path. Submit all calls, then collect via ``as_completed`` in
        # THIS thread (no cross-thread counter mutation): each result is placed
        # at its input index so the returned list matches ``prompts`` order.
        out: list[Optional[OracleResult]] = [None] * len(prompts)
        done = 0
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(self.generate, p, response_format): idx
                          for idx, p in enumerate(prompts)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    res = fut.result()
                    out[idx] = res
                    done += 1
                    if progress_callback:
                        progress_callback(done, len(prompts), res)
        finally:
            self.flush_cache()
        return [r for r in out if r is not None]  # type: ignore[list-item]

    def get_stats(self) -> dict:
        """Usage statistics for the run."""
        return {
            "total_calls": self.total_calls,
            "cached_calls": sum(1 for r in self.cache.values() if r.cached),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "cache_size": len(self.cache),
        }

    # ── helpers ──

    @staticmethod
    def _extract(outer: dict) -> tuple[str, int, int]:
        """Pull ``(content, input_tokens, output_tokens)`` from the chat response."""
        try:
            content = outer["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"Oracle response missing choices[0].message.content: {outer}"
            ) from e
        usage = outer.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        return content, input_tokens, output_tokens

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Parse the model's JSON content into a dict.

        Strips accidental ``` fences, then falls back to the outermost ``{...}``
        span (handles leading/trailing prose). Raises ``RuntimeError`` with the
        raw content when neither recovers a dict.

        Deliberately does NOT salvage a nested inner object: the Oracle payload
        is ONE object (unlike Bonsai's list-of-relation-objects, whose salvage
        scanner that pattern was ported from). Under ``response_format:
        json_object`` the model emits a single object, so a parse failure means
        the response was truncated — and recovering a small inner fragment
        (e.g. one ``extracted_relations`` entry) would silently return a
        wrong label. A loud failure lets the caller raise ``max_tokens``.
        """
        body = (content or "").strip()
        fence = _FENCE_RE.match(body)
        if fence:
            body = fence.group(1).strip()
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        # Fallback: outermost {...} span (handles trailing/leading prose). Safe
        # under json_object mode — the whole outermost {...} IS the payload.
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(body[start : end + 1])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"Oracle returned unparseable JSON: {content!r}") from None


class _Retryable(Exception):
    """Internal sentinel: retryable Oracle failure (connection / 5xx / 429)."""
    pass