"""Scene-block rendering for the LLM SYSTEM prompt (A3 cache-stable split).

Scene blocks (B1, ``--scene-blocks``) are the session-stable, LLM-authored
topic-level macro-memory stored IN WaveDB. They ride the SAME retrieval pipeline
as episodes/docs (one candidate set, hydrated as ``kind == "scene"``), but at
the PRESENTATION layer they are macro memory -- a stable "here's what we know
about this topic" frame -- not per-query chunkable evidence. So the orchestrator
pulls them out of the episode set before chunking and renders them into the
system-prompt SUFFIX (cache-friendly: stable across on-topic turns) instead of
the user-message context blob.

This module mirrors ``format_fade_block`` (``fade.py``): a pure string
formatter, no LLM/HTTP/file IO, returns ``""`` when there is nothing to render
so the caller's no-append path stays byte-identical to flag-off.
"""

from __future__ import annotations


def format_scene_block(scenes: list[dict]) -> str:
    """Render scene blocks as a ``[SCENE MEMORY]`` block for the LLM SYSTEM prompt.

    Takes the hydrated scene dict shape the retriever's ``_hydrate_scene`` builds
    (``graph_traversal.py``: ``{episode_id, kind, summary=topic, text=body,
    timestamp, topics, heat, source_eps, ...}``). Each scene renders as a
    ``--- Scene {id} (topic: ..., heat: ...) ---`` head, an optional ``Topics:``
    line, then the Markdown body.

    Returns ``""`` when no scenes are present (no empty header -> the
    orchestrator's system-prompt-suffix append is skipped entirely ->
    byte-identical when ``--scene-blocks`` is off or no scenes were retrieved).
    Mirrors ``format_fade_block`` (``fade.py:87``). Best-effort by construction:
    a malformed scene dict yields empty fields, never a raise (the orchestrator
    seam also wraps the call in a swallow).
    """
    lines: list[str] = []
    for sc in scenes or []:
        eid = sc.get("episode_id", "")
        topic = sc.get("summary", "")
        heat = float(sc.get("heat") or 0.0)
        body = (sc.get("text") or "").strip()
        topics = [t for t in sc.get("topics", []) if t]
        lines.append(f"--- Scene {eid} (topic: {topic}, heat: {heat:.2f}) ---")
        if topics:
            lines.append(f"Topics: {', '.join(topics)}")
        if body:
            lines.append(body)
    if not lines:
        return ""
    header = ("[SCENE MEMORY -- your synthesized topic-level memory of this "
              "conversation]\n(stable macro memory of a topic; persists across "
              "turns. Treat as background context.)")
    return header + "\n" + "\n\n".join(lines)