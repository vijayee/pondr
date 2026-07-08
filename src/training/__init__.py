"""Phase 1b/1d training infrastructure.

Phase 1b (Phase G) shipped the subgraph-extraction plumbing
(``OracleLabelingPipeline``, ``sample_episode_centers``) — no live Oracle calls.

Phase 1d adds the live Oracle API client (``OracleClient``) that labels
training data via the local Ollama DeepSeek endpoint, plus the prompt library
in ``src/training/prompts.py`` and the validators in
``src/training/validators.py``.

Phase 3a Task 3 removed the dead single-label ``ORACLE_GNN_LABELING_PROMPT``
(it contradicted the 5-prompt library in ``prompts.py``); use the ``gnn_*``
functions there instead.
"""

from .oracle_labeling import (
    OracleClient,
    OracleConfig,
    OracleLabelingPipeline,
    OracleResult,
    sample_episode_centers,
)

__all__ = [
    "OracleClient",
    "OracleConfig",
    "OracleLabelingPipeline",
    "OracleResult",
    "sample_episode_centers",
]