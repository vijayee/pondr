"""Phase 1b/1d training infrastructure.

Phase 1b (Phase G) shipped the subgraph-extraction plumbing
(``OracleLabelingPipeline``, ``sample_episode_centers``,
``ORACLE_GNN_LABELING_PROMPT``) — no live Oracle calls.

Phase 1d adds the live Oracle API client (``OracleClient``) that labels
training data via the local Ollama DeepSeek endpoint, plus the prompt library
in ``src/training/prompts.py`` and the validators in
``src/training/validators.py``.
"""

from .oracle_labeling import (
    ORACLE_GNN_LABELING_PROMPT,
    OracleClient,
    OracleConfig,
    OracleLabelingPipeline,
    OracleResult,
    sample_episode_centers,
)

__all__ = [
    "ORACLE_GNN_LABELING_PROMPT",
    "OracleClient",
    "OracleConfig",
    "OracleLabelingPipeline",
    "OracleResult",
    "sample_episode_centers",
]