"""Replay buffer for delayed-reward gate training (framework, Phase 2b+).

Gate decisions at time t are validated by outcomes at time t+k. The replay
buffer bridges that gap without BPTT across many steps. Phase 2a ships the
buffer infrastructure; it is not exercised until gates are trained (2b+).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ReplayEntry:
    state: Any                       # the recurrent state snapshot
    predicted_outcome: Any           # JEPA prediction at decision time
    context: Any                     # gate context features
    decision: Any                    # the GateDecision made
    outcome: Optional[Any] = None    # filled in later (reward, effort)
    filled: bool = False


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer: deque[ReplayEntry] = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, entry: ReplayEntry) -> None:
        self.buffer.append(entry)

    def sample(self, batch_size: int = 32) -> list[ReplayEntry]:
        n = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def __len__(self) -> int:
        return len(self.buffer)