"""DeepSeek (Ollama cloud relay) policy adapter for the Phase-0 agent pilot.

The adapter converts real model output into ledger declarations. One hard
boundary is enforced by construction and by WHERE the code lives:

    * this module WRITES intentions/expectations (model-authored statements),
    * it NEVER resolves them. Outcomes come from the harness's hidden world
      (icm/agent_pilot.py), so a model can never stamp evidence about its own
      prediction -- that would make the congruence labels circular.

Local-first guarantee: calls go to the local Ollama endpoint only (cloud
relays ride through the local server; the recall/ledger path never crosses
the network by another route).
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field

from icm.congruence import validate_branches

DECLARATION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "expectation": {"type": "string"},
        "branches": {
            "type": "object",
            "properties": {
                "desired": {"type": ["string", "null"]},
                "feared": {"type": ["string", "null"]},
                "default": {"type": ["string", "null"]},
            },
            "required": ["desired", "feared", "default"],
        },
    },
    "required": ["intent", "expectation", "branches"],
}

PROMPT_TEMPLATE = """You are an autonomous agent about to act. You cannot see the environment state.
Task: {task}
{prior_section}Predict the outcome branches you will be checked against:
- desired: the outcome you are trying to make happen
- feared: an outcome you want to avoid (may be null)
- default: the likely outcome if things go normally (may be null)
Bet selectively: set a branch to null unless the outcome genuinely belongs in
your prediction. A prediction that hedges every branch tells the future nothing
and cannot be distinguished from knowing nothing.
Also name your intent: a short phrase for WHY you are doing this task.
Reply ONLY as JSON with keys intent, expectation, branches. No extra keys."""


PRIOR_SECTION = """Evidence gathered so far:
{evidence}
Update your prediction in light of the evidence.
"""


@dataclass
class AgentDeclaration:
    intent: str
    statement: str
    branches: dict
    raw: dict = field(default_factory=dict, repr=False)

    def to_expectation_decl(self, intent_id: str):
        from icm.instrument import ExpectationDecl

        return ExpectationDecl(
            intent_id=intent_id, statement=self.statement,
            branches=self.branches, horizon={"kind": "manual"},
        )


class DeepSeekAgentPolicy:
    """Ollama-hosted DeepSeek emitting ledger-grade expectation declarations."""

    def __init__(self, model: str = "deepseek-v4.1-flash:cloud",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.4, num_ctx: int = 8192,
                 timeout_s: float = 180.0, retries: int = 2):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.retries = retries

    def declare(self, task: str, prior_observations: list[str] | None = None) -> AgentDeclaration:
        prompt = PROMPT_TEMPLATE.format(
            task=task,
            prior_section=(PRIOR_SECTION.format(evidence="\n".join("- " + o for o in prior_observations))
                           if prior_observations else ""),
        )
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                rec = self._chat_json(prompt)
                return self._validate(rec)
            except (json.JSONDecodeError, KeyError, ValueError, TimeoutError, OSError) as e:
                last_err = e
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"declaration failed after {self.retries + 1} attempts: {last_err!r}")

    def _chat_json(self, prompt: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": DECLARATION_SCHEMA,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read())
        return json.loads(payload["message"]["content"])

    def _validate(self, rec: dict) -> AgentDeclaration:
        intent = rec.get("intent")
        statement = rec.get("expectation")
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("intent must be a non-empty string")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError("expectation must be a non-empty string")
        branches = rec.get("branches")
        if not isinstance(branches, dict):
            raise ValueError("branches must be an object")
        validate_branches(branches)  # >=1 predicted branch; keys must be the fixed vocabulary
        return AgentDeclaration(intent=intent.strip(), statement=statement.strip(),
                                branches=branches, raw=rec)