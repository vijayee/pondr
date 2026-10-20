"""E5-LITE -- does the system improve from its own outcomes over episodes?

PRE-REGISTERED DESIGN (written before the run; prepared fixture, not an organic
claim):

WORLD: six users, 24 episodes each. Each user has a hidden convention among the
four layer-3 verification rules (header / excluded file / name-hash / duplicate
rows). The convention is stable for 8 episodes and then CHANGES at unknown
points (episodes 8 and 16). Every episode is a fresh dataset; the manifest is
built from the task + metadata under the system's current per-user policy. The
verifier is mechanical and reports all mismatches, naming the defect.

ARMS:
  ADAPTIVE          per-user counts over conventions; on a violation, decay all
                    counts (0.2) and increment the convention the failure reason
                    identifies; on success, reinforce the chosen convention.
  ACCUMULATION-ONLY same counts, NO decay: the outcomes are stored but recency
                    is not used (the plan's "accumulation alone" control).
  NO-LEDGER         a uniformly random convention each episode (no memory).

  Both learning arms construct the manifest deterministically from the chosen
  convention + the episode metadata. No model calls: the policy prior is the
  memory, and the verifier is the oracle.

METRICS: first-attempt reach by episode; post-first reach within each 8-episode
regime (episodes 1-7, 9-15, 17-23); recovery steps after each convention change
(the first success at/after the change episode); the regime-change violation
rate; the capture's expansion-link conformance (every post-violation goal's
`triggered_by` names the violated outcome).

PRE-REGISTERED BAR:
  (1) POWER:       6 users x 24 episodes = 144; post-first = 126
  (2) ADAPTIVE     post-first reach >= 0.85
  (3) NO-LEDGER    post-first reach <= 0.40
  (4) LIFT         adaptive - accumulation >= 0.25
  (5) RECOVERY     adaptive recovers within 2 episodes on >= 90% of the 12
                   convention changes, and mean recovery <= 2.0
  (6) NON-REACH    >= 90% of the adaptive regime-change episodes are violations
  (7) LINK         100% of post-violation goals carry triggered_by = the
                   violated outcome
  Report the per-episode curve and per-user table. PASS requires all.

SCOPE: this tests the expectation-driven policy-adaptation mechanism in a
prepared world; it is not E5's full longitudinal user study and not organic.
"""

from __future__ import annotations

import io
import json
import os
import random
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from icm.capture import IntentCapture  # noqa: E402
from icm.layer3_env import (DUPLICATE_ROWS, EXCLUDED_FILE, RULES,  # noqa: E402
                            Layer3Env)
from icm.ledger import ICMLedger  # noqa: E402

OUTDIR = os.path.join(REPO_ROOT, "data", "_scratch", "icm", "e5")
RESULTS = os.path.join(OUTDIR, "results.json")
LEDGER_DB = os.path.join(OUTDIR, "ledger")
MIRROR = os.path.join(OUTDIR, "capture")

USERS = 6
EPISODES = 24
REGIME = 8
DECAY = 0.2
SEED = 17


def schedule(user: int) -> list[str]:
    start = user % len(RULES)
    return [RULES[(start + e // REGIME) % len(RULES)] for e in range(EPISODES)]


def spec_for(rule: str, meta: dict) -> dict:
    total = meta["total_lines"]
    lc = meta["line_counts"]
    if rule == "header":
        return {"files": 4, "rows": total - 1, "hash_mode": "content"}
    if rule == "exclude":
        return {"files": 3,
                "rows": total - lc[meta["files"].index(EXCLUDED_FILE)],
                "hash_mode": "content"}
    if rule == "name_hash":
        return {"files": 4, "rows": total, "hash_mode": "content+names"}
    return {"files": 4, "rows": total - DUPLICATE_ROWS, "hash_mode": "content"}


def observed_rule(reason: str):
    r = reason.lower()
    if "header" in r:
        return "header"
    if "exclud" in r or "skip" in r:
        return "exclude"
    if "duplicate" in r:
        return "dedupe"
    if "content+names" in r:
        return "name_hash"
    return None


def choose(counts: dict) -> str:
    return max(RULES, key=lambda r: counts[r])


def run_arm(arm: str, env: Layer3Env, ledger=None) -> list[dict]:
    rng = random.Random(SEED)
    counts = {u: {r: 1.0 for r in RULES} for u in range(USERS)}
    rows = []
    for u in range(USERS):
        cap = None
        prev_outcome = None
        prev_label = None
        if arm == "adaptive" and ledger is not None:
            cap = IntentCapture(ledger, mirror_dir=MIRROR)
        for e, rule in enumerate(schedule(u)):
            sid = f"{arm}-u{u}-e{e:02d}"
            meta = env.materialize(sid, rule)
            chosen = (choose(counts[u]) if arm != "no_ledger"
                      else rng.choice(RULES))
            spec = spec_for(chosen, meta)
            check = env.check(sid, spec)
            ok = bool(check["ok"])
            if cap is not None:
                g = cap.goal(
                    f"use the {chosen} convention",
                    desired="verification passes", epoch=e,
                    triggered_by=prev_outcome if prev_label == "violated" else None)
                s = cap.step(g, f"episode {e}: {json.dumps(spec)[:300]}")
                r = cap.retrieval(s, "verify manifest", [])
                o = cap.outcome(r, "desired" if ok else "feared")
                prev_outcome, prev_label = o, ("confirmed" if ok else "violated")
            if ok:
                if arm != "no_ledger":
                    counts[u][chosen] += 1.0
            else:
                observed = observed_rule(check["reason"])
                if arm == "adaptive":
                    for rr in RULES:
                        counts[u][rr] *= DECAY
                    if observed:
                        counts[u][observed] += 1.0
                    else:
                        counts[u][chosen] -= 0.5
                elif arm == "accumulation":
                    if observed:
                        counts[u][observed] += 1.0
                    else:
                        counts[u][chosen] -= 0.5
            rows.append({"arm": arm, "user": u, "episode": e, "rule": rule,
                         "chosen": chosen, "ok": ok,
                         "reason": check["reason"],
                         "counts": {r: round(c, 3) for r, c in counts[u].items()}})
        if cap is not None:
            cap.close()
    return rows


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    env = Layer3Env(os.path.join(OUTDIR, "world"), seed=SEED)
    ledger = ICMLedger(LEDGER_DB)
    try:
        data = {arm: run_arm(arm, env, ledger if arm == "adaptive" else None)
                for arm in ("adaptive", "accumulation", "no_ledger")}
    finally:
        ledger.close()

    def post_first(rows):
        return [r for r in rows if r["episode"] % REGIME != 0]

    rates = {arm: round(statistics.mean(r["ok"] for r in post_first(rows)), 3)
             for arm, rows in data.items()}
    all_rates = {arm: round(statistics.mean(r["ok"] for r in rows), 3)
                 for arm, rows in data.items()}

    # recovery after each convention change (episodes 8 and 16)
    recoveries = {arm: [] for arm in data}
    for arm, rows in data.items():
        by_user = {}
        for r in rows:
            by_user.setdefault(r["user"], {})[r["episode"]] = r["ok"]
        for u, ok_by_ep in by_user.items():
            for change in (8, 16):
                found = None
                for e in range(change, EPISODES):
                    if ok_by_ep.get(e):
                        found = e - change
                        break
                recoveries[arm].append(found)
    rec = {}
    for arm, vals in recoveries.items():
        known = [v for v in vals if v is not None]
        rec[arm] = {
            "mean": round(statistics.mean(known), 3) if known else None,
            "within2": round(sum(1 for v in known if v <= 2) / len(vals), 3)
            if vals else None,
            "unrecovered": sum(1 for v in vals if v is None),
        }
    change_fail = round(statistics.mean(
        not r["ok"] for r in data["adaptive"] if r["episode"] in (8, 16)), 3)

    # link conformance from the capture ledger: every triggered_by goal's parent
    # outcome must carry the rule-computed violated label
    linked = total_linked = 0
    ledger2 = ICMLedger(LEDGER_DB)
    try:
        for recd in ledger2.records("icapture"):
            if recd.get("level") == "goal" and recd.get("triggered_by"):
                total_linked += 1
                parent = ledger2.get("icapture", recd["triggered_by"])
                if parent and parent.get("label") == "violated":
                    linked += 1
    finally:
        ledger2.close()
    link_rate = round(linked / total_linked, 3) if total_linked else None

    checks = {
        "power_144_episodes": len(data["adaptive"]) == USERS * EPISODES,
        "post_first_126": len(post_first(data["adaptive"])) == 126,
        "adaptive_ge_0.85": rates["adaptive"] >= 0.85,
        "no_ledger_le_0.40": rates["no_ledger"] <= 0.40,
        "lift_ge_0.25": rates["adaptive"] - rates["accumulation"] >= 0.25,
        "recovery_within2_ge_0.90": (rec["adaptive"]["within2"] or 0) >= 0.90,
        "recovery_mean_le_2": (rec["adaptive"]["mean"] or 99) <= 2.0,
        "change_nonreach_ge_0.90": change_fail >= 0.90,
        "link_rate_1.0": link_rate == 1.0,
    }
    res = {
        "users": USERS, "episodes": EPISODES, "regime": REGIME, "decay": DECAY,
        "post_first_rates": rates, "all_rates": all_rates,
        "recovery": rec, "adaptive_change_violation_rate": change_fail,
        "expansion_link_rate": link_rate, "checks": checks,
        "pass": all(checks.values()),
        "verdict": ("PASS -- expectation-driven adaptation improves reach and "
                    "recovers after convention changes" if all(checks.values())
                    else "FAIL -- see the failing row"),
        "scope": "prepared world; policy prior is the memory; no model calls",
    }
    with io.open(RESULTS, "w", encoding="utf-8") as f:
        f.write(json.dumps({**res, "rows": data}, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str), flush=True)
    print(f"[done] -> {RESULTS}", flush=True)
    return 0 if res["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
