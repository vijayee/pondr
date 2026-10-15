"""Map CodeTraceBench trajectories onto the ICM capture schema.

For each trajectory (the 32 verified SWE tars pulled, mapped via the manifest):
  goal     = the task (task_name; agent+model recorded; `solved` known)
  plan     = the AGENT'S OWN self-declared plan from its first model call
             (OpenHands-style `plan` tool-call carrying a task_list with
             phase titles) = a self-declared step-intent chain (v2-miner's
             "goal = the agent's reply", materialized)
  batch    = each manifest stage (start/end step ids — the annotator's
             hierarchical grouping)
  step     = each model call (response action + derived file accesses),
             aligned by call index (checked against step_count: +-few)
  labels   = human-verified: manifest incorrect_stages -> step labels
             (incorrect / unuseful). Third label basis alongside organic
             feedback and rule-computed congruence — carried in label_basis.
  outcome  = an END-OF-TRAJECTORY congruence record: declared desired
             "task solved"; observed = solved/other; label COMPUTED BY THE
             RULE (never stated).

Outputs:
  data/_scratch/icm/capture/codetrace_mapped.jsonl   (capture-shaped)
  data/_scratch/codetrace/codetrace_flip_table.json  (shared files -> tasks
                                                      + their labeled steps)
"""

from __future__ import annotations

import collections
import glob
import io
import json
import os
import re
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icm.congruence import evaluate_congruence  # noqa: E402

MANIFEST = os.path.join("data", "_scratch", "codetrace", "bench_manifest.verified.jsonl")
TRAJ_DIRS = [os.path.join("data", "_scratch", "codetrace", "trajs2"),
             os.path.join("data", "_scratch", "codetrace", "trajs")]
OUT = os.path.join("data", "_scratch", "icm", "capture", "codetrace_mapped.jsonl")
FLIP = os.path.join("data", "_scratch", "codetrace", "codetrace_flip_table.json")
FILE_EXT = r"\.(py|js|ts|json|md|txt|rst|yaml|yml|toml|cfg|ini|c|cpp|h|hpp|vue|ipynb|html|css)\b"


def main() -> int:
    manifest = {os.path.basename(r["artifact_path"]): r
                for r in (json.loads(l) for l in io.open(MANIFEST, encoding="utf-8") if l.strip())
                if r.get("artifact_path")}
    tars = ordered_tars()
    print(f"[load] {len(tars)} trajectory tars")
    n_records = 0
    stats = collections.Counter()
    file_to_tasks = collections.defaultdict(lambda: collections.defaultdict(list))

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        def emit(r):
            nonlocal n_records
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")
            n_records += 1

        for tar in tars:
            b = os.path.basename(tar)
            row = manifest.get(b)
            if row is None or "__" not in (row.get("task_name") or ""):
                continue  # keep the SWE slice (TerminalBench = isolated stores)
            task = row["task_name"]
            hid = f"ct-{task}-{row.get('agent', '?')[:12]}"
            emit({"kind": "icapture", "level": "goal", "parent": None,
                  "id": hid, "text": f"resolve {task} ({row.get('agent')}/{row.get('model')})",
                  "branches": {"desired": "patch resolves the task; verifier passes",
                               "feared": "patch fails tests", "default": None},
                  "declared_at": None,
                  "label_basis": "human-verified (codetrace)"})
            stats["goals"] += 1
            stage_out = None
            # hierarchy from the manifest's stages
            for st in row.get("stages") or []:
                emit({"kind": "icapture", "level": "batch", "parent": hid,
                      "id": f"{hid}-st{st['stage_id']}",
                      "text": f"stage {st['stage_id']} (steps {st['start_step_id']}-{st['end_step_id']})",
                      "declared_at": None,
                      "label_basis": "human-verified (codetrace)"})
                stats["batches"] += 1

            # parse the trajectory: the agent's plan + per-call steps
            calls = []
            plan_text = None
            try:
                t = tarfile.open(tar, "r:*")
            except Exception:
                continue
            try:
                for mem in sorted([m for m in t.getmembers()
                                   if "tensorblock" in m.name and m.name.endswith(".json")],
                                  key=lambda m: m.name):
                    try:
                        data = json.load(t.extractfile(mem))
                    except Exception:
                        continue
                    resp = data.get("response") or {}
                    if not resp.get("choices"):
                        continue
                    msg = resp["choices"][0].get("message") or {}
                    action = ""
                    for tc in (msg.get("tool_calls") or []):
                        fn = (tc.get("function") or {})
                        arguments = fn.get("arguments") or ""
                        action += " " + str(arguments)[:300]
                    if msg.get("content"):
                        action = (str(msg["content"])[:200] + " | " + action)[:400]
                    if calls == [] or plan_text is None:
                        # the agent's plan declaration (v2 rule: goal revealed
                        # by the agent's own first pursuing reply)
                        for tc in (msg.get("tool_calls") or []):
                            args_s = str(((tc.get("function") or {}).get("arguments") or ""))
                            if '"plan"' in args_s or "task_list" in args_s:
                                plan_text = args_s[:600]
                                break
                        if plan_text is None and msg.get("content"):
                            plan_text = str(msg["content"])[:600]
                    calls.append({"action": action})
            finally:
                t.close()
            if plan_text:
                emit({"kind": "icapture", "level": "plan", "parent": hid,
                      "id": f"{hid}-plan", "text": plan_text,
                      "declared_at": None,
                      "label_basis": "human-verified (codetrace)"})
                stats["plans"] += 1

            # human step-utility labels from the manifest
            step_labels = {}
            for st in row.get("incorrect_stages") or []:
                for s in st.get("steps") or []:
                    if "incorrect" in (s.get("labels") or []):
                        step_labels[s["step_id"]] = "incorrect"
                for sid in st.get("unuseful_step_ids") or []:
                    step_labels[sid] = "unuseful"

            # align calls to step ids (1-based); record steps + file accesses
            for si, call in enumerate(calls, start=1):
                act = call["action"]
                fpaths = {tok.strip(" ,'\"();").lower() for tok in
                          re.findall(r"[\w./\\-]{5,140}", act)
                          if "/" in tok and re.search(FILE_EXT, tok)
                          and not any(x in tok for x in ("tensorblock", "swe_raw", "verified"))}
                lab = step_labels.get(si)
                emit({"kind": "icapture", "level": "step", "parent": hid,
                      "id": f"{hid}-s{si:04d}", "text": act[:400],
                      "human_label": lab,
                      "step_id": si, "declared_at": None,
                      "label_basis": "human-verified (codetrace)"})
                stats["steps"] += 1
                for fp in fpaths:
                    file_to_tasks[fp][task].append({"step_id": si, "label": lab})
                    stats["file_accesses"] += 1

            # the trajectory-end outcome: rule-computed congruence
            solved = bool(row.get("solved"))
            result = evaluate_congruence(
                {"desired": "verifier passes (solved)",
                 "feared": "patch fails tests (unsolved)", "default": None},
                "desired" if solved else "feared")
            emit({"kind": "icapture", "level": "outcome",
                  "parent": hid, "id": f"{hid}-solved",
                  "observed": "desired" if solved else "feared",
                  "label": result.congruence.value,   # COMPUTED by the rule
                  "note": f"verifier: resolved={solved}",
                  "declared_at": None,
                  "label_basis": "congruence (rule-computed; reality oracle)"})
            stats["outcomes"] += 1

        # ---- the flip table: shared files -> tasks + step utilities ----
        flip = {}
        for fp, tasks in file_to_tasks.items():
            if len(tasks) >= 2:
                flip[fp] = {t: v for t, v in sorted(tasks.items(),
                                                    key=lambda kv: -len(kv[1]))}
        flip_sorted = dict(sorted(flip.items(), key=lambda kv:
                                  -sum(len(v) for v in kv[1].values())))
        with io.open(FLIP, "w", encoding="utf-8") as f:
            f.write(json.dumps({"files_shared_across_tasks": len(flip_sorted),
                                "table": flip_sorted}, indent=1))
    print(f"[mapped] {n_records} records; stats {json.dumps(stats)}")
    print(f"[flip] {len(flip_sorted)} file-units shared across tasks; -> {FLIP}")
    return 0


def ordered_tars():
    seen = []
    for d in TRAJ_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.tar.zst"))):
            seen.append(p)
    return seen


if __name__ == "__main__":
    sys.exit(main())