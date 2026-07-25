"""One-off: persist Stage 3 artifacts to the PRIVATE vijayee/pondr-models repo.

Uploads ONLY model weights + the aggregate (UUID-free) 6-seed gate summary.
Does NOT upload onyx-derived traces, the doc corpus store, per-session JSONs,
or serve logs -- those carry private conversational data (binding constraint:
do not save onyx data to HF unsanitized). Scratch; not committed.
"""
import os, sys
from huggingface_hub import HfApi, upload_file

REPO = "vijayee/pondr-models"
SUB = "strm_1f7_stage3"
BB = "data/training/strm_backbone_relevance"
RD = "data/training/strm_state_readout/head_to_head_onyx"
GATE = "data/training/strm_relevance/serve_gate_1f7_stage1_summary.json"

api = HfApi()
print("authed as:", api.whoami().get("name"), "| repo:", REPO, "(private)")

uploads = [
    (f"{BB}/backbone_v2_full.pt",            f"{SUB}/backbone_v2_full_original.pt"),
    (f"{BB}/backbone_v2_full_finetuned.pt",  f"{SUB}/backbone_v2_full_finetuned.pt"),
    (f"{RD}/bilinear_s0/final.pt",           f"{SUB}/readout_bilinear_s0.pt"),
    (f"{RD}/bilinear_s1/final.pt",           f"{SUB}/readout_bilinear_s1.pt"),
    (f"{RD}/bilinear_s2/final.pt",           f"{SUB}/readout_bilinear_s2.pt"),
    (f"{RD}/bilinear_s3/final.pt",           f"{SUB}/readout_bilinear_s3.pt"),
    (f"{RD}/bilinear_s4/final.pt",           f"{SUB}/readout_bilinear_s4.pt"),
    (f"{RD}/bilinear_s5/final.pt",           f"{SUB}/readout_bilinear_s5.pt"),
    (GATE,                                    f"{SUB}/serve_gate_6seed_summary.json"),
]

for local, repopath in uploads:
    if not os.path.exists(local):
        print(f"MISSING local: {local} -- skip")
        continue
    sz = os.path.getsize(local) / 1e6
    print(f"uploading {local} ({sz:.1f} MB) -> {REPO}:{repopath}", flush=True)
    upload_file(
        path_or_fileobj=local,
        path_in_repo=repopath,
        repo_id=REPO,
        repo_type="model",
        commit_message=f"Stage 3 STRM readout: {os.path.basename(repopath)}",
    )
    print(f"  ok: {repopath}", flush=True)

print("DONE. Files now in repo:")
for f in api.list_repo_files(REPO, repo_type="model"):
    if f.startswith(SUB):
        print("  ", f)