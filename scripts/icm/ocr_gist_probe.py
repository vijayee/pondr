"""OCR gist probe (steal R10) -- DeepSeek-OCR as the lossy re-encoder.

Question: on OUR text (agent work-log style), what does the model reconstruct
at each of its native resolution modes, and specifically what happens to
NUMBERS (the gist-bottleneck lesson says arbitrary values die in a compressed
bottleneck; the DeepSeek-OCR paper says ~97% precision <10x)? The gist seam
(icm-forget/1 blur stage) wants the answer before the swap.

Modes (vision tokens): Tiny 512/512=64, Small 640/640=100, Base 1024/640=256,
Large 1280/640=400.

Run inside .venv-icm-ocr:  .venv-icm-ocr/Scripts/python scripts/icm/ocr_gist_probe.py
Downloads deepseek-ai/DeepSeek-OCR on first use (~7 GB into the HF cache).
"""

from __future__ import annotations

import difflib
import io
import json
import os
import re
import sys
import time

from PIL import Image, ImageDraw, ImageFont

SAMPLE_TEXTS = [
    chr(10).join([
        "WORK LOG - 22 June, on-call shift notes (long form).",
        "Checkpoint [deploy]: staging restart finished, probe returned 42ms, retry path green. "
        "Run sbx-003 ended clean; canary budget 0.3 percent; rollback rehearsal ran in 4 minutes.",
        "Checkpoint [db]: schema migration step 12 applied; lock timeout 300 seconds; index rebuild took 11 minutes; "
        "the table locked twice during the rebuild; vacuum reclaimed 18 GB.",
        "Checkpoint [ci]: CI runner pruned 212 stale images, freed 31 GB; builder cache cleared; "
        "first cold build took 7 minutes; second warm build took 46 seconds; queue depth peaked at 9.",
        "Checkpoint [backup]: nightly backup completed at 03:40, verify checksum matched; disk at 94 percent; "
        "rotation window 24 hours; API credentials rotated with a 300-second overlap window.",
        "Checkpoint [search]: benchmark P95 at 88ms, regression against last week's 71ms; ingest throughput 4.1k rows/s; "
        "cache hit rate 61 percent; the recall probe flagged 3 low-confidence entities.",
        "Notes: on-call handoff at 14:00 to the storage team; page rate 2 pages/hour; "
        "the weekly report went out with 3 action items assigned to infra and 2 to search.",
        "Incident detail: the flaky dependency probe failed twice at 09:12 and 09:18, retry succeeded on the third try; "
        "the manifest verify found a checksum mismatch on part-07; the artifact cache was rebuilt; "
        "the whole incident cost 47 minutes of downtime.",
    ]),
    chr(10).join([
        "WORK LOG - 23 June, morning shift (long form).",
        "Checkpoint [deploy]: production deploy at 02:10; probe latency 57ms; health checks 12/12 green; "
        "error budget burn 0.9 percent; soak ran for 40 minutes.",
        "Checkpoint [db]: replica lag peaked at 1400 seconds; catch-up took 22 minutes; "
        "the lock timeout was raised to 480 seconds; 2 slow queries flagged over 1200ms.",
        "Checkpoint [ci]: 128 stale images pruned (18 GB); warm build 41 seconds; cold build 9 minutes; "
        "the runner queue peaked at 14 jobs.",
        "Checkpoint [search]: benchmark P95 92ms; ingest 3.8k rows/s; cache hit 58 percent; "
        "the drift telemetry fired once on entity turnover.",
        "Notes: the weekly review at 10:00; 4 action items for infra; disk trend +3 GB/day; "
        "backup retention moved to 35 days; credentials rotation scheduled for 26 June 06:00.",
        "Follow-ups: re-run the checksum probe on parts 3-9; verify the replica's WAL segments; "
        "schedule the P95 regression analysis for Friday.",
        "Incident detail: backup was skipped at 03:40 because disk hit 97 percent; cleanup script ran manually; "
        "the verify pass found no mismatch; overall delay 2 hours 20 minutes.",
    ]),
]

FONT_CANDIDATES = [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\arial.ttf"]


def render_text_image(text: str, size: int, out_path: str) -> str:
    """White page, wrapped black mono text, margin; deterministic."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for cand in FONT_CANDIDATES:
        if os.path.exists(cand):
            font = ImageFont.truetype(cand, max(14, size // 48))
            break
    if font is None:
        font = ImageFont.load_default()
    # wrap by measured width
    max_w = size - 2 * (size // 24)
    margin = size // 24
    y = margin
    max_y = size - margin
    for para in text.split(" "):
        paragraph, line = para, ""
        words = para.split()
        pass
    for word_para in text.split("\n"):
        line = ""
        for word in word_para.split():
            trial = (line + " " + word).strip()
            if draw.textlength(trial, font=font) <= max_w:
                line = trial
            else:
                draw.text((margin, y), line, fill="black", font=font)
                y += size // 20
                if y > max_y:
                    break
                line = word
        if y > max_y:
            break
        if line:
            draw.text((margin, y), line, fill="black", font=font)
        y += size // 20
    img.save(out_path)
    return out_path


def norm_words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9.:/%-]+", s.lower())


def word_precision(source: str, decoded: str) -> dict:
    src = norm_words(source)
    dec = norm_words(decoded)
    keep2 = {w for w in dec}
    hits = sum(1 for w in src if w in keep2)
    precision = hits / max(1, len(src))  # of source words, how many appear in decode
    numbers_src = [w for w in src if re.search(r"\d", w)]
    numbers_kept = sum(1 for w in numbers_src if w in keep2)
    return {"word_recall": round(precision, 4),
            "n_source_words": len(src),
            "number_tokens_source": len(numbers_src),
            "number_tokens_kept": numbers_kept,
            "number_recall": round(numbers_kept / max(1, len(numbers_src)), 4)}


MODES = {"tiny": (512, 512), "small": (640, 640), "base": (1024, 640), "large": (1280, 640)}


def main() -> int:
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ.setdefault("HF_HOME", os.path.join("data", "_scratch", "icm", "hf_cache"))
    model_name = "deepseek-ai/DeepSeek-OCR"
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                      use_safetensors=True,
                                      _attn_implementation="eager")
    model = model.eval().cuda().to(torch.bfloat16)
    print(f"[load] model ready in {time.time()-t0:.0f}s", flush=True)

    out = {"model": model_name, "attn": "eager", "modes": {}}
    for mode, (base_size, image_size) in MODES.items():
        rows = []
        for si, text in enumerate(SAMPLE_TEXTS):
            img_path = os.path.join("data", "_scratch", "icm", f"probe_page_{si}.png")
            render_text_image(text, image_size, img_path)
            t1 = time.time()
            out_dir = os.path.join("data", "_scratch", "icm", "ocr_out")
            # infer returns the decoded text ONLY in eval_mode (save_results
            # prints + writes elsewhere); eval_mode is the clean capture
            res = model.infer(tokenizer,
                              prompt="<image>\nFree OCR.",
                              image_file=img_path,
                              output_path=out_dir,
                              base_size=base_size, image_size=image_size,
                              crop_mode=False, save_results=False,
                              eval_mode=True, test_compress=True)
            decoded = res if isinstance(res, str) else str(res or "")
            m = word_precision(text, decoded)
            m["latency_s"] = round(time.time() - t1, 1)
            m["decoded"] = decoded[:400]
            rows.append(m)
            print(f"[{mode}/{si}] word_recall={m['word_recall']} "
                  f"number_recall={m['number_recall']} ({m['latency_s']}s)", flush=True)
        out["modes"][mode] = {
            "tokens": {"tiny": 64, "small": 100, "base": 256, "large": 400}[mode],
            "avg_word_recall": round(sum(r["word_recall"] for r in rows) / len(rows), 4),
            "avg_number_recall": round(sum(r["number_recall"] for r in rows) / len(rows), 4),
            "rows": rows,
        }
    path = os.path.join("data", "_scratch", "icm", "ocr_gist_probe.json")
    json.dump(out, io.open(path, "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: {a: b for a, b in v.items() if a != "rows"}
                      for k, v in out["modes"].items()}, indent=2))
    print(f"[done] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())