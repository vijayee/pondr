"""OCR re-encode daemon -- runs in .venv-icm-ocr ONLY.

Loads DeepSeek-OCR once, then serves line-delimited JSON requests on stdin:

    {"text": "...", "mode": "small", "id": "0"}
    -> {"id": "0", "gist": "[ocr-gist/small] ..."} | {"id": "0", "error": "..."}

The main environment never imports torch/transformers; this process never
imports wavedb. One line of stderr startup chatter is expected (model load
~100s warm cache). EOF on stdin = clean exit.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

PROBE_MODES = {"tiny": (512, 512), "small": (640, 640), "base": (1024, 640), "large": (1280, 640)}


def render_page(text: str, size: int, out_path: str) -> str:
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for cand in (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if os.path.exists(cand):
            font = ImageFont.truetype(cand, max(14, size // 48))
            break
    if font is None:
        font = ImageFont.load_default()
    margin = size // 24
    max_w = size - 2 * margin
    y = margin
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = (line + " " + word).strip()
            if draw.textlength(trial, font=font) <= max_w:
                line = trial
            else:
                draw.text((margin, y), line, "black", font=font)
                y += size // 20
                line = word
        if y > size - margin:
            break
        if line:
            draw.text((margin, y), line, "black", font=font)
        y += size // 20
    img.save(out_path)
    return out_path


def main() -> int:
    os.environ.setdefault("HF_HOME", os.path.join("data", "_scratch", "icm", "hf_cache"))
    model_name = "deepseek-ai/DeepSeek-OCR"
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                      use_safetensors=True,
                                      _attn_implementation="eager")
    model = model.eval().cuda().to(torch.bfloat16)
    print("WAVEDB_OCR_LOAD:" + json.dumps({"loaded": round(time.time() - t0, 1)}), flush=True)
    page_dir = os.path.join("data", "_scratch", "icm", "ocr_pages")
    os.makedirs(page_dir, exist_ok=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            text = req["text"]
            mode = req.get("mode", "small")
            if mode not in PROBE_MODES:
                raise ValueError(f"mode {mode!r} unknown")
            image_size, base_size = PROBE_MODES[mode][1], PROBE_MODES[mode][0]
            img = os.path.join(page_dir, f"daemon_{int(time.time() * 1000)}.png")
            render_page(text, image_size, img)
            res = model.infer(tokenizer, prompt="<image>\nFree OCR.",
                              image_file=img, output_path=page_dir,
                              base_size=base_size, image_size=image_size,
                              crop_mode=False, save_results=False,
                              eval_mode=True, test_compress=False)
            gist = res.strip() if isinstance(res, str) and res.strip() else None
            if gist is None:
                print("WAVEDB_OCR_RESULT:" + json.dumps({"id": req.get("id"), "error": "empty decode"}),
                      flush=True)
            else:
                print("WAVEDB_OCR_RESULT:" + json.dumps(
                    {"id": req.get("id"), "gist": "[ocr-gist/" + mode + "] " + gist}),
                    flush=True)
        except Exception as e:  # noqa: BLE001 -- daemon keeps serving
            print("WAVEDB_OCR_RESULT:" + json.dumps(
                {"id": (locals().get("req", {}) or {}).get("id"),
                 "error": f"{type(e).__name__}: {e}"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())