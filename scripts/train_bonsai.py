#!/usr/bin/env python
"""Stage B.2: PEFT (QLoRA) fine-tune Qwen3-8B on the contradiction pairs from
generate_contradiction_training_data.py (plan mellow-jumping-token.md).

Why QLoRA on the local 5080 (16 GB): bf16 Qwen3-8B is ~16 GB just to load the
frozen base, leaving no headroom -- so we load the base in 4-bit (bitsandbytes
nf4, ~5-6 GB) and train a LoRA adapter. Probed viable on sm_120 (Blackwell):
4-bit forward+backward succeed (probe 2026-07-16). The adapter is NEVER merged
into the ternary Bonsai base (merging into ternary rounds deltas to zero -- see
docs/Phase 3c.md Sec 7.5/ternative); it is served at runtime via
``llama-server --lora`` applied at F32. So we export the adapter to gguf and
ship base + adapter, no merge.

Training format: each record is ``{"messages": [user, assistant]}`` where the
user turn is the EXACT deploy-time prompt (BONSAI_RELATION_PROMPT /
bonsai_contradiction_decision_prompt) and the assistant turn is clean gold JSON.
Loss is masked to the assistant turn only (prompt-completion SFT) via the Qwen3
chat template + prefix masking. transformers Trainer is used directly (no
trl/datasets dependency) to minimize install surface.

    python scripts/train_bonsai.py \
        --data data/training/bonsai/contradiction_pairs.jsonl \
        --output data/training/bonsai/lora_adapter \
        --epochs 3 --lr 2e-4 --lora-r 16 --batch-size 2 --grad-accum 8
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

MODEL_DEFAULT = "Qwen/Qwen3-8B"


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_examples(records: list[dict], tokenizer, max_len: int) -> list[dict]:
    """Tokenize chat-message pairs with the USER prompt masked from loss.

    The user turn is the deploy-time prompt; the assistant turn is gold JSON we
    train on. We render the prompt (user msg + assistant header) and the full
    conversation separately, then set labels = -100 for the prompt-length prefix
    so the model is only penalized on the completion (standard prompt-completion
    SFT -- prevents overfitting the fixed prompt).
    """
    examples = []
    for r in records:
        msgs = r["messages"]
        # Prompt = user turn rendered with the assistant header appended, so the
        # model learns to CONTINUE from the assistant header (matches deploy where
        # the server sends the user msg + generates the assistant turn).
        prompt_text = tokenizer.apply_chat_template(
            msgs[:1], tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
        labels = list(full_ids)
        n_mask = min(len(prompt_ids), len(full_ids))
        for i in range(n_mask):
            labels[i] = -100
        if len(full_ids) <= n_mask:
            # degenerate (completion shorter than prompt prefix) -- skip
            continue
        examples.append({"input_ids": full_ids, "labels": labels,
                         "attention_mask": [1] * len(full_ids)})
    return examples


@dataclass
class SimpleCollator:
    """Pad input_ids + labels + attention_mask to the longest in the batch."""
    pad_token_id: int

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> int:
    p = argparse.ArgumentParser(description="PEFT QLoRA fine-tune Qwen3-8B on contradiction pairs.")
    p.add_argument("--data", default="data/training/bonsai/contradiction_pairs.jsonl")
    p.add_argument("--model", default=MODEL_DEFAULT, help="HF base model (default %(default)s)")
    p.add_argument("--output", default="data/training/bonsai/lora_adapter",
                   help="Output dir for the LoRA adapter")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", action="store_true",
                   help="Save the adapter to --output (default: train only, dry-run save)")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(Path(args.data))
    print(f"Loaded {len(records)} pairs from {args.data}")

    print(f"Loading tokenizer + 4-bit base {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    qnfr = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=qnfr, device_map="cuda",
        dtype=torch.bfloat16, attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True)
    if hasattr(model, "config"):
        model.config.use_cache = False

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=target_modules, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    examples = build_examples(records, tok, args.max_len)
    print(f"Built {len(examples)} trainable examples (prompt-masked)")
    if not examples:
        print("ERROR: no examples after tokenization"); return 2

    collator = SimpleCollator(pad_token_id=tok.pad_token_id)
    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=examples,
        data_collator=collator,
    )
    print("Training...")
    trainer.train()

    if args.save:
        model.save_pretrained(str(out))
        tok.save_pretrained(str(out))
        print(f"Saved LoRA adapter -> {out}")

    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak VRAM: {vram:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())