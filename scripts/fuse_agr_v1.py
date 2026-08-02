#!/usr/bin/env python3
"""Fuse zephyr_lora_agr_v1 LoRA adapter with mlx-community/Qwen3-4B-4bit base model."""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.utils import load_adapters
from safetensors.mx import save_file as mx_save
import shutil, json
from pathlib import Path

MODEL = "mlx-community/Qwen3-4B-4bit"
ADAPTER = Path("zephyr_lora_agr_v1")
OUT_DIR = Path("zephyr_agr_v1_fused")

print("[1/3] Loading base model...")
model, tokenizer = load(MODEL)

print("[2/3] Loading and fusing LoRA adapter...")
model = load_adapters(model, ADAPTER)

print("[3/3] Saving fused model...")
OUT_DIR.mkdir(exist_ok=True)

def flatten_params(params, prefix=""):
    flat = {}
    for k, v in params.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_params(v, full_key))
        else:
            flat[full_key] = v
    return flat

flat = flatten_params(model.parameters())
print(f"  {len(flat)} tensors")
mx_save(flat, str(OUT_DIR / "model.safetensors"))

for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "added_tokens.json", "special_tokens_map.json"]:
    src = ADAPTER / f
    if src.exists():
        shutil.copy2(src, OUT_DIR / f)

import transformers
config_src = Path(transformers.__file__).parent / "models" / "qwen3" / "config.json"
if config_src.exists():
    shutil.copy2(config_src, OUT_DIR / "config.json")

sz = (OUT_DIR / "model.safetensors").stat().st_size / 1e9
print(f"Done! Saved {sz:.1f} GB to {OUT_DIR}")
