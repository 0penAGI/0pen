#!/usr/bin/env python3
"""Fuse a LoRA adapter into the Gemma base model (MLX) and save the fused model.

Works with vision-architecture Gemma bases (``Gemma3ForConditionalGeneration`` /
``Gemma4ForConditionalGeneration``) whose text submodule lives under a
``language_model.`` prefix — this is what 0pen v0.1 was trained on
(``gemma4-e4b-mlx`` + ``zephyr_lora_agr_v1``).

Usage::

    python scripts/fuse_gemma_v1.py \
        --model gemma4-e4b-mlx \
        --adapter zephyr_lora_agr_v1 \
        --out zephyr_gemma_fused
"""
import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.utils import load_adapters

try:
    from mlx_vlm import load as mlx_vlm_load
    HAS_VLM = True
except ImportError:
    mlx_vlm_load = None
    HAS_VLM = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4-e4b-mlx", help="Path or HF id of the MLX base model")
    parser.add_argument("--adapter", default="zephyr_lora_agr_v1", help="Path to the LoRA adapter directory")
    parser.add_argument("--out", default="zephyr_gemma_fused", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    adapter_dir = Path(args.adapter)

    print(f"[1/4] Loading base model: {args.model}")
    config_path = Path(args.model) / "config.json"
    is_vlm = False
    if config_path.exists():
        import json
        with open(config_path) as f:
            arch = json.load(f).get("architectures", [])
        is_vlm = any("ConditionalGeneration" in a for a in arch)

    if is_vlm and HAS_VLM:
        print("  Detected VLM architecture, loading with mlx-vlm")
        model, tokenizer = mlx_vlm_load(args.model)
    else:
        model, tokenizer = mlx_load(args.model, tokenizer_config={"trust_remote_code": True})

    print(f"[2/4] Loading LoRA adapter: {args.adapter}")
    model = load_adapters(model, str(adapter_dir))

    print("[3/4] Fusing adapter into base weights")
    model.fuse()

    print(f"[4/4] Saving fused model to {out_dir}")
    out_dir.mkdir(exist_ok=True)

    from safetensors.mx import save_file as mx_save

    def flatten_params(params, prefix=""):
        flat = {}
        for k, v in params.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_params(v, full))
            else:
                flat[full] = v
        return flat

    flat = flatten_params(model.parameters())
    print(f"  {len(flat)} tensors")
    mx_save(flat, str(out_dir / "model.safetensors"))

    import shutil
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "added_tokens.json",
              "special_tokens_map.json", "chat_template.json"]:
        src = adapter_dir / f
        if not src.exists():
            src = Path(args.model) / f
        if src.exists():
            shutil.copy2(src, out_dir / f)

    sz = (out_dir / "model.safetensors").stat().st_size / 1e9
    print(f"Done! {sz:.1f} GB saved to {out_dir}")


if __name__ == "__main__":
    main()
