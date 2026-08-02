#!/usr/bin/env python3
"""Prepare a fused Gemma MLX model for GGUF export.

Steps:
  1. Read the fused MLX safetensors (packed uint32 int4 + bf16 scales/biases,
     with a ``language_model.`` prefix from the vision-architecture base).
  2. Dequantize every quantized layer: ``w = scales * nibble + biases`` (group 64).
  3. Strip the ``language_model.`` prefix so keys look like ``model.layers.N.*``.
  4. Write a ``Gemma3ForCausalLM``-style config (text-only) derived from the
     base model's ``text_config`` + the values baked into the GGUF.
  5. Save as a clean HF directory (bf16) for ``convert_hf_to_gguf.py``.

Usage::

    python scripts/prepare_gemma_hf.py \
        --fused zephyr_gemma_fused \
        --base gemma4-e4b-mlx \
        --out zephyr_gemma_hf
"""
import argparse
import json
import shutil
import struct
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

GROUP_SIZE = 64


def bf16_to_f32(tensor_bytes, shape):
    arr = np.frombuffer(tensor_bytes, dtype=np.uint16).reshape(shape)
    return (arr.astype(np.uint32) << 16).view(np.float32)


def dequant_mlx_int4(w_bytes, w_shape, s_bytes, s_shape, b_bytes, b_shape):
    """MLX 4-bit dequantization: result = scales * unsigned_nibble + biases."""
    packed = np.frombuffer(w_bytes, dtype=np.uint32).reshape(w_shape)
    n_groups = s_shape[1]
    in_f = w_shape[1] * 8

    unpacked = np.zeros((w_shape[0], in_f), dtype=np.float32)
    for i in range(8):
        unpacked[:, i::8] = ((packed >> (i * 4)) & 0xF).astype(np.float32)

    scales = bf16_to_f32(s_bytes, s_shape)
    biases = bf16_to_f32(b_bytes, b_shape)

    result = np.zeros((w_shape[0], in_f), dtype=np.float32)
    for g in range(n_groups):
        start = g * GROUP_SIZE
        end = min(start + GROUP_SIZE, in_f)
        result[:, start:end] = unpacked[:, start:end] * scales[:, g:g+1] + biases[:, g:g+1]
    return result


def build_text_config(base_config: dict) -> dict:
    """Build a Gemma3ForCausalLM text config from the VLM base + known values."""
    text = base_config.get("text_config", {})

    cfg = {
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3",
        "hidden_size": text.get("hidden_size", 2560),
        "intermediate_size": text.get("intermediate_size", 10240),
        "num_hidden_layers": text.get("num_hidden_layers", 34),
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_scaling": text.get("rope_scaling", {"factor": 8.0, "rope_type": "linear"}),
        "sliding_window": text.get("sliding_window", 1024),
        "vocab_size": 262208,
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "torch_dtype": "bfloat16",
    }
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused", default="zephyr_gemma_fused", help="Fused MLX model directory")
    parser.add_argument("--base", default="gemma4-e4b-mlx", help="Original base model directory (for tokenizer/config)")
    parser.add_argument("--out", default="zephyr_gemma_hf", help="Output HF directory")
    args = parser.parse_args()

    fused_dir = Path(args.fused)
    base_dir = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/5] Reading fused safetensors header...")
    with open(fused_dir / "model.safetensors", "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        metadata = json.loads(f.read(header_len).decode())
        raw = f.read()
    print(f"  {len(metadata)} entries ({time.time()-t0:.1f}s)")

    quant_bases = set()
    for key in metadata:
        if key.endswith(".scales"):
            quant_bases.add(key.rsplit(".scales", 1)[0])
    print(f"  {len(quant_bases)} quantized layers")

    print("[2/5] Dequantizing + stripping 'language_model.' prefix...")
    out = {}
    for key, info in metadata.items():
        if key == "__metadata__":
            continue
        dtype, shape, offsets = info["dtype"], info["shape"], info["data_offsets"]
        raw_bytes = raw[offsets[0]:offsets[1]]

        clean = key[len("language_model."):] if key.startswith("language_model.") else key

        if key.endswith(".weight") and key.rsplit(".weight", 1)[0] in quant_bases:
            base = key.rsplit(".weight", 1)[0]
            s_info = metadata[f"{base}.scales"]
            b_info = metadata[f"{base}.biases"]
            s_raw = raw[s_info["data_offsets"][0]:s_info["data_offsets"][1]]
            b_raw = raw[b_info["data_offsets"][0]:b_info["data_offsets"][1]]
            deq = dequant_mlx_int4(raw_bytes, tuple(shape), s_raw, tuple(s_info["shape"]),
                                   b_raw, tuple(b_info["shape"]))
            out[clean] = deq.astype(np.float16)
        elif key.endswith(".scales") or key.endswith(".biases"):
            continue
        elif dtype == "BF16":
            out[clean] = bf16_to_f32(raw_bytes, tuple(shape)).astype(np.float16)
        elif dtype == "F32":
            out[clean] = np.frombuffer(raw_bytes, dtype=np.float32).reshape(tuple(shape))
        else:
            print(f"  SKIP {key}: {dtype}")
    print(f"  {len(out)} tensors dequantized ({time.time()-t0:.1f}s)")

    print("[3/5] Writing config.json...")
    with open(base_dir / "config.json") as f:
        base_config = json.load(f)
    cfg = build_text_config(base_config)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print("[4/5] Copying tokenizer files...")
    for f in ["tokenizer.json", "tokenizer_config.json", "tokenizer.model",
              "vocab.json", "merges.txt", "added_tokens.json",
              "special_tokens_map.json", "chat_template.json", "generation_config.json"]:
        src = fused_dir / f
        if not src.exists():
            src = base_dir / f
        if src.exists():
            shutil.copy2(src, out_dir / f)
            print(f"  Copied {f}")

    print("[5/5] Saving model.safetensors...")
    save_file(out, str(out_dir / "model.safetensors"))
    sz = sum(v.nbytes for v in out.values()) / 1e9
    print(f"Done! {len(out)} tensors, {sz:.1f} GB -> {out_dir} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
