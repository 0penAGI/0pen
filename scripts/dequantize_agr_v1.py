#!/usr/bin/env python3
"""Dequantize zephyr_agr_v1_fused (uint32 packed int4 + bf16 scales/biases) -> float16 safetensors"""
import struct, json, time, os
import numpy as np
from safetensors.numpy import save_file

SRC = "zephyr_agr_v1_fused/model.safetensors"
DST_DIR = "zephyr_agr_v1_f16"
DST = os.path.join(DST_DIR, "model.safetensors")
GROUP_SIZE = 64
os.makedirs(DST_DIR, exist_ok=True)

t0 = time.time()

with open(SRC, "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    metadata = json.loads(f.read(header_len).decode())
    raw = f.read()

print(f"Read header in {time.time()-t0:.1f}s")

def bf16_to_f32(tensor_bytes, shape):
    arr = np.frombuffer(tensor_bytes, dtype=np.uint16).reshape(shape)
    return (arr.astype(np.uint32) << 16).view(np.float32)

def dequant_mlx_int4(w_bytes, w_shape, s_bytes, s_shape, b_bytes, b_shape):
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

# Find quantized layers
quant_bases = set()
for key in metadata:
    if key.endswith(".scales"):
        quant_bases.add(key.rsplit(".scales", 1)[0])

out = {}
for key, info in metadata.items():
    if key == "__metadata__":
        continue
    dtype, shape, offsets = info["dtype"], info["shape"], info["data_offsets"]
    raw_bytes = raw[offsets[0]:offsets[1]]

    if key.endswith(".weight") and key.rsplit(".weight", 1)[0] in quant_bases:
        base = key.rsplit(".weight", 1)[0]
        s_info = metadata[f"{base}.scales"]
        b_info = metadata[f"{base}.biases"]
        s_raw = raw[s_info["data_offsets"][0]:s_info["data_offsets"][1]]
        b_raw = raw[b_info["data_offsets"][0]:b_info["data_offsets"][1]]

        deq = dequant_mlx_int4(raw_bytes, tuple(shape),
                                s_raw, tuple(s_info["shape"]),
                                b_raw, tuple(b_info["shape"]))
        out[key] = deq.astype(np.float16)
        print(f"  {key}: {shape} -> [{shape[0]}, {shape[1]*8}] fp16 ({time.time()-t0:.1f}s)")
    elif key.endswith(".scales") or key.endswith(".biases"):
        continue  # skip, handled above
    elif dtype == "BF16":
        out[key] = bf16_to_f32(raw_bytes, tuple(shape)).astype(np.float16)
        print(f"  {key}: bf16->fp16 {shape}")
    else:
        print(f"  SKIP {key}: {dtype}")

print(f"Writing {len(out)} tensors...")
save_file(out, DST)
sz = os.path.getsize(DST) / 1e9
print(f"Done: {sz:.1f} GB in {time.time()-t0:.1f}s")
