#!/usr/bin/env python3
"""Convert zephyr_agr_v1_fused (4-bit MLX) directly to GGUF Q4_K_M — one pass, no intermediate files.
Includes llama-cli smoke test before ollama create."""
import struct, json, subprocess, sys, time, os
import numpy as np
from pathlib import Path
from gguf import GGUFWriter

FUSED_DIR = Path("zephyr_agr_v1_fused")
OUT_F16 = "zephyr_f16.gguf"
OUT_Q4 = "Zephyr_4q.gguf"
GROUP_SIZE = 64

HF_TO_GGUF = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
}
LAYER_MAP = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}

t0 = time.time()

print("[1/6] Reading raw safetensors header...")
with open(FUSED_DIR / "model.safetensors", "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    metadata = json.loads(f.read(header_len).decode())
    raw = f.read()

print(f"  {time.time()-t0:.1f}s")

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

quant_bases = set()
for key in metadata:
    if key.endswith(".scales"):
        quant_bases.add(key.rsplit(".scales", 1)[0])

print("[2/6] Loading config & tokenizer...")
with open(FUSED_DIR / "config.json") as f:
    config = json.load(f)
with open(FUSED_DIR / "tokenizer.json") as f:
    tokenizer = json.load(f)
with open(FUSED_DIR / "vocab.json") as f:
    vocab = json.load(f)
with open(FUSED_DIR / "merges.txt") as f:
    merges = [line.strip() for line in f if line.strip()]

print("[3/6] Writing F16 GGUF (dequantizing on the fly)...")
writer = GGUFWriter(str(OUT_F16), arch="qwen3")

writer.add_name("Zephyr")
writer.add_context_length(config.get("max_position_embeddings", 40960))
writer.add_embedding_length(config.get("hidden_size", 2560))
writer.add_block_count(config.get("num_hidden_layers", 36))
writer.add_feed_forward_length(config.get("intermediate_size", 9728))
writer.add_head_count(config.get("num_attention_heads", 32))
writer.add_head_count_kv(config.get("num_key_value_heads", 8))
writer.add_rope_freq_base(config.get("rope_theta", 1000000))
writer.add_layer_norm_rms_eps(config.get("rms_norm_eps", 1e-6))
writer.add_key_length(config.get("head_dim", 128))
writer.add_value_length(config.get("head_dim", 128))
writer.add_file_type(1)

writer.add_tokenizer_model("gpt2")
writer.add_tokenizer_pre("qwen2")

added = tokenizer.get("added_tokens", [])
added_sorted = sorted(added, key=lambda x: x.get("id", 0))
max_id = max(t.get("id", 0) for t in added_sorted) if added_sorted else 0

tokens = [""] * (max_id + 1)
scores = [0.0] * (max_id + 1)
token_types = [1] * (max_id + 1)

for token, score in vocab.items():
    if score < len(tokens):
        tokens[score] = token
        token_types[score] = 1

for at in added_sorted:
    tid = at.get("id", 0)
    tokens[tid] = at["content"]
    scores[tid] = -1000.0
    token_types[tid] = 3 if at.get("special", False) else 1

actual_vocab = metadata["model.embed_tokens.weight"]["shape"][0]
while len(tokens) < actual_vocab:
    tokens.append(f"<|pad_{len(tokens)}|>")
    scores.append(-1000.0)
    token_types.append(3)
tokens = tokens[:actual_vocab]
scores = scores[:actual_vocab]
token_types = token_types[:actual_vocab]

writer.add_token_list(tokens)
writer.add_token_scores(scores)
writer.add_token_types(token_types)
writer.add_token_merges(merges)

bos = next((t["id"] for t in added_sorted if t.get("content") == "<|begin_of_text|>"), None)
eos = next((t["id"] for t in added_sorted if t.get("content") == "<|end_of_text|>"), None)
if bos is not None: writer.add_bos_token_id(bos)
if eos is not None: writer.add_eos_token_id(eos)

print(f"  Tokenizer: {len(tokens)} tokens, {len(merges)} merges")

mapped = 0
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
        deq = dequant_mlx_int4(raw_bytes, tuple(shape), s_raw, tuple(s_info["shape"]), b_raw, tuple(b_info["shape"]))
        arr = deq.astype(np.float16)
        print(f"  {base}: dequant done ({time.time()-t0:.1f}s)")
    elif key.endswith(".scales") or key.endswith(".biases"):
        continue
    elif dtype == "BF16":
        arr = bf16_to_f32(raw_bytes, tuple(shape)).astype(np.float16)
    else:
        continue

    if key in HF_TO_GGUF:
        gguf_name = HF_TO_GGUF[key]
    elif key.startswith("model.layers."):
        rest = key[len("model.layers."):]
        dot = rest.index(".")
        layer_num = int(rest[:dot])
        suffix = rest[dot+1:]
        gguf_name = f"blk.{layer_num}.{LAYER_MAP.get(suffix, suffix)}"
    else:
        gguf_name = key

    writer.add_tensor(gguf_name, arr)
    mapped += 1

if config.get("tie_word_embeddings", False):
    embed_info = metadata["model.embed_tokens.weight"]
    embed_raw = raw[embed_info["data_offsets"][0]:embed_info["data_offsets"][1]]
    embed_s = metadata["model.embed_tokens.scales"]
    embed_b = metadata["model.embed_tokens.biases"]
    embed_s_raw = raw[embed_s["data_offsets"][0]:embed_s["data_offsets"][1]]
    embed_b_raw = raw[embed_b["data_offsets"][0]:embed_b["data_offsets"][1]]
    deq = dequant_mlx_int4(embed_raw, tuple(embed_info["shape"]),
                            embed_s_raw, tuple(embed_s["shape"]),
                            embed_b_raw, tuple(embed_b["shape"]))
    writer.add_tensor("output.weight", deq.astype(np.float16))
    mapped += 1
    print("  Added output.weight (tied)")

print(f"  {mapped} tensors written")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

f16_size = Path(OUT_F16).stat().st_size / 1e9
print(f"  F16 GGUF: {OUT_F16} ({f16_size:.1f} GB) [{time.time()-t0:.0f}s]")

print("[4/6] Quantizing to Q4_K_M...")
t1 = time.time()
result = subprocess.run(["llama-quantize", OUT_F16, "Q4_K_M"], capture_output=True, text=True)
if result.returncode != 0:
    print(f"  ERROR: {result.stderr}")
    sys.exit(1)
auto_out = Path("ggml-model-Q4_K_M.gguf")
if auto_out.exists():
    auto_out.rename(OUT_Q4)
q4_size = Path(OUT_Q4).stat().st_size / 1e9
print(f"  Q4_K_M GGUF: {OUT_Q4} ({q4_size:.1f} GB) [{time.time()-t1:.0f}s]")

print("[5/6] Smoke testing with llama-cli...")
t2 = time.time()
result = subprocess.run(
    ["llama-cli", "-m", OUT_Q4, "-p", "Hello", "-n", "20", "--no-display-prompt", "-t", "4"],
    capture_output=True, text=True, timeout=120
)
if result.returncode != 0:
    print(f"  SMOKE TEST FAILED:")
    print(f"  stdout: {result.stdout[-500:]}")
    print(f"  stderr: {result.stderr[-500:]}")
    sys.exit(1)
output = result.stdout.strip()
print(f"  Smoke test passed ({time.time()-t2:.0f}s): {output[:100]}")

print("[6/6] Creating Ollama model...")
result = subprocess.run(["ollama", "create", "Zephyr", "-f", "Modelfile_Zephyr"], capture_output=True, text=True)
if result.returncode != 0:
    print(f"  ERROR: {result.stderr}")
    sys.exit(1)
print(f"Done! [{time.time()-t0:.0f}s total]")
