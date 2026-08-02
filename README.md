# 0pen by 0penAGI

**Experimental conversational identity — a local assistant exploring continuity, memory, and process-oriented dialogue.**

> ⚠️ Research Preview (v0.1). This is an early release for testing and discussion, not a finished product.

**Downloads & runnable model:** [huggingface.co/0penAGI/0pen](https://huggingface.co/0penAGI/0pen)

---

## What is 0pen?

0pen is not another LoRA-on-Gemma. It is an experimental **conversational identity** — a first attempt to build a local assistant that explores *continuity, memory, and process-oriented dialogue* rather than only instruction following.

The project grew from scratch in this repository:

- **Dataset** — written from real dialogues, cleaned, augmented, deduplicated, and scored (see `lora.py`).
- **Training** — LoRA fine-tuning on Gemma 4 (e4b-mlx) with an experimental **AGR** (Attractor Geometry Repeller) regularization to prevent mode collapse.
- **Export** — fused → dequantized → GGUF (Q4_K_M), ready for Ollama.

## Why this model exists

Most fine-tuned models are optimized for one thing: following instructions well. 0pen is interested in something slightly different — being a **consistent presence in a conversation**:

- a personality that persists across turns and sessions,
- honest self-description (including admitting limits),
- dialogue that feels like an exchange, not a lookup table,
- reflecting on *how* it answers, not just *what* it answers.

It is not trained to be "the best model". It is trained to be *a model worth talking to* — and worth watching evolve.

## Repository layout

```
.
├── lora.py               # Data pipeline: clean, dedup, augment, score, link, DPO + DCAT/AGR configs
├── agr.py                # Attractor Geometry Repeller (latent-space regularization)
├── agr_train.py          # AGR-integrated mlx_lm training wrapper (+ CLI)
├── Modelfile             # Ollama Modelfile (temperature, system prompt, ctx)
├── requirements.txt
├── adapters/             # Final LoRA adapter weights + AGR state
│   ├── adapter_config.json
│   ├── adapters.safetensors
│   ├── agr_state.json
│   └── agr_state.npz
├── data/                 # Training/validation data (JSONL, chat schema)
│   ├── train.jsonl
│   ├── train_old.jsonl
│   └── valid.jsonl
├── scripts/              # Gemma pipeline: fuse → prepare HF → convert GGUF
│   ├── fuse_gemma_v1.py       # Fuse LoRA adapter into Gemma base (MLX)
│   ├── prepare_gemma_hf.py    # Dequantize + strip language_model. prefix → HF dir
│   ├── convert_gemma_to_gguf.py  # convert_hf_to_gguf.py + llama-quantize Q4_K_M
│   ├── fuse_agr_v1.py         # (historic, Qwen-based iteration)
│   ├── dequantize_agr_v1.py   # (historic, Qwen-based iteration)
│   └── convert_agr_v1_to_gguf.py  # (historic, Qwen-based iteration)
```

## The full Gemma pipeline

0pen v0.1 was trained on `gemma4-e4b-mlx` (Gemma 4, 4-bit MLX, vision
architecture). The LLM submodule lives under a `language_model.` prefix, so the
LoRA adapter keys carry that prefix too. The pipeline below is exactly what
produced the released `0pen.gguf`.

### 1. Train the adapter

```bash
python agr_train.py \
  --model ./gemma4-e4b-mlx \
  --data data_zephyr_enhanced \
  --train \
  --num-layers 12 --rank 8 --scale 20 --learning-rate 1e-5 \
  --iters 4000 --max-seq-length 1792 --mask-prompt \
  --adapter-path adapters \
  --agr --agr-lambda 0.01 --agr-centers 32 --agr-ema 0.99
```

The adapter (with AGR state) is saved to `adapters/`.

### 2. Fuse the adapter into the base

```bash
python scripts/fuse_gemma_v1.py \
  --model ./gemma4-e4b-mlx \
  --adapter zephyr_lora_agr_v1 \
  --out zephyr_gemma_fused
```

### 3. Prepare an HF dir (dequantize + strip prefix)

```bash
python scripts/prepare_gemma_hf.py \
  --fused zephyr_gemma_fused \
  --base ./gemma4-e4b-mlx \
  --out zephyr_gemma_hf
```

### 4. Convert to GGUF and quantize

```bash
pip install llama-cpp-python   # provides bin/convert_hf_to_gguf.py

python scripts/convert_gemma_to_gguf.py \
  --hf zephyr_gemma_hf \
  --out 0pen.gguf
```

### 5. Create the Ollama model

```bash
ollama create 0pen -f Modelfile
```

## Quick start

### Run the model (Ollama)

```bash
ollama create 0pen -f Modelfile
ollama run 0pen
```

### Run the model (llama.cpp)

```bash
llama-cli -m /path/to/0pen.gguf -p "Привет, что ты умеешь?" -n 256
```

The GGUF is on [Hugging Face](https://huggingface.co/0penAGI/0pen).

### Train it yourself

Dataset prep, AGR training, fusing and GGUF export are documented in
[The full Gemma pipeline](#the-full-gemma-pipeline) above.

## Training summary (v0.1)

| Parameter | Value |
|---|---|
| Base model | `gemma4-e4b-mlx` (Gemma 4 e4b, 4-bit MLX, vision arch) |
| Method | LoRA (rank 8, scale 20.0, dropout 0.0) |
| Adapted layers | 12 of 34 |
| Iterations | 4000 |
| Learning rate | 1e-05 |
| Max sequence length | 1792 |
| AGR | enabled (32 centers, EMA 0.99, lambda 0.01) |
| Dataset | `data_zephyr_enhanced` (Russian + English dialogue) |
| Quantization | GGUF Q4_K_M |

## Known limitations

- Self-description often still inherits the base **Gemma** ("I am Google's model").
- The LoRA was trained on only **12 of 34 layers** — top layers are unadapted.
- `scale=20.0` is unusually high (normal is 1–4); some behavior may be skewed.
- The model can be unstable on long reasoning chains.

These are known, accepted limitations of a research preview. They are part of the experiment, not hidden bugs.

## Roadmap

| Version | Focus |
|---|---|
| **0pen v0.1** | First public release — identity, tone, basic dialogue |
| **0pen v0.2** | Improved identity and continuity, more stable training |
| **0pen v1.0** | Stable architecture, full layer coverage, documented eval |

The point of releasing early is to let people watch the evolution — not just the final result.

## License

MIT — see [LICENSE](LICENSE).

## Links

- **Model (GGUF):** [Hugging Face — 0penAGI/0pen](https://huggingface.co/0penAGI/0pen)
- **Source:** [GitHub — 0penAGI/0pen](https://github.com/0penAGI/0pen)

*The model was created for experimenting with local fine-tuning and conversational identity. Not recommended for production use without additional validation.*
