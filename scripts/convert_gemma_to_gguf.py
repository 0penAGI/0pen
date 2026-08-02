#!/usr/bin/env python3
"""Convert a prepared HF Gemma model to GGUF (F16) and quantize to Q4_K_M.

Wraps the llama.cpp scripts ``convert_hf_to_gguf.py`` (which ships with
``llama-cpp-python`` as ``bin/convert_hf_to_gguf.py``) and ``llama-quantize``.

Usage::

    python scripts/convert_gemma_to_gguf.py \
        --hf zephyr_gemma_hf \
        --out 0pen.gguf \
        --convert-hf-to-gguf /path/to/convert_hf_to_gguf.py \
        --llama-quantize llama-quantize
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_convert_script():
    """Locate llama.cpp convert_hf_to_gguf.py (ships inside llama-cpp-python)."""
    import llama_cpp
    site = Path(llama_cpp.__file__).parent.parent
    cand = site / "bin" / "convert_hf_to_gguf.py"
    if cand.exists():
        return str(cand)
    which = shutil.which("convert_hf_to_gguf.py")
    if which:
        return which
    raise FileNotFoundError(
        "convert_hf_to_gguf.py not found. Pass --convert-hf-to-gguf explicitly, "
        "or pip install llama-cpp-python."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", default="zephyr_gemma_hf", help="Prepared HF model directory")
    parser.add_argument("--out", default="0pen.gguf", help="Output GGUF path")
    parser.add_argument("--outtype", default="f16", choices=["f16", "bf16", "q8_0", "auto"])
    parser.add_argument("--convert-hf-to-gguf", default=None,
                        help="Path to llama.cpp convert_hf_to_gguf.py")
    parser.add_argument("--llama-quantize", default="llama-quantize",
                        help="Path to llama-quantize binary")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Only produce the F16 GGUF, skip Q4_K_M quantization")
    args = parser.parse_args()

    convert = args.convert_hf_to_gguf or find_convert_script()
    hf_dir = Path(args.hf)
    out_path = Path(args.out)
    f16_path = out_path.with_name(out_path.stem + "_f16.gguf")

    print(f"[1/2] Converting {hf_dir} -> {f16_path}")
    r = subprocess.run(
        [sys.executable, convert, str(hf_dir), "--outfile", str(f16_path),
         "--outtype", args.outtype],
        capture_output=True, text=True,
    )
    print(r.stdout[-2000:] if r.stdout else "")
    if r.returncode != 0:
        print(r.stderr[-2000:] if r.stderr else "")
        sys.exit(f"convert_hf_to_gguf failed with code {r.returncode}")

    if args.skip_quantize:
        print(f"Done! F16 GGUF: {f16_path}")
        return

    print(f"[2/2] Quantizing {f16_path} -> Q4_K_M ({out_path})")
    r = subprocess.run(
        [args.llama_quantize, str(f16_path), "Q4_K_M", str(out_path)],
        capture_output=True, text=True,
    )
    print(r.stdout[-2000:] if r.stdout else "")
    if r.returncode != 0:
        print(r.stderr[-2000:] if r.stderr else "")
        sys.exit(f"llama-quantize failed with code {r.returncode}")

    sz = out_path.stat().st_size / 1e9
    print(f"Done! {out_path} ({sz:.2f} GB)")


if __name__ == "__main__":
    main()
