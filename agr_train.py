"""AGR-integrated training wrapper for mlx_lm.

Patches the MLX training loop to capture hidden states and inject the
Attractor Geometry Repeller loss into the gradient path.

Usage::

    from agr_train import agr_train
    from agr import AttractorRepeller

    agr = AttractorRepeller(dim=2560, num_centers=32)

    agr_train(
        model=model,
        optimizer=optimizer,
        train_dataset=dataset,
        val_dataset=val_dataset,
        agr=agr,
        agr_lambda=0.01,
        args=training_args,
    )

Or as a drop-in replacement for ``mlx_lm.tuner.trainer.train``::

    from agr_train import install_hidden_capture, make_agr_loss

    install_hidden_capture(model)
    custom_loss = make_agr_loss(agr, agr_lambda=0.01)
    train(model, optimizer, train_dataset, loss=custom_loss, ...)
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from agr import AttractorRepeller


# ---------------------------------------------------------------------------
# Hidden-state capture via monkey-patch
# ---------------------------------------------------------------------------

def install_hidden_capture(model) -> None:
    """Patch LlamaModel.__call__ to stash the last hidden state.

    After calling ``model(inputs)``, the attribute
    ``model._last_hidden_state`` holds the pre-norm hidden activations
    from the final transformer layer, shape ``(batch, seq_len, dim)``.

    This works with ``mx.compile`` because the attribute assignment is a
    Python side-effect on the model object, not inside the MX graph.

    Supports LlamaModel, Qwen3Model, and any architecture that follows
    the ``Model -> self.model (XModel) -> layers + norm`` pattern.
    """
    # Navigate: Model -> self.language_model (Gemma4) or self.model (XModel)
    inner = getattr(model, "language_model", None) or getattr(model, "model", model)

    # Validate: must have .layers (Llama/Qwen/Gemma4 pattern)
    if not hasattr(inner, "layers"):
        raise TypeError(
            f"Cannot patch {type(inner).__name__}: missing .layers. "
            "AGR hidden capture requires a transformer architecture with accessible layers."
        )

    inner_cls = type(inner)
    original_call = inner_cls.__call__

    def _patched_call(self, inputs, cache=None, input_embeddings=None):
        # Run the original forward pass
        out = original_call(self, inputs, cache, input_embeddings)

        # Store detached hidden state only for AGR sampling.
        # Do not keep the training graph alive through the model object.
        self._last_hidden_state = out
        return out

    inner_cls.__call__ = _patched_call

    # Store reference for cleanup
    model._agr_original_call = original_call
    model._agr_inner = inner
    model._agr_inner_cls = inner_cls


def uninstall_hidden_capture(model) -> None:
    """Restore the original inner model __call__."""
    if hasattr(model, "_agr_original_call") and hasattr(model, "_agr_inner_cls"):
        model._agr_inner_cls.__call__ = model._agr_original_call
        del model._agr_original_call
        del model._agr_inner
        del model._agr_inner_cls


# ---------------------------------------------------------------------------
# Custom loss with AGR repeller
# ---------------------------------------------------------------------------

def make_agr_loss(
    agr: AttractorRepeller,
    agr_lambda: float = 0.01,
    base_loss_fn: Callable | None = None,
) -> Callable:
    """Return a loss function that adds AGR repeller loss to standard CE.

    Parameters
    ----------
    agr : AttractorRepeller
        The AGR module instance.
    agr_lambda : float
        Weight for the repeller loss component.
    base_loss_fn : callable, optional
        Base loss function (defaults to ``default_loss``).

    Returns
    -------
    callable
        ``loss(model, batch, lengths) -> (loss_value, ntokens)``
    """
    import mlx.nn as nn

    if base_loss_fn is None:
        from mlx_lm.tuner.trainer import default_loss
        base_loss_fn = default_loss

    def agr_loss(model, batch, lengths):
        # Standard CE loss
        ce, ntoks = base_loss_fn(model, batch, lengths)

        inner = getattr(model, "_agr_inner", None)
        if inner is None:
            inner = getattr(model, "model", model)

        hidden = getattr(inner, "_last_hidden_state", None)
        if hidden is None:
            return ce, ntoks

        hidden_pooled = hidden.mean(axis=1)

        # Gemma4 can return multimodal flattened hidden states during eval.
        # Keep only the language hidden dimension expected by AGR.
        if hidden_pooled.shape[-1] != agr.dim:
            hidden_pooled = hidden_pooled.reshape(-1, hidden_pooled.shape[-1])[:, :agr.dim]

        repeller = agr.mlx_repeller_loss(hidden_pooled)

        # Do not update AGR centers inside the loss function.
        # MLX traces loss functions with transformations and Python .item()
        # calls inside update() are not allowed during grad/vmap/compile.

        del hidden
        del hidden_pooled
        inner._last_hidden_state = None

        return ce + agr_lambda * repeller, ntoks

    return agr_loss


# ---------------------------------------------------------------------------
# Drop-in train wrapper
# ---------------------------------------------------------------------------


def agr_train(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args=None,
    agr: AttractorRepeller | None = None,
    agr_lambda: float = 0.01,
    iterate_batches=None,
    training_callback=None,
    agr_save_path: str | None = None,
):
    """Drop-in replacement for ``mlx_lm.tuner.trainer.train`` with AGR.

    All standard ``train()`` parameters are forwarded. Additional args:

    agr : AttractorRepeller
        AGR module instance.
    agr_lambda : float
        Weight for repeller loss.
    agr_save_path : str, optional
        Path to save AGR checkpoint (saved alongside adapter weights).
    """
    from mlx_lm.tuner.trainer import train as mlx_train

    if agr is None:
        # No AGR — fall through to standard training
        return mlx_train(
            model, optimizer, train_dataset, val_dataset,
            args=args, iterate_batches=iterate_batches,
            training_callback=training_callback,
        )

    # Install hidden-state capture
    install_hidden_capture(model)

    # Build custom loss
    custom_loss = make_agr_loss(agr, agr_lambda=agr_lambda)
    mx.clear_cache()

    # Print AGR info
    print(f"\n[AGR] Attractor Geometry Repeller active")
    print(f"  Centers: {agr.num_centers}, dim: {agr.dim}")
    print(f"  Lambda: {agr_lambda}, EMA: {agr.ema}")
    print(f"  Repeller weight: {agr.repeller_weight}")
    print(f"  Min distance: {agr.min_distance}\n")

    # Run standard training with custom loss
    train_kwargs = {
        "args": args,
        "loss": custom_loss,
    }
    if iterate_batches is not None:
        train_kwargs["iterate_batches"] = iterate_batches
    if training_callback is not None:
        train_kwargs["training_callback"] = training_callback

    mlx_train(
        model, optimizer, train_dataset, val_dataset,
        **train_kwargs,
    )
    mx.clear_cache()

    # Save AGR state after training
    if agr_save_path:
        agr.save_checkpoint(agr_save_path)
        print(f"[AGR] Saved attractor state to {agr_save_path}")

    # Cleanup
    uninstall_hidden_capture(model)


# ---------------------------------------------------------------------------
# CLI — drop-in replacement for mlx_lm.lora with AGR
# ---------------------------------------------------------------------------

def build_parser():
    """Mirror ``mlx_lm.lora.build_parser`` and add AGR-specific flags."""
    from mlx_lm.lora import build_parser as lora_parser

    parser = lora_parser()
    agr_group = parser.add_argument_group("AGR — Attractor Geometry Repeller")
    agr_group.add_argument(
        "--agr", action="store_true", default=False,
        help="Enable AGR repeller loss during training",
    )
    agr_group.add_argument(
        "--agr-lambda", type=float, default=0.01,
        help="Weight for AGR repeller loss (default: 0.01)",
    )
    agr_group.add_argument(
        "--agr-centers", type=int, default=32,
        help="Number of attractor centers (default: 32)",
    )
    agr_group.add_argument(
        "--agr-ema", type=float, default=0.99,
        help="EMA decay for center updates (default: 0.99)",
    )
    agr_group.add_argument(
        "--agr-weight", type=float, default=0.01,
        help="Base repeller weight (default: 0.01)",
    )
    agr_group.add_argument(
        "--agr-min-dist", type=float, default=1.0,
        help="Minimum distance penalty threshold (default: 1.0)",
    )
    agr_group.add_argument(
        "--agr-checkpoint", type=str, default=None,
        help="Path to save/load AGR attractor state",
    )
    return parser


def _resolve_hidden_size(model) -> int:
    """Extract the hidden dimension from the model architecture."""
    inner = getattr(model, "language_model", None) or getattr(model, "model", model)

    if hasattr(inner, "args") and hasattr(inner.args, "hidden_size"):
        return inner.args.hidden_size

    if hasattr(inner, "config"):
        config = inner.config
        for key in ("hidden_size", "text_config"):
            value = getattr(config, key, None) if not isinstance(config, dict) else config.get(key)
            if key == "text_config" and value is not None:
                if hasattr(value, "hidden_size"):
                    return value.hidden_size
                if isinstance(value, dict) and "hidden_size" in value:
                    return value["hidden_size"]
            elif value is not None:
                return value

    embed = getattr(inner, "embed_tokens", None)
    if embed is not None:
        return embed.weight.shape[1]

    raise ValueError(f"Cannot determine hidden_size from model: {type(inner).__name__}")


def agr_main():
    """Entry point: ``python agr_train.py --model ... --data ... --train --agr``."""
    import os
    import types as _types

    try:
        from mlx_vlm import load as mlx_vlm_load
    except ImportError:
        mlx_vlm_load = None
    from mlx_lm.utils import load as mlx_load
    from mlx_lm.lora import CONFIG_DEFAULTS
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters, print_trainable_parameters
    import mlx.optimizers as optim
    from mlx_lm.tuner.utils import build_schedule

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    parser = build_parser()
    args = parser.parse_args()
    config = args.config
    args = vars(args)
    if config:
        import yaml
        print("Loading configuration file", config)
        with open(config, "r") as file:
            config = yaml.load(file, yaml.SafeLoader)
        for k, v in config.items():
            if args.get(k, None) is None:
                args[k] = v

    for k, v in CONFIG_DEFAULTS.items():
        if args.get(k, None) is None:
            args[k] = v

    args = _types.SimpleNamespace(**args)
    # Enable AGR automatically when a non-zero AGR lambda is provided.
    # This keeps `--agr-lambda 0.01` from silently running standard training.
    if args.agr_lambda > 0:
        args.agr = True

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # ---- Load model ----
    print("Loading pretrained model")
    import json
    from pathlib import Path

    model_config_path = Path(args.model) / "config.json"
    is_gemma4_vlm = False
    if model_config_path.exists():
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
        is_gemma4_vlm = "Gemma4ForConditionalGeneration" in model_config.get("architectures", [])

    if is_gemma4_vlm:
        if mlx_vlm_load is None:
            raise ImportError("Gemma4 requires mlx-vlm. Install with: pip install mlx-vlm")
        print("Detected Gemma4 VLM architecture")
        model, tokenizer = mlx_vlm_load(args.model)
    else:
        model, tokenizer = mlx_load(args.model, tokenizer_config={"trust_remote_code": True})

    # ---- Load datasets ----
    print("Loading datasets")
    train_set, valid_set, _ = load_dataset(args, tokenizer)

    # ---- LoRA setup ----
    model.freeze()
    if args.num_layers > len(model.layers):
        raise ValueError(
            f"Requested to train {args.num_layers} layers "
            f"but the model only has {len(model.layers)} layers."
        )

    if args.fine_tune_type == "full":
        for l in model.layers[-max(args.num_layers, 0) :]:
            l.unfreeze()
        args.lora_parameters = None
    elif args.fine_tune_type in ["lora", "dora"]:
        linear_to_lora_layers(
            model, args.num_layers, args.lora_parameters,
            use_dora=(args.fine_tune_type == "dora"),
        )
    else:
        raise ValueError(f"Unknown fine-tune-type: {args.fine_tune_type}")

    if args.resume_adapter_file is not None:
        print(f"Loading fine-tuned weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    # ---- Adapter output path ----
    from pathlib import Path
    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_path / "adapters.safetensors"

    from mlx_lm.utils import save_config
    save_config(vars(args), adapter_path / "adapter_config.json")

    # ---- Optimizer ----
    lr = build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
    optimizer_name = args.optimizer.lower()
    optimizer_config = args.optimizer_config.get(optimizer_name, {})
    opt_classes = {
        "adam": optim.Adam, "adamw": optim.AdamW, "muon": optim.Muon,
        "sgd": optim.SGD, "adafactor": optim.Adafactor,
    }
    opt = opt_classes[optimizer_name](learning_rate=lr, **optimizer_config)

    # ---- TrainingArgs ----
    from mlx_lm.tuner.trainer import TrainingArgs
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
        clear_cache_threshold=args.clear_cache_threshold,
    )

    # ---- AGR setup ----
    if args.agr:
        hidden_size = _resolve_hidden_size(model)
        agr = AttractorRepeller(
            dim=hidden_size,
            num_centers=args.agr_centers,
            ema=args.agr_ema,
            repeller_weight=args.agr_weight,
            min_distance=args.agr_min_dist,
        )

        # Resume AGR state if checkpoint exists
        agr_ckpt = args.agr_checkpoint or str(adapter_path / "agr_state")
        agr_meta = Path(agr_ckpt).with_suffix(".json")
        if agr_meta.exists():
            print(f"Loading AGR state from {agr_ckpt}")
            agr.load_checkpoint(agr_ckpt)

        print(f"\n[AGR] Attractor Geometry Repeller enabled")
        print(f"  Hidden dim: {hidden_size}")
        print(f"  Centers: {args.agr_centers}, EMA: {args.agr_ema}")
        print(f"  Lambda: {args.agr_lambda}, Weight: {args.agr_weight}")
        print(f"  Min distance: {args.agr_min_dist}\n")

        agr_train(
            model=model,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=training_args,
            agr=agr,
            agr_lambda=args.agr_lambda,
            agr_save_path=agr_ckpt,
        )
    else:
        # Standard training without AGR
        from mlx_lm.tuner.trainer import train as mlx_train
        print("Training configuration: standard AGR disabled")
        mlx_train(
            model=model,
            args=training_args,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
        )


if __name__ == "__main__":
    agr_main()
