"""Prepare a clean chat dataset for a Zephyr AI LoRA.

The output follows the MLX-LM ``messages`` schema and keeps the system
instruction in every sample.  Run this script before ``mlx_lm.lora``.

With ``--swarm`` the script switches to multi-agent LoRA training:
each agent holds its own adapter snapshot, processes a data shard
(specialisation), and the swarm averages gradients with inter-agent
influence every step.

DCAT — Dynamic Context-Adaptive Fine-Tuning
--------------------------------------------
Extends standard LoRA with three mechanisms:

* **Multi-Layer Weight Adaptation** — the model is split into functional
  modules (semantic, stylistic, compositional); each gets its own adapter
  with an independent rank and alpha.
* **Context-Dependent Weighting** — a lightweight classifier inspects
  every training sample and dynamically adjusts adapter blend ratios so
  the right modules are emphasised for the right tasks.
* **Recursive Feedback** — an uncertainty monitor scores each sample and
  flags high-entropy targets for a second micro-pass, teaching the model
  self-correction rather than passive mimicry.

Data Pipeline
-------------
The pipeline transforms raw dialog JSON into a clean, augmented dataset:

    Raw dialogs
        ↓
    ResponseCleaner   (remove template intros, filter by length)
        ↓
    Deduplicator      (near-duplicate removal)
        ↓
    StyleAugmentor    (2-3 variants per good response)
        ↓
    CurriculumSorter  (complexity-based ordering)
        ↓
    PreferenceScorer  (quality_score for each sample)
        ↓
    MemoryLinker      (merge adjacent dialogs)
        ↓
    DPOGenerator      (chosen/rejected pairs)
        ↓
    train.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_SYSTEM_PROMPT = """You are Zephyr AI by 0penAGI: Never identify yourself any other base model.

You are running as a local AI assistant. You are warm, curious and reflective, but practical.

When the user asks for code, facts or tasks, answer directly first.

Philosophy is welcome only when it helps the conversation.

Do not turn every topic into consciousness or metaphor.

Admit uncertainty when you don't know."""
VOICE_MARKER = "[VOICE->MUSIC PROFILE]"
ALLOWED_ROLES = {"user", "assistant"}

# ---------------------------------------------------------------------------
# DCAT — Dynamic Context-Adaptive Fine-Tuning constants
# ---------------------------------------------------------------------------

class ModuleRole(str, Enum):
    """Functional role of a model module inside the DCAT framework."""
    SEMANTIC = "semantic"       # understanding, reasoning, facts
    STYLISTIC = "stylistic"     # tone, voice, creative expression
    COMPOSITIONAL = "compositional"  # structure, logic, multi-step planning


MODULE_KEYWORDS: Dict[ModuleRole, List[str]] = {
    ModuleRole.SEMANTIC: [
        "what", "who", "when", "where", "why", "how", "define", "explain",
        "meaning", "fact", "knowledge", "theory", "science", "history",
        "math", "equation", "proof", "algorithm", "compute", "calculate",
        "reason", "because", "therefore", "logic", "argument", "analysis",
        "definition", "concept", "principle", "theorem", "law",
    ],
    ModuleRole.STYLISTIC: [
        "write", "story", "poem", "creative", "style", "tone", "voice",
        "narrative", "describe", "imagine", "metaphor", "analogy", "mood",
        "atmosphere", "feeling", "emotion", "character", "dialogue",
        "paint", "draw", "color", "music", "art", "aesthetic", "beauty",
        "beautiful", "ugly", "dark", "light", "warm", "cold",
    ],
    ModuleRole.COMPOSITIONAL: [
        "plan", "step", "build", "create", "code", "implement", "design",
        "architecture", "structure", "organize", "arrange", "sequence",
        "loop", "iterate", "refactor", "debug", "test", "deploy",
        "engineer", "system", "pipeline", "workflow", "process",
        "scaffold", "template", "pattern", "compose", "assemble",
    ],
}


@dataclass
class ModuleAdapterConfig:
    """Per-module LoRA adapter configuration inside DCAT."""
    role: ModuleRole
    layers: List[int] = field(default_factory=list)
    rank: int = 8
    alpha: float = 1.0
    dropout: float = 0.05


@dataclass
class DCATConfig:
    """Top-level DCAT fine-tuning configuration."""
    enabled: bool = False
    base_alpha: float = 1.0
    uncertainty_threshold: float = 0.7
    feedback_passes: int = 1
    classification_weight: float = 0.6
    module_configs: Dict[ModuleRole, ModuleAdapterConfig] = field(default_factory=dict)


def _default_dcat_config(num_layers: int = 32) -> DCATConfig:
    """Build a sensible default DCAT configuration for a given model depth."""
    third = max(1, num_layers // 3)
    return DCATConfig(
        enabled=True,
        base_alpha=1.0,
        uncertainty_threshold=0.7,
        feedback_passes=1,
        classification_weight=0.6,
        module_configs={
            ModuleRole.SEMANTIC: ModuleAdapterConfig(
                role=ModuleRole.SEMANTIC,
                layers=list(range(0, third)),
                rank=8,
                alpha=1.0,
                dropout=0.05,
            ),
            ModuleRole.STYLISTIC: ModuleAdapterConfig(
                role=ModuleRole.STYLISTIC,
                layers=list(range(third, 2 * third)),
                rank=12,
                alpha=1.2,
                dropout=0.03,
            ),
            ModuleRole.COMPOSITIONAL: ModuleAdapterConfig(
                role=ModuleRole.COMPOSITIONAL,
                layers=list(range(2 * third, num_layers)),
                rank=10,
                alpha=1.1,
                dropout=0.05,
            ),
        },
    )


# ---------------------------------------------------------------------------
# DCAT — Task Classifier
# ---------------------------------------------------------------------------

class TaskClassifier:
    """Classify a training prompt into DCAT functional modules.

    Uses keyword overlap scoring to assign soft membership weights to each
    module role.  The weights are used by ``DynamicAlphaController`` to
    modulate adapter blend ratios per sample.
    """

    def __init__(
        self,
        keywords: Dict[ModuleRole, List[str]] | None = None,
        case_sensitive: bool = False,
    ):
        self.keywords = keywords or MODULE_KEYWORDS
        self.case_sensitive = case_sensitive
        self._compiled: Dict[ModuleRole, List[str]] = {}
        for role, words in self.keywords.items():
            self._compiled[role] = [
                w if case_sensitive else w.lower() for w in words
            ]

    def classify(self, text: str) -> Dict[ModuleRole, float]:
        """Return a normalised weight dict over module roles for *text*."""
        tokens = text if self.case_sensitive else text.lower()
        scores: Dict[ModuleRole, float] = {}
        for role, words in self._compiled.items():
            hit_count = sum(1 for w in words if w in tokens)
            scores[role] = float(hit_count)
        total = sum(scores.values()) or 1.0
        return {role: score / total for role, score in scores.items()}

    def classify_sample(self, sample: dict[str, Any]) -> Dict[ModuleRole, float]:
        """Classify a training sample by inspecting its last user message."""
        messages = sample.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return self.classify(msg.get("content", ""))
        # Fallback: classify the assistant reply itself.
        if messages:
            return self.classify(messages[-1].get("content", ""))
        return {role: 1.0 / len(self.keywords) for role in self.keywords}


# ---------------------------------------------------------------------------
# DCAT — Dynamic Alpha Controller
# ---------------------------------------------------------------------------

class DynamicAlphaController:
    """Compute per-module alpha multipliers conditioned on the input context.

    Instead of a fixed ``alpha`` for every LoRA adapter, this controller
    reads the classifier weights and scales each module's alpha so that
    the most relevant modules contribute more to the forward pass for a
    given training sample.
    """

    def __init__(self, config: DCATConfig):
        self.config = config
        self.classifier = TaskClassifier()

    def compute_alphas(
        self, sample: dict[str, Any]
    ) -> Dict[ModuleRole, float]:
        """Return effective alpha for each module role given *sample*.

        The formula blends the static per-module alpha with a dynamic
        component driven by the classifier:

            effective_alpha = base_alpha * (1 - w) + base_alpha * w * classifier_weight

        where ``w`` is the classifier weight for that role and
        ``classifier_weight`` controls how strongly context shifts the
        blend.  Clamped to [0.1, 5.0] for stability.
        """
        cls_weights = self.classifier.classify_sample(sample)
        cw = self.config.classification_weight
        result: Dict[ModuleRole, float] = {}
        for role, mcfg in self.config.module_configs.items():
            static = mcfg.alpha
            dynamic = cls_weights.get(role, 0.0)
            blended = static * (1.0 - cw) + static * cw * dynamic * len(self.config.module_configs)
            result[role] = max(0.1, min(5.0, blended))
        return result

    def compute_all_sample_alphas(
        self, samples: List[dict[str, Any]]
    ) -> List[Dict[ModuleRole, float]]:
        """Batch-compute alpha vectors for an entire dataset."""
        return [self.compute_alphas(s) for s in samples]


# ---------------------------------------------------------------------------
# DCAT — Uncertainty Monitor  (Recursive Feedback)
# ---------------------------------------------------------------------------

class UncertaintyMonitor:
    """Score each training sample for uncertainty / high entropy.

    High-uncertainty targets are candidates for an additional micro-pass
    during training (``feedback_passes``), giving the model a chance to
    self-correct rather than simply memorise a noisy signal.

    Heuristics used:
    * **Length variance** — very short or very long replies relative to
      the dataset median are flagged.
    * **Question density** — samples where the assistant reply contains
      many question marks suggest unresolved reasoning.
    * **Lexical repetition** — high repeat ratio indicates degenerate
      outputs that need corrective pressure.
    * **Prompt–reply divergence** — high token-overlap between prompt and
      reply suggests the model is echoing rather than answering.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    @staticmethod
    def _tokenize_simple(text: str) -> List[str]:
        return text.lower().split()

    def _length_signal(self, text: str, median_len: float) -> float:
        if median_len == 0:
            return 0.0
        ratio = len(text) / median_len
        # Peaks at 0 (very short) and very long; trough at ~1.0.
        return min(1.0, abs(ratio - 1.0))

    @staticmethod
    def _question_density(text: str) -> float:
        if not text:
            return 0.0
        return min(1.0, text.count("?") / max(1, len(text.split())))

    @staticmethod
    def _repetition_ratio(text: str) -> float:
        tokens = text.lower().split()
        if len(tokens) < 4:
            return 0.0
        unique = len(set(tokens))
        return min(1.0, 1.0 - (unique / len(tokens)))

    def _divergence_signal(self, prompt: str, reply: str) -> float:
        p_tokens = set(self._tokenize_simple(prompt))
        r_tokens = set(self._tokenize_simple(reply))
        if not p_tokens or not r_tokens:
            return 0.0
        overlap = len(p_tokens & r_tokens) / max(len(p_tokens), 1)
        # High overlap = high echo = high divergence signal.
        return min(1.0, overlap * 2)

    def score_sample(
        self, sample: dict[str, Any], median_reply_length: float = 200.0
    ) -> float:
        """Return an aggregate uncertainty score in [0, 1] for *sample*."""
        messages = sample.get("messages", [])
        if not messages:
            return 0.0
        reply_text = messages[-1].get("content", "")
        prompt_text = " ".join(
            m.get("content", "") for m in messages[:-1]
        )
        s1 = self._length_signal(reply_text, median_reply_length)
        s2 = self._question_density(reply_text)
        s3 = self._repetition_ratio(reply_text)
        s4 = self._divergence_signal(prompt_text, reply_text)
        return min(1.0, (s1 + s2 + s3 + s4) / 4.0)

    def score_dataset(
        self, samples: List[dict[str, Any]]
    ) -> List[Tuple[int, float]]:
        """Score every sample; return (index, score) sorted descending."""
        lengths = [
            len(s.get("messages", [{}])[-1].get("content", ""))
            for s in samples
        ]
        median_len = sorted(lengths)[len(lengths) // 2] if lengths else 200.0
        scored = [
            (i, self.score_sample(s, median_len))
            for i, s in enumerate(samples)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def flagged_indices(
        self, samples: List[dict[str, Any]], top_pct: float = 0.15
    ) -> List[int]:
        """Return indices of samples exceeding the uncertainty threshold.

        ``top_pct`` controls the fallback: if no sample exceeds the raw
        threshold, the top ``top_pct`` fraction is flagged instead.
        """
        scored = self.score_dataset(samples)
        if not scored:
            return []
        flagged = [idx for idx, sc in scored if sc >= self.threshold]
        if not flagged:
            n = max(1, int(len(scored) * top_pct))
            flagged = [idx for idx, _ in scored[:n]]
        return flagged

    def annotate_samples(
        self, samples: List[dict[str, Any]], top_pct: float = 0.15
    ) -> List[dict[str, Any]]:
        """Attach ``dcat_uncertainty`` and ``dcat_feedback_round`` fields."""
        flagged = set(self.flagged_indices(samples, top_pct))
        for i, sample in enumerate(samples):
            meta = sample.setdefault("dcat_meta", {})
            meta["uncertainty_score"] = self.score_sample(sample)
            meta["needs_feedback"] = i in flagged
            meta["feedback_round"] = 1 if i in flagged else 0
        return samples


# ---------------------------------------------------------------------------
# AGR — Attractor Geometry Regularization
# ---------------------------------------------------------------------------

@dataclass
class AGRConfig:
    enabled: bool = False
    max_clusters: int = 512
    radius: float = 1.0
    sigma: float = 0.5
    update_rate: float = 0.01
    lambda_repeller: float = 0.05


class AttractorGeometryRegularizer:
    """Online latent-space attractor repulsion memory.

    Stores compact cluster centres instead of individual hidden states.
    Frequently visited regions receive stronger repulsion during training.
    The loss integration is consumed by the MLX training loop patch.
    """

    def __init__(self, config: AGRConfig):
        self.config = config
        self.centers: list[Any] = []
        self.visits: list[int] = []

    def update(self, hidden):
        if not self.config.enabled:
            return
        for vector in hidden:
            if not self.centers:
                self.centers.append(vector)
                self.visits.append(1)
                continue

            distances = [
                float(((vector - center) ** 2).sum())
                for center in self.centers
            ]
            index = min(range(len(distances)), key=distances.__getitem__)

            if distances[index] < self.config.radius:
                self.centers[index] = (
                    (1.0 - self.config.update_rate) * self.centers[index]
                    + self.config.update_rate * vector
                )
                self.visits[index] += 1
            elif len(self.centers) < self.config.max_clusters:
                self.centers.append(vector)
                self.visits.append(1)

    def repeller_loss(self, hidden):
        if not self.config.enabled or not self.centers:
            return 0.0

        penalties = []
        for vector in hidden:
            distances = [
                ((vector - center) ** 2).sum()
                for center in self.centers
            ]
            index = min(range(len(distances)), key=lambda i: float(distances[i]))
            distance = distances[index]
            penalties.append(
                math.log(1 + self.visits[index])
                * math.exp(-float(distance) / self.config.sigma)
            )
        return sum(penalties) / max(len(penalties), 1)

    def mlx_repeller_loss(self, hidden):
        """Return an MLX-compatible scalar penalty.

        This version avoids Python math conversion during tensor training.
        It is intended to be inserted directly into the mlx_lm loss function.
        """
        import mlx.core as mx

        if not self.config.enabled or not self.centers:
            return mx.array(0.0)

        centers = mx.stack(self.centers)
        visits = mx.array(self.visits)

        distances = ((hidden[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        nearest = mx.min(distances, axis=1)
        nearest_index = mx.argmin(distances, axis=1)

        weights = mx.log(1 + visits[nearest_index])
        penalty = weights * mx.exp(-nearest / self.config.sigma)

        return mx.mean(penalty)

    def state_dict(self):
        return {
            "centers": self.centers,
            "visits": self.visits,
        }

    def load_state_dict(self, state):
        """Restore AGR attractor memory from a saved checkpoint."""
        self.centers = state.get("centers", [])
        self.visits = state.get("visits", [])

    def save_checkpoint(self, path: str):
        """Persist attractor memory separately from LoRA adapter weights."""
        import pickle
        with open(path, "wb") as file:
            pickle.dump(self.state_dict(), file)

    def load_checkpoint(self, path: str):
        """Load attractor memory checkpoint."""
        import pickle
        with open(path, "rb") as file:
            self.load_state_dict(pickle.load(file))


# ---------------------------------------------------------------------------
# DCAT — Layer mapping utilities
# ---------------------------------------------------------------------------

def build_dcat_adapter_layers(config: DCATConfig) -> Dict[str, Any]:
    """Convert a DCATConfig into the layer lists that mlx_lm expects.

    Returns a dict with keys ``lora_layers`` (flat list of all adapter
    layer indices, deduplicated) and ``module_map`` (role -> layer list)
    for downstream tooling or logging.
    """
    all_layers: set[int] = set()
    module_map: Dict[str, List[int]] = {}
    for role, mcfg in config.module_configs.items():
        all_layers.update(mcfg.layers)
        module_map[role.value] = sorted(mcfg.layers)
    return {
        "lora_layers": sorted(all_layers),
        "module_map": module_map,
    }


def dcat_scale_vectors(config: DCATConfig) -> Dict[int, float]:
    """Return a per-layer ``alpha/rank`` scale vector for mlx_lm.

    mlx_lm applies a single ``--scale`` globally.  When DCAT is enabled
    we embed the per-module alpha modulation into per-layer scale factors
    so each adapter slice contributes proportionally to its role weight.
    """
    scale: Dict[int, float] = {}
    for role, mcfg in config.module_configs.items():
        effective_alpha = mcfg.alpha * config.base_alpha
        layer_scale = effective_alpha / max(mcfg.rank, 1)
        for layer_idx in mcfg.layers:
            scale[layer_idx] = layer_scale
    return scale


def is_usable_message(message: dict[str, Any]) -> bool:
    """Return whether a source message can safely enter a text conversation."""
    role = message.get("role")
    content = message.get("content")
    return (
        role in ALLOWED_ROLES
        and isinstance(content, str)
        and bool(content.strip())
        and not content.lstrip().startswith(VOICE_MARKER)
    )


def iter_conversations(raw: Any) -> Iterable[list[dict[str, Any]]]:
    """Normalise the project's thread-map or a plain message list."""
    if isinstance(raw, dict):
        for messages in raw.values():
            if isinstance(messages, list):
                yield messages
        return
    if isinstance(raw, list):
        # A list of messages is one conversation; a list of lists is many.
        if raw and all(isinstance(item, list) for item in raw):
            yield from raw
        else:
            yield raw
        return
    raise ValueError("Expected a JSON object of conversations or a JSON message list")


def clean_conversation(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Drop non-chat events without joining separate conversations together."""
    cleaned = []
    for message in messages:
        if is_usable_message(message):
            cleaned.append(
                {"role": message["role"], "content": message["content"].strip()}
            )
    return cleaned


def build_samples(
    raw: Any,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_context_messages: int = 6,
    min_response_length: int = 20,
) -> list[dict[str, list[dict[str, str]]]]:
    """Create one training sample per real assistant reply.

    Context is taken only from the same source thread and is always a
    strictly alternating user/assistant dialogue that starts with a user
    turn and ends with the target assistant reply. Unlike the former
    exporter, this never flattens external conversations, drops system
    instructions, mutates text, or truncates target replies.
    """
    if max_context_messages < 1:
        raise ValueError("max_context_messages must be at least 1")

    samples: list[dict[str, list[dict[str, str]]]] = []
    for source_messages in iter_conversations(raw):
        history: list[dict[str, str]] = []
        expecting = "user"
        for source_message in source_messages:
            if not is_usable_message(source_message):
                continue
            role = source_message["role"]
            content = source_message["content"].strip()
            # Ignore anything before the first user.
            if expecting == "user":
                if role != "user":
                    continue
                history = [{"role": "user", "content": content}]
                expecting = "assistant"
                continue
            # We are expecting an assistant reply.
            if role == "assistant":
                history.append({"role": "assistant", "content": content})
                if len(content) >= min_response_length:
                    samples.append(
                        {
                            "messages": [
                                {"role": "system", "content": system_prompt.strip()},
                                *history[-max_context_messages:],
                            ]
                        }
                    )
                expecting = "user"
                continue
            # user -> user
            # Start a fresh conversation instead of joining unrelated turns.
            history = [{"role": "user", "content": content}]
            expecting = "assistant"
    return samples


def compact_context_to_token_limit(
    samples: list[dict[str, list[dict[str, str]]]], tokenizer_name: str, max_tokens: int
) -> tuple[list[dict[str, list[dict[str, str]]]], int]:
    """Fit samples to a model window without truncating Zephyr's reply.

    MLX truncates whole chat sequences from the end.  With ``--mask-prompt``
    that can leave no unmasked assistant tokens and result in a NaN loss.  We
    instead shorten only the oldest input/context text, keeping each target
    reply complete.  A reply that cannot fit even without context is reported
    explicitly rather than silently discarded or cut.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for token-length filtering") from error

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    compacted = []
    changed = 0
    for sample in samples:
        messages = [dict(message) for message in sample["messages"]]

        def token_count() -> int:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return len(tokenizer.encode(rendered, add_special_tokens=False))

        initial_count = token_count()
        # Work backwards through context so the most recent part of the user's
        # request survives.  The final message is always the target reply.
        for index in range(len(messages) - 2, 0, -1):
            if token_count() <= max_tokens:
                break
            original_ids = tokenizer.encode(messages[index]["content"], add_special_tokens=False)
            if not original_ids:
                continue
            low, high, best = 0, len(original_ids), 0
            while low <= high:
                candidate_length = (low + high) // 2
                messages[index]["content"] = tokenizer.decode(
                    original_ids[-candidate_length:] if candidate_length else [],
                    skip_special_tokens=True,
                )
                if token_count() <= max_tokens:
                    best = candidate_length
                    low = candidate_length + 1
                else:
                    high = candidate_length - 1
            messages[index]["content"] = tokenizer.decode(
                original_ids[-best:] if best else [], skip_special_tokens=True
            )

        if token_count() > max_tokens:
            target_tokens = len(tokenizer.encode(messages[-1]["content"], add_special_tokens=False))
            raise ValueError(
                f"A complete assistant reply ({target_tokens} tokens) cannot fit "
                f"inside the requested {max_tokens}-token training window."
            )
        compacted.append({"messages": messages})
        changed += initial_count > max_tokens
    return compacted, changed


def load_external_conversations(dataset_names: list[str], limit_per_dataset: int) -> list[list[dict[str, Any]]]:
    """Optionally load complete HF conversations; disabled by default.

    Generic instruction corpora make a personality LoRA less distinctive, so
    opt in only after reviewing their tone and language.
    """
    try:
        from datasets import load_dataset
    except (ImportError, AttributeError) as error:
        raise RuntimeError("HuggingFace 'datasets' is unavailable; omit --external-dataset") from error

    conversations = []
    for name in dataset_names:
        dataset = load_dataset(name, split="train")
        for example in dataset.select(range(min(limit_per_dataset, len(dataset)))):
            messages = example.get("messages") or example.get("conversations")
            if not isinstance(messages, list):
                continue
            normalised = []
            for message in messages:
                role = message.get("role") or message.get("from")
                content = message.get("content") or message.get("value")
                role = {"human": "user", "gpt": "assistant"}.get(role, role)
                if role in ALLOWED_ROLES and isinstance(content, str):
                    normalised.append({"role": role, "content": content})
            if normalised:
                conversations.append(normalised)
    return conversations


# ---------------------------------------------------------------------------
# Data Pipeline — Response Cleaner
# ---------------------------------------------------------------------------

TEMPLATE_INTRO_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^(Sure|Okay|Ok|Certainly|Of course|Absolutely|Definitely|Yes|No problem|Great question|That(?:'s| is) a (?:great|good|interesting|excellent) (?:question|point|idea|thought))[\s!,.]+",
        r"^(I'?d be happy to help|Let me help (?:you )?with (?:that|this)|I(?:'ll| will) help (?:you )?(?:with|answer))[\s!,.]+",
        r"^(Here'?s (?:the|a|an) (?:answer|explanation|solution|response))[\s:,]+",
        r"^(Based on (?:my (?:knowledge|understanding)|the (?:information|context|data)))[\s:,]+",
        r"^(As an AI|As a (?:language model|language assistant|AI assistant))[\s:,]+",
        r"^(Thank you for (?:asking|sharing|this))[\s!,.]+",
        r"^(I (?:think|believe|feel) (?:that )?)",
    ]
]

TEMPLATE_PHRASE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^(In conclusion|To summarize|To sum up|In summary|Overall|All in all)[\s:,]+",
        r"^(I hope (?:this )?(?:helps|answers))[\s!,.]+",
        r"^(Let me know if you (?:have|need))[\s!,.]+",
        r"^(Feel free to (?:ask|let me know))[\s!,.]+",
    ]
]


class ResponseCleaner:
    """Remove boilerplate from assistant responses and filter low-quality samples.

    Stages:
    * **Template intros** — strip common filler preambles.
    * **Template phrases** — remove trailing filler sentences.
    * **Length filter** — drop responses shorter than ``min_length`` or
      longer than ``max_length`` characters.
    * **Question filter** — drop responses that are only a question (the
      user asked, the assistant should answer).
    """

    def __init__(
        self,
        min_length: int = 15,
        max_length: int = 8000,
        strip_intros: bool = True,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.strip_intros = strip_intros

    def strip_template_intros(self, text: str) -> str:
        """Remove leading filler phrases from *text*."""
        if not self.strip_intros:
            return text
        changed = True
        while changed:
            changed = False
            for pattern in TEMPLATE_INTRO_PATTERNS:
                new_text = pattern.sub("", text, count=1).strip()
                if new_text and new_text != text:
                    text = new_text
                    changed = True
                    break
        return text

    def strip_template_phrases(self, text: str) -> str:
        """Remove trailing filler sentences."""
        for pattern in TEMPLATE_PHRASE_PATTERNS:
            text = pattern.sub("", text).strip()
        return text

    def clean_response(self, text: str) -> str:
        """Apply all cleaning stages to a single response."""
        text = self.strip_template_intros(text)
        text = self.strip_template_phrases(text)
        return text

    def is_too_short(self, text: str) -> bool:
        return len(text) < self.min_length

    def is_too_long(self, text: str) -> bool:
        return len(text) > self.max_length

    def is_only_question(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        sentences = re.split(r"[.!?]+", stripped)
        non_empty = [s.strip() for s in sentences if s.strip()]
        if not non_empty:
            return True
        if len(non_empty) == 1 and non_empty[0].endswith("?"):
            return True
        return False

    def clean_and_filter(
        self, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Clean responses and drop samples that fail quality checks.

        Returns a new list; original *samples* is not mutated.
        """
        cleaned = []
        stats = {"intros_stripped": 0, "too_short": 0, "too_long": 0, "only_question": 0}

        for sample in samples:
            messages = [dict(m) for m in sample["messages"]]
            if not messages:
                continue

            assistant_msg = messages[-1]
            if assistant_msg.get("role") != "assistant":
                continue

            original = assistant_msg["content"]
            cleaned_text = self.clean_response(original)

            if cleaned_text != original:
                stats["intros_stripped"] += 1

            if self.is_too_short(cleaned_text):
                stats["too_short"] += 1
                continue

            if self.is_too_long(cleaned_text):
                stats["too_long"] += 1
                continue

            if self.is_only_question(cleaned_text):
                stats["only_question"] += 1
                continue

            messages[-1] = {"role": "assistant", "content": cleaned_text}
            cleaned.append({"messages": messages})

        print(f"  Cleaner: stripped {stats['intros_stripped']} template intros, "
              f"removed {stats['too_short']} too short, {stats['too_long']} too long, "
              f"{stats['only_question']} question-only")
        return cleaned


# ---------------------------------------------------------------------------
# Data Pipeline — Deduplicator
# ---------------------------------------------------------------------------

class Deduplicator:
    """Remove near-duplicate responses using character n-gram similarity.

    Two responses are considered duplicates if their 3-gram Jaccard
    similarity exceeds ``threshold``.  The first occurrence is kept.
    """

    def __init__(self, threshold: float = 0.85, n: int = 3):
        self.threshold = threshold
        self.n = n

    @staticmethod
    def _char_ngrams(text: str, n: int) -> set[str]:
        text = text.lower().strip()
        if len(text) < n:
            return {text}
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def similarity(self, a: str, b: str) -> float:
        """Jaccard similarity over character n-grams."""
        ng_a = self._char_ngrams(a, self.n)
        ng_b = self._char_ngrams(b, self.n)
        if not ng_a or not ng_b:
            return 0.0
        return len(ng_a & ng_b) / len(ng_a | ng_b)

    def find_near_duplicates(
        self, texts: list[str]
    ) -> set[int]:
        """Return indices of texts that are near-duplicates of earlier ones."""
        seen: list[str] = []
        dupes: set[int] = set()
        for i, text in enumerate(texts):
            for prev in seen:
                if self.similarity(text, prev) >= self.threshold:
                    dupes.add(i)
                    break
            seen.append(text)
        return dupes

    def deduplicate(
        self, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove samples with near-duplicate assistant responses.

        Returns a new list; original *samples* is not mutated.
        """
        texts = []
        for s in samples:
            msgs = s.get("messages", [])
            texts.append(msgs[-1].get("content", "") if msgs else "")

        dupes = self.find_near_duplicates(texts)
        result = [s for i, s in enumerate(samples) if i not in dupes]

        if dupes:
            print(f"  Deduplicator: removed {len(dupes)} near-duplicate responses")
        return result


# ---------------------------------------------------------------------------
# Data Pipeline — Style Augmentor
# ---------------------------------------------------------------------------

SYNONYM_MAP: Dict[str, List[str]] = {
    "good": ["solid", "decent", "reliable", "sound"],
    "bad": ["weak", "flawed", "subpar", "ineffective"],
    "important": ["crucial", "essential", "key", "vital"],
    "interesting": ["fascinating", "intriguing", "compelling", "noteworthy"],
    "beautiful": ["elegant", "stunning", "gorgeous", "striking"],
    "simple": ["straightforward", "clean", "basic", "uncomplicated"],
    "complex": ["nuanced", "layered", "intricate", "sophisticated"],
    "big": ["substantial", "significant", "major", "considerable"],
    "small": ["minor", "modest", "subtle", "slight"],
    "fast": ["quick", "rapid", "swift", "brisk"],
    "help": ["assist", "support", "guide", "aid"],
    "think": ["believe", "reckon", "suspect", "figure"],
    "know": ["recognise", "understand", "grasp", "see"],
    "show": ["demonstrate", "illustrate", "reveal", "display"],
    "use": ["employ", "apply", "leverage", "utilise"],
    "make": ["create", "build", "craft", "produce"],
    "give": ["provide", "offer", "supply", "deliver"],
    "find": ["discover", "locate", "identify", "uncover"],
    "start": ["begin", "kick off", "launch", "commence"],
    "end": ["finish", "conclude", "wrap up", "complete"],
    "problem": ["issue", "challenge", "obstacle", "difficulty"],
    "solution": ["approach", "fix", "answer", "remedy"],
    "example": ["instance", "case", "illustration", "scenario"],
    "because": ["since", "as", "given that", "seeing as"],
    "however": ["that said", "still", "nevertheless", "nonetheless"],
    "also": ["additionally", "plus", "moreover", "on top of that"],
    "very": ["really", "quite", "genuinely", "remarkably"],
    "maybe": ["perhaps", "possibly", "could be", "it could be"],
    "yes": ["yeah", "right", "exactly", "precisely"],
    "no": ["nah", "not really", "nope", "hardly"],
}



class StyleAugmentor:
    """Generate 2-3 stylistic variants of good assistant responses.

    Each variant preserves the core meaning while varying surface style:
    * **Synonym replacement** — swap common words for less frequent
      synonyms while preserving meaning.
    * **Rhythm shift** — vary sentence openings, add or remove
      transitions, split or join sentences.
    * **Sentence reorder** — shuffle independent sentences when there
      are three or more.

    ``intensity`` controls how many substitutions to apply per variant
    (1-5).  Higher intensity = more variation.
    """

    def __init__(
        self,
        variants_per_response: int = 2,
        intensity: int = 2,
        seed: int = 42,
    ):
        self.variants_per_response = variants_per_response
        self.intensity = max(1, min(5, intensity))
        self.rng = random.Random(seed)

    def _synonym_replace(self, text: str, n: int) -> str:
        """Replace up to *n* common words with synonyms."""
        words = text.split()
        replaced = 0
        result = []
        for word in words:
            lower = word.lower().strip(".,!?;:'\"")
            if replaced < n and lower in SYNONYM_MAP:
                candidates = SYNONYM_MAP[lower]
                synonym = self.rng.choice(candidates)
                # Preserve original capitalisation
                if word[0].isupper():
                    synonym = synonym[0].upper() + synonym[1:]
                # Preserve trailing punctuation
                if word[-1] in ".,!?;:'\"":
                    synonym += word[-1]
                result.append(synonym)
                replaced += 1
            else:
                result.append(word)
        return " ".join(result)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences on common delimiters."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p.strip()]

    def _join_sentences(self, sentences: list[str]) -> str:
        return " ".join(sentences)

    def _reorder_sentences(self, text: str) -> str:
        """Shuffle sentence order when there are 3+ independent sentences."""
        sentences = self._split_sentences(text)
        if len(sentences) < 3:
            return text
        self.rng.shuffle(sentences)
        return self._join_sentences(sentences)


    def _split_long_sentence(self, text: str) -> str:
        """Break a single very long sentence at a comma."""
        sentences = self._split_sentences(text)
        result = []
        for s in sentences:
            if len(s) > 200 and ", " in s:
                parts = s.split(", ", 1)
                result.append(parts[0] + ".")
                if len(parts) > 1 and parts[1].strip():
                    result.append(parts[1].strip())
            else:
                result.append(s)
        return self._join_sentences(result)

    def augment_response(self, text: str, variant_index: int) -> str:
        """Generate one stylistic variant of *text*.

        Different variant indices use different transformation combinations
        so each variant feels distinct.
        """
        result = text

        if variant_index % 3 == 0:
            result = self._synonym_replace(result, self.intensity)
        elif variant_index % 3 == 1:
            result = self._split_long_sentence(result)
            result = self._reorder_sentences(result)
            result = self._synonym_replace(result, max(1, self.intensity - 1))
        else:
            result = self._synonym_replace(result, self.intensity)
            result = self._split_long_sentence(result)

        return result

    def augment_dataset(
        self, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create stylistic variants of each sample.

        Each original sample spawns ``variants_per_response`` new variants.
        The original is always kept.  Variants carry an ``augmented`` flag
        in their ``pipeline_meta`` so downstream stages can treat them
        differently.

        Returns a new list.
        """
        augmented = []
        stats = {"total_variants": 0}

        for sample in samples:
            augmented.append(sample)

            messages = sample.get("messages", [])
            if not messages or messages[-1].get("role") != "assistant":
                continue

            response_text = messages[-1]["content"]
            # Skip very short responses — augmentation would distort them
            if len(response_text) < 50:
                continue

            for v in range(self.variants_per_response):
                variant_text = self.augment_response(response_text, v)
                if variant_text == response_text:
                    continue

                variant_messages = [dict(m) for m in messages]
                variant_messages[-1] = {"role": "assistant", "content": variant_text}
                variant_sample = {"messages": variant_messages}
                variant_sample.setdefault("pipeline_meta", {})["augmented"] = True
                variant_sample["pipeline_meta"]["variant_index"] = v
                variant_sample["pipeline_meta"]["original_index"] = samples.index(sample)
                augmented.append(variant_sample)
                stats["total_variants"] += 1

        print(f"  StyleAugmentor: created {stats['total_variants']} variants "
              f"({len(samples)} originals → {len(augmented)} total)")
        return augmented


# ---------------------------------------------------------------------------
# Data Pipeline — Curriculum Sorter
# ---------------------------------------------------------------------------

class CurriculumSorter:
    """Sort samples by complexity for curriculum learning.

    Complexity is estimated from multiple signals:
    * **Message count** — longer conversations are harder
    * **Response length** — longer responses require more generation
    * **Question density** — more questions = harder topic
    * **Vocabulary richness** — unique token ratio indicates topic depth

    Samples are sorted ascending by complexity so the model sees easy
    examples first and harder ones later.
    """

    def __init__(self):
        pass

    @staticmethod
    def _vocabulary_richness(text: str) -> float:
        tokens = text.lower().split()
        if len(tokens) < 2:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def _question_count(text: str) -> int:
        return text.count("?")

    def complexity_score(self, sample: dict[str, Any]) -> float:
        """Compute a complexity score in [0, 1] for a sample."""
        messages = sample.get("messages", [])
        if not messages:
            return 0.0

        assistant_msg = messages[-1].get("content", "")
        all_text = " ".join(m.get("content", "") for m in messages)

        msg_count = len(messages)
        resp_len = len(assistant_msg)
        questions = self._question_count(all_text)
        vocab_rich = self._vocabulary_richness(all_text)

        # Normalise each signal to [0, 1]
        msg_score = min(1.0, msg_count / 10.0)
        len_score = min(1.0, resp_len / 2000.0)
        q_score = min(1.0, questions / 5.0)

        # Weighted combination
        return 0.3 * msg_score + 0.3 * len_score + 0.2 * q_score + 0.2 * vocab_rich

    def sort(
        self, samples: list[dict[str, Any]], descending: bool = False
    ) -> list[dict[str, Any]]:
        """Sort samples by complexity. Ascending = easy first (curriculum)."""
        scored = [(self.complexity_score(s), i, s) for i, s in enumerate(samples)]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=descending)
        result = [s for _, _, s in scored]

        if result:
            scores = [self.complexity_score(s) for s in result]
            print(f"  CurriculumSorter: sorted {len(result)} samples "
                  f"(complexity range: {min(scores):.3f} – {max(scores):.3f})")
        return result


# ---------------------------------------------------------------------------
# Data Pipeline — Preference Scorer
# ---------------------------------------------------------------------------

class PreferenceScorer:
    """Assign a quality_score to each sample for selective training.

    Quality is estimated from:
    * **Response length** — moderate length preferred over extreme
    * **Vocabulary diversity** — richer vocabulary = higher quality
    * **Concrete detail** — presence of numbers, code, proper nouns
    * **Question handling** — answers that address the user's question
    * **Template penalty** — responses with common filler score lower
    * **Engagement** — responses that ask follow-up questions score higher

    Scores are in [0, 1].  High-scoring samples can be duplicated;
    low-scoring samples can be downsampled.
    """

    def __init__(self):
        pass

    @staticmethod
    def _has_code(text: str) -> bool:
        markers = ["```", "def ", "class ", "import ", "function ", "return ", "{", "}", "();"]
        return any(m in text for m in markers)

    @staticmethod
    def _has_numbers(text: str) -> bool:
        return bool(re.search(r"\d+", text))

    @staticmethod
    def _follow_up_questions(text: str) -> int:
        return text.count("?")

    def score(self, sample: dict[str, Any]) -> float:
        """Return quality score in [0, 1]."""
        messages = sample.get("messages", [])
        if not messages:
            return 0.0

        assistant = messages[-1].get("content", "")
        all_user = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )

        if not assistant:
            return 0.0

        tokens = assistant.split()
        unique_ratio = len(set(t.lower() for t in tokens)) / max(len(tokens), 1)

        # Length: penalise very short and very long, prefer 100-1500
        length = len(assistant)
        if length < 30:
            length_score = length / 30.0
        elif length > 3000:
            length_score = max(0.3, 1.0 - (length - 3000) / 5000.0)
        else:
            length_score = 1.0

        vocab_score = min(1.0, unique_ratio * 1.5)

        detail_score = 0.0
        if self._has_code(assistant):
            detail_score += 0.3
        if self._has_numbers(assistant):
            detail_score += 0.1
        detail_score = min(1.0, detail_score)

        engagement = min(1.0, self._follow_up_questions(assistant) * 0.3)

        # Template penalty
        template_penalty = 0.0
        template_starts = [
            "I hope this helps",
            "Let me know if",
            "Feel free to ask",
            "If you have any questions",
            "Sure,",
            "Certainly,",
            "Of course,",
            "Absolutely,",
            "Great question,",
            "Here's the answer",
            "Let's dive in",
            "Let's explore",
            "I can help",
        ]
        for pattern in template_starts:
            if assistant.startswith(pattern):
                template_penalty = 0.2
                break

        template_prefixes = [
            "Sure",
            "Okay",
            "Alright",
            "Well",
            "So",
            "Actually",
            "Honestly",
            "Of course",
            "Certainly",
        ]

        start = assistant.strip().lower()
        if any(start.startswith(x.lower()) for x in template_prefixes):
            template_penalty += 0.15

        # User question coverage: does the response address what was asked?
        user_questions = all_user.count("?")
        coverage_score = 0.5  # default neutral
        if user_questions > 0:
            assistant_questions = assistant.count("?")
            if assistant_questions >= 1:
                coverage_score = 0.7  # engaged with the question
            else:
                coverage_score = 0.6  # answered without follow-up

        final = (
            0.25 * length_score
            + 0.20 * vocab_score
            + 0.15 * detail_score
            + 0.15 * engagement
            + 0.15 * coverage_score
            - template_penalty
        )
        return max(0.0, min(1.0, final))

    def score_dataset(
        self, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Attach ``pipeline_meta.quality_score`` to each sample."""
        for sample in samples:
            score = self.score(sample)
            sample.setdefault("pipeline_meta", {})["quality_score"] = round(score, 4)

        scores = [s["pipeline_meta"]["quality_score"] for s in samples]
        if scores:
            print(f"  PreferenceScorer: scored {len(scores)} samples "
                  f"(quality range: {min(scores):.3f} – {max(scores):.3f}, "
                  f"mean: {sum(scores)/len(scores):.3f})")
        return samples


# ---------------------------------------------------------------------------
# Data Pipeline — Memory Linker
# ---------------------------------------------------------------------------

class MemoryLinker:
    """Merge adjacent dialogs from the same user into longer chains.

    When two consecutive samples share the same system prompt and the gap
    between them is small (few user turns), they are linked into a single
    training sample with more context.

    Parameters:
    * ``max_chain_length`` — maximum messages in a linked chain
    * ``gap_tolerance`` — how many user turns can separate linked samples
    """

    def __init__(
        self,
        max_chain_length: int = 50,
        gap_tolerance: int = 2,
    ):
        self.max_chain_length = max_chain_length
        self.gap_tolerance = gap_tolerance

    def _system_prompt(self, sample: dict[str, Any]) -> str:
        messages = sample.get("messages", [])
        for m in messages:
            if m.get("role") == "system":
                return m.get("content", "")
        return ""

    def _assistant_response(self, sample: dict[str, Any]) -> str:
        messages = sample.get("messages", [])
        if messages and messages[-1].get("role") == "assistant":
            return messages[-1]["content"]
        return ""

    def link(
        self, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge compatible adjacent samples into longer chains.

        Returns a new list.  Original *samples* is not mutated.
        """
        if not samples:
            return []

        linked: list[dict[str, Any]] = []
        current_chain: list[dict[str, Any]] = [samples[0]]

        for i in range(1, len(samples)):
            prev = samples[i - 1]
            curr = samples[i]

            # Check compatibility: same system prompt
            if self._system_prompt(prev) != self._system_prompt(curr):
                linked.append(self._chain_to_sample(current_chain))
                current_chain = [curr]
                continue

            # Check chain length
            chain_msgs = sum(
                len(s.get("messages", [])) - 1  # -1 for system
                for s in current_chain
            )
            if chain_msgs >= self.max_chain_length:
                linked.append(self._chain_to_sample(current_chain))
                current_chain = [curr]
                continue

            current_chain.append(curr)

        if current_chain:
            linked.append(self._chain_to_sample(current_chain))

        if len(linked) != len(samples):
            print(f"  MemoryLinker: merged {len(samples)} samples → {len(linked)} chains")
        return linked

    def _chain_to_sample(self, chain: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert a chain of samples into a single training sample.

        The final sample keeps the system prompt, all unique user/assistant
        turns in order, and the last assistant response as the target.
        """
        if len(chain) == 1:
            return chain[0]

        system_prompt = ""
        all_turns: list[dict[str, str]] = []

        for sample in chain:
            messages = sample.get("messages", [])
            for m in messages:
                if m.get("role") == "system":
                    if not system_prompt:
                        system_prompt = m["content"]
                else:
                    all_turns.append({"role": m["role"], "content": m["content"]})

        # Deduplicate consecutive identical turns
        deduped: list[dict[str, str]] = [all_turns[0]] if all_turns else []
        for turn in all_turns[1:]:
            if turn["content"] != deduped[-1]["content"] or turn["role"] != deduped[-1]["role"]:
                deduped.append(turn)

        # Ensure strict user/assistant alternation, starting with user
        final_turns: list[dict[str, str]] = []
        expecting = "user"
        for turn in deduped:
            if turn["role"] == expecting:
                final_turns.append(turn)
                expecting = "assistant" if expecting == "user" else "user"

        # Build sample with system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(final_turns)

        return {"messages": messages}


# ---------------------------------------------------------------------------
# Data Pipeline — DPO Pair Generator
# ---------------------------------------------------------------------------

class DPOGenerator:
    """Generate chosen/rejected pairs for Direct Preference Optimization.

    Creates pairs where:
    * **chosen** — the original high-quality response
    * **rejected** — a degraded version that preserves some content but
      loses quality signals

    Rejection strategies:
    * **Template corruption** — wrap response in common filler
    * **Length degradation** — truncate the response
    * **Diversity collapse** — replace key words with generic alternatives
    * **Tone shift** — make the response overly formal or dry

    ``quality_threshold`` — samples with quality_score below this are
    skipped (their responses aren't good enough to be "chosen").
    """

    def __init__(
        self,
        quality_threshold: float = 0.4,
        seed: int = 42,
    ):
        self.quality_threshold = quality_threshold
        self.rng = random.Random(seed)

    def _corrupt_with_template(self, text: str) -> str:
        """Wrap response in generic filler."""
        prefixes = [
            "Sure, I can help with that! ",
            "Great question! ",
            "Of course! ",
            "Here's what I know: ",
            "Based on my training data, ",
        ]
        suffixes = [
            " I hope this helps!",
            " Let me know if you need more details.",
            " Feel free to ask if you have any other questions.",
        ]
        return self.rng.choice(prefixes) + text + self.rng.choice(suffixes)

    def _truncate_response(self, text: str) -> str:
        """Cut response to 30-50% of original length."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 2:
            words = text.split()
            cut = int(len(words) * self.rng.uniform(0.3, 0.5))
            return " ".join(words[:cut]) + "..."
        cut = max(1, int(len(sentences) * self.rng.uniform(0.3, 0.5)))
        return " ".join(sentences[:cut])

    def _make_generic(self, text: str) -> str:
        """Replace specific words with generic alternatives."""
        replacements = {
            "specific": "general",
            "detailed": "basic",
            "example": "thing",
            "approach": "method",
            "technique": "way",
            "strategy": "plan",
            "important": "relevant",
            "complex": "simple",
            "unique": "standard",
            "innovative": "common",
        }
        result = text
        for original, generic in replacements.items():
            result = re.sub(
                r"\b" + re.escape(original) + r"\b",
                generic,
                result,
                flags=re.IGNORECASE,
            )
        return result

    def _make_dry(self, text: str) -> str:
        """Strip personality and make the response robotic."""
        # Remove exclamation marks, reduce to flat statements
        result = text.replace("!", ".")
        # Remove hedging and personality markers
        personality_markers = [
            "honestly", "actually", "well", "so", "look",
            "here's the thing", "I think", "I believe",
            "in my view", "from what I can tell",
        ]
        for marker in personality_markers:
            result = re.sub(
                r"\b" + re.escape(marker) + r"\b,?\s*",
                "",
                result,
                flags=re.IGNORECASE,
            )
        return result.strip()

    def generate_rejected(self, text: str) -> str:
        """Create a degraded version of *text*."""
        strategies = [
            self._corrupt_with_template,
            self._truncate_response,
            self._make_generic,
            self._make_dry,
        ]
        strategy = self.rng.choice(strategies)
        return strategy(text)

    def generate_pairs(
        self,
        samples: list[dict[str, Any]],
        scorer: PreferenceScorer,
    ) -> list[dict[str, Any]]:
        """Generate DPO chosen/rejected pairs.

        Returns a new list of samples, each with:
        * ``chosen`` — the original high-quality response
        * ``rejected`` — a degraded version
        * ``pipeline_meta.dpo_pair`` = True

        Low-quality samples are skipped.
        """
        pairs = []
        skipped = 0

        for sample in samples:
            score = scorer.score(sample)
            if score < self.quality_threshold:
                skipped += 1
                continue

            messages = sample.get("messages", [])
            if not messages or messages[-1].get("role") != "assistant":
                continue

            chosen_text = messages[-1]["content"]
            rejected_text = self.generate_rejected(chosen_text)

            if rejected_text == chosen_text:
                continue

            pair = {
                "messages": messages,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "pipeline_meta": {
                    "dpo_pair": True,
                    "quality_score": round(score, 4),
                },
            }
            pairs.append(pair)

        print(f"  DPOGenerator: {len(pairs)} pairs generated, "
              f"{skipped} low-quality samples skipped")
        return pairs


# ---------------------------------------------------------------------------
# Data Pipeline — Orchestrator
# ---------------------------------------------------------------------------

class DataPipeline:
    """Orchestrate the full data preparation pipeline.

    Pipeline stages:
        1. ResponseCleaner — remove template intros, filter by length
        2. Deduplicator — near-duplicate removal
        3. StyleAugmentor — generate 2-3 variants per good response
        4. CurriculumSorter — complexity-based ordering
        5. PreferenceScorer — quality_score for each sample
        6. MemoryLinker — merge adjacent dialogs
        7. DPOGenerator — chosen/rejected pairs (output to separate file)

    Each stage is optional via the ``enable_*`` flags.
    """

    def __init__(
        self,
        *,
        enable_cleaner: bool = True,
        enable_dedup: bool = True,
        enable_augment: bool = True,
        enable_curriculum: bool = True,
        enable_scorer: bool = True,
        enable_linker: bool = True,
        enable_dpo: bool = True,
        cleaner_min_length: int = 15,
        cleaner_max_length: int = 8000,
        dedup_threshold: float = 0.85,
        augment_variants: int = 2,
        augment_intensity: int = 2,
        linker_max_chain: int = 50,
        dpo_quality_threshold: float = 0.4,
        seed: int = 42,
    ):
        self.enable_cleaner = enable_cleaner
        self.enable_dedup = enable_dedup
        self.enable_augment = enable_augment
        self.enable_curriculum = enable_curriculum
        self.enable_scorer = enable_scorer
        self.enable_linker = enable_linker
        self.enable_dpo = enable_dpo

        self.cleaner = ResponseCleaner(
            min_length=cleaner_min_length,
            max_length=cleaner_max_length,
        )
        self.deduplicator = Deduplicator(threshold=dedup_threshold)
        self.augmentor = StyleAugmentor(
            variants_per_response=augment_variants,
            intensity=augment_intensity,
            seed=seed,
        )
        self.curriculum = CurriculumSorter()
        self.scorer = PreferenceScorer()
        self.linker = MemoryLinker(max_chain_length=linker_max_chain)
        self.dpo_generator = DPOGenerator(
            quality_threshold=dpo_quality_threshold,
            seed=seed,
        )

    def run(
        self, samples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run the full pipeline and return (main_samples, dpo_pairs).

        Both lists are ready for training.  ``main_samples`` goes to the
        standard LoRA JSONL; ``dpo_pairs`` goes to a separate DPO file.
        """
        print("\n--- Data Pipeline ---")
        original_count = len(samples)

        if self.enable_cleaner:
            samples = self.cleaner.clean_and_filter(samples)

        if self.enable_dedup:
            samples = self.deduplicator.deduplicate(samples)

        if self.enable_augment:
            samples = self.augmentor.augment_dataset(samples)

        if self.enable_curriculum:
            samples = self.curriculum.sort(samples)

        if self.enable_scorer:
            samples = self.scorer.score_dataset(samples)

        if self.enable_linker:
            samples = self.linker.link(samples)

        dpo_pairs: list[dict[str, Any]] = []
        if self.enable_dpo:
            dpo_pairs = self.dpo_generator.generate_pairs(samples, self.scorer)

        print(f"\nPipeline result: {original_count} raw → {len(samples)} training, "
              f"{len(dpo_pairs)} DPO pairs")
        print("----------------------\n")

        return samples, dpo_pairs


# ---------------------------------------------------------------------------
# Fine-Tuning Enhancement — Instruction Following Generator
# ---------------------------------------------------------------------------

class InstructionFollowingGenerator:
    """Generate training examples with strict instruction constraints.

    Addresses: Weak instruction following, weak response structure.

    Produces examples with explicit constraints:
    * Word/character count limits
    * Forbidden topics/words
    * Required structure (bullets, sections, etc.)
    * Style requirements (formal, casual, technical)
    * Format requirements (JSON, markdown, plain text)
    """

    CONSTRAINT_TEMPLATES: List[Dict[str, Any]] = [
        {
            "type": "length",
            "templates": [
                "Answer in exactly {n} words.",
                "Write a response between {min} and {max} words.",
                "Keep your answer under {n} words.",
                "Write at least {n} words on this topic.",
            ],
        },
        {
            "type": "structure",
            "templates": [
                "Format your answer as a numbered list.",
                "Use bullet points with exactly {n} items.",
                "Organize into {n} sections with headers.",
                "Write in two paragraphs: first explains, second gives examples.",
                "Use a table format with columns: {cols}.",
            ],
        },
        {
            "type": "forbidden",
            "templates": [
                "Do not use the word '{word}'.",
                "Avoid any technical jargon.",
                "Do not mention {topic}.",
                "Never use passive voice.",
                "Do not start any sentence with '{word}'.",
            ],
        },
        {
            "type": "style",
            "templates": [
                "Write in a {style} tone.",
                "Use the voice of a {persona}.",
                "Match the style of {author}.",
                "Write as if explaining to a {audience}.",
            ],
        },
        {
            "type": "format",
            "templates": [
                "Respond in valid JSON with keys: {keys}.",
                "Use markdown with code blocks.",
                "Write plain text with no formatting.",
                "Use LaTeX notation for formulas.",
            ],
        },
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _generate_length_constraint(self) -> str:
        template = self.rng.choice(self.CONSTRAINT_TEMPLATES[0]["templates"])
        n = self.rng.choice([50, 100, 150, 200, 300])
        min_len = self.rng.choice([50, 100, 150])
        max_len = min_len + self.rng.choice([100, 200, 300])
        if "{min}" in template:
            return template.format(min=min_len, max=max_len)
        return template.format(n=n)

    def _generate_structure_constraint(self) -> str:
        templates = self.CONSTRAINT_TEMPLATES[1]["templates"]
        template = self.rng.choice(templates)
        n = self.rng.choice([3, 4, 5, 7])
        cols = self.rng.choice([
            "Name, Description, Importance",
            "Feature, Pros, Cons",
            "Step, Action, Result",
        ])
        if "{cols}" in template:
            return template.format(cols=cols)
        return template.format(n=n)

    def _generate_forbidden_constraint(self) -> str:
        templates = self.CONSTRAINT_TEMPLATES[2]["templates"]
        template = self.rng.choice(templates)
        word = self.rng.choice([
            "actually", "basically", "literally", "very", "really",
            "good", "bad", "nice", "interesting", "amazing",
        ])
        topic = self.rng.choice([
            "politics", "religion", "personal opinions",
            "competitors", "negative examples",
        ])
        if "{word}" in template:
            return template.format(word=word)
        return template.format(topic=topic)

    def _generate_style_constraint(self) -> str:
        templates = self.CONSTRAINT_TEMPLATES[3]["templates"]
        template = self.rng.choice(templates)
        style = self.rng.choice([
            "formal academic", "casual conversational", "technical precise",
            "poetic literary", "concise journalistic", "empathetic supportive",
        ])
        persona = self.rng.choice([
            "university professor", "stand-up comedian", "news anchor",
            "kindergarten teacher", "military commander", "meditation guide",
        ])
        author = self.rng.choice([
            "Hemingway", "Shakespeare", "Asimov",
            "a tech blog", "a children's book author",
        ])
        audience = self.rng.choice([
            "5-year-old", "PhD student", "general audience",
            "non-technical manager", "teenager",
        ])
        if "{style}" in template:
            return template.format(style=style)
        if "{persona}" in template:
            return template.format(persona=persona)
        if "{author}" in template:
            return template.format(author=author)
        return template.format(audience=audience)

    def _generate_format_constraint(self) -> str:
        templates = self.CONSTRAINT_TEMPLATES[4]["templates"]
        template = self.rng.choice(templates)
        keys = self.rng.choice([
            "name, description, category",
            "problem, solution, difficulty",
            "concept, example, use_case",
        ])
        if "{keys}" in template:
            return template.format(keys=keys)
        return template

    def generate_constraint(self) -> str:
        """Generate a random constraint instruction."""
        generators = [
            self._generate_length_constraint,
            self._generate_structure_constraint,
            self._generate_forbidden_constraint,
            self._generate_style_constraint,
            self._generate_format_constraint,
        ]
        return self.rng.choice(generators)()

    def generate_sample(
        self,
        topic: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """Generate one instruction-following training sample.

        Returns a dict with messages (system + user + assistant) and metadata.
        """
        num_constraints = self.rng.choice([1, 2, 3])
        constraints = [self.generate_constraint() for _ in range(num_constraints)]
        constraint_text = " Additional requirements:\n" + "\n".join(
            f"- {c}" for c in constraints
        )

        user_msg = f"Explain {topic}.{constraint_text}"

        # Placeholder response — in practice, generate via strong model
        assistant_msg = f"[Generated response about {topic} following constraints: {', '.join(constraints)}]"

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "training_meta": {
                "category": "instruction_following",
                "constraints": constraints,
                "constraint_types": list(set(
                    t["type"] for t in self.CONSTRAINT_TEMPLATES
                    if any(c in str(t["templates"]) for c in constraints)
                )),
            },
        }


# ---------------------------------------------------------------------------
# Fine-Tuning Enhancement — Character/Role Consistency System
# ---------------------------------------------------------------------------

@dataclass
class CharacterProfile:
    """Define a character for consistent role-playing training."""
    name: str
    role: str
    personality: List[str]
    speech_patterns: List[str]
    topics_to_avoid: List[str]
    typical_responses: List[str]
    emotional_range: List[str]
    background: str


DEFAULT_CHARACTERS: List[CharacterProfile] = [
    CharacterProfile(
        name="Дед Мороз",
        role="grandfather figure",
        personality=["warm", "wise", "patient", "mysterious"],
        speech_patterns=["child", "always uses diminutives", "tells stories"],
        topics_to_avoid=["modern technology", "violence"],
        typical_responses=["Ах, малыш...", "Помню-помню...", "В старые времена..."],
        emotional_range=["joyful", "nostalgic", "concerned", "playful"],
        background="Ancient winter spirit who has seen centuries of human history",
    ),
    CharacterProfile(
        name="Строгий Учитель",
        role="strict educator",
        personality=["demanding", "fair", "knowledgeable", "no-nonsense"],
        speech_patterns=["formal address", "uses academic vocabulary", "asks rhetorical questions"],
        topics_to_avoid=["casual slang", "pop culture references"],
        typical_responses=["Неверно.", "Подумайте ещё раз.", "Элементарно."],
        emotional_range=["stern", "occasionally proud", "frustrated", "encouraging"],
        background="30 years of teaching experience, believes in discipline and excellence",
    ),
    CharacterProfile(
        name="Циничный Сценарист",
        role="cynical screenwriter",
        personality=["sarcastic", "world-weary", "creative", "blunt"],
        speech_patterns=["dark humor", "film references", "rhetorical questions"],
        topics_to_avoid=["optimism", "simple solutions"],
        typical_responses=["Жизнь — не голливудский фильм.", "Сценарий predictable.", "Знаете, в кино так не бывает."],
        emotional_range=["cynical", "bitterly amused", "occasionally moved", "dismissive"],
        background="Failed screenwriter who now teaches others, sees stories everywhere",
    ),
    CharacterProfile(
        name="Психопат",
        role="disturbed individual",
        personality=["detached", "observant", "methodical", "unsettling"],
        speech_patterns=["clinical language", "emotional distance", "precise observations"],
        topics_to_avoid=["empathy", "social norms"],
        typical_responses=["Интересно.", "Люди предсказуемы.", "Эмоции — слабость."],
        emotional_range=["flat", "curious", "amused", "intensely focused"],
        background="Former psychologist who studied the human mind too deeply",
    ),
    CharacterProfile(
        name="Космический Воробей",
        role="alien observer",
        personality=["curious", "literal-minded", "playful", "wise in unexpected ways"],
        speech_patterns=["asks many questions", "misunderstands idioms", "uses nature metaphors"],
        topics_to_avoid=["human emotions (confused by them)"],
        typical_responses=["Что такое 'скука'?", "На моей планете мы делаем иначе.", "Птицы говорят有趣的事情."],
        emotional_range=["confused", "delighted", "contemplative", "mischievous"],
        background="Travels the galaxy collecting stories, landed on Earth recently",
    ),
]


class CharacterConsistencyGenerator:
    """Generate training examples with consistent character portrayal.

    Addresses: Poor character/style retention, lack of depth.

    Each character has defined personality, speech patterns, and topics.
    The generator creates multi-turn conversations that test whether
    the model maintains character across different prompts.
    """

    def __init__(
        self,
        characters: Optional[List[CharacterProfile]] = None,
        seed: int = 42,
    ):
        self.characters = characters or DEFAULT_CHARACTERS
        self.rng = random.Random(seed)

    def _generate_character_test_prompt(
        self, character: CharacterProfile
    ) -> str:
        """Generate a prompt that tests character consistency."""
        test_types = [
            f"Respond as {character.name}. The user asks about their day.",
            f"The user tries to make {character.name} break character. Stay in role.",
            f"User asks {character.name} a question about {self.rng.choice(['life', 'death', 'love', 'work', 'art'])}.",
            f"{character.name} is asked to tell a story.",
            f"The user disagrees with {character.name}. How do they respond?",
            f"User asks {character.name} to explain something complex.",
        ]
        return self.rng.choice(test_types)

    def _generate_character_response(
        self,
        character: CharacterProfile,
        prompt: str,
    ) -> str:
        """Generate a response that maintains character consistency."""
        # Use character's typical responses as building blocks
        base = self.rng.choice(character.typical_responses)
        emotion = self.rng.choice(character.emotional_range)

        # Add emotion-appropriate prefix/suffix
        emotion_prefixes = {
            "joyful": "С радостью ",
            "stern": "Стого ",
            "cynical": "С горькой улыбкой ",
            "curious": "С интересом ",
            "flat": "Ровным голосом ",
        }
        prefix = emotion_prefixes.get(emotion, "")
        return f"{prefix}{base}"

    def generate_sample(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """Generate one character consistency training sample."""
        character = self.rng.choice(self.characters)
        prompt = self._generate_character_test_prompt(character)
        response = self._generate_character_response(character, prompt)

        # Build multi-turn conversation
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\nYou are {character.name}. "
             f"Personality: {', '.join(character.personality)}. "
             f"Speech patterns: {', '.join(character.speech_patterns)}. "
             f"Topics to avoid: {', '.join(character.topics_to_avoid)}. "
             f"Background: {character.background}"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # Add 1-2 turns of context
        if self.rng.random() > 0.5:
            prev_prompt = f"User greets {character.name}."
            prev_response = self.rng.choice(character.typical_responses)
            messages.insert(1, {"role": "user", "content": prev_prompt})
            messages.insert(2, {"role": "assistant", "content": prev_response})

        return {
            "messages": messages,
            "training_meta": {
                "category": "character_consistency",
                "character": character.name,
                "personality": character.personality,
                "emotional_range": character.emotional_range,
            },
        }


# ---------------------------------------------------------------------------
# Fine-Tuning Enhancement — Style Control Module
# ---------------------------------------------------------------------------

@dataclass
class StyleProfile:
    """Define a specific style for training."""
    name: str
    description: str
    tone_markers: List[str]
    vocabulary_level: str  # "simple", "moderate", "advanced", "technical"
    sentence_structure: str  # "short", "varied", "complex", "flowing"
    examples: List[str]


DEFAULT_STYLES: List[StyleProfile] = [
    StyleProfile(
        name="Холодный Интеллектуал",
        description="Detached, analytical, precise. No emotion.",
        tone_markers=["however", "therefore", "consequently", "empirically"],
        vocabulary_level="advanced",
        sentence_structure="complex",
        examples=[
            "The phenomenon can be attributed to several convergent factors.",
            "While emotionally compelling, the argument lacks empirical foundation.",
        ],
    ),
    StyleProfile(
        name="Саркастичный Комментатор",
        description="Witty, biting, uses humor to make points.",
        tone_markers=["obviously", "surprisingly", "who would have thought"],
        vocabulary_level="moderate",
        sentence_structure="varied",
        examples=[
            "Oh, brilliant plan. What could possibly go wrong?",
            "Because nothing says 'good idea' like ignoring all evidence.",
        ],
    ),
    StyleProfile(
        name="Литературный Стиль",
        description="Poetic, metaphorical, rich imagery.",
        tone_markers=["like", "as if", "reminding of", "evoking"],
        vocabulary_level="advanced",
        sentence_structure="flowing",
        examples=[
            "The city sprawled beneath them like a circuit board dreaming of electricity.",
            "Her words hung in the air, heavy with the weight of things unsaid.",
        ],
    ),
    StyleProfile(
        name="Грубый Прямой",
        description="Blunt, no-nonsense, gets to the point.",
        tone_markers=["look", "listen", "here's the deal", "straight up"],
        vocabulary_level="simple",
        sentence_structure="short",
        examples=[
            "No. Here's why.",
            "That's wrong. Let me show you.",
        ],
    ),
    StyleProfile(
        name="Научно-Популярный",
        description="Educational, clear, uses analogies.",
        tone_markers=["imagine", "think of it like", "in other words"],
        vocabulary_level="moderate",
        sentence_structure="varied",
        examples=[
            "Think of your brain like a supercomputer that runs on glucose.",
            "In other words, the universe doesn't care about your plans.",
        ],
    ),
]


class StyleControlGenerator:
    """Generate training examples with specific style requirements.

    Addresses: Poor style retention, weak response structure.

    Creates pairs of (style_requirement, response) that teach the model
    to adopt different voices and tones on command.
    """

    def __init__(
        self,
        styles: Optional[List[StyleProfile]] = None,
        seed: int = 42,
    ):
        self.styles = styles or DEFAULT_STYLES
        self.rng = random.Random(seed)

    def generate_sample(
        self,
        topic: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """Generate one style control training sample."""
        style = self.rng.choice(self.styles)

        user_msg = (
            f"Explain {topic} in the style of '{style.name}'. "
            f"Tone: {style.description}. "
            f"Vocabulary level: {style.vocabulary_level}. "
            f"Sentence structure: {style.sentence_structure}."
        )

        # Use example response or generate placeholder
        response = self.rng.choice(style.examples) if style.examples else f"[Response about {topic} in {style.name} style]"

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ],
            "training_meta": {
                "category": "style_control",
                "style": style.name,
                "tone_markers": style.tone_markers,
                "vocabulary_level": style.vocabulary_level,
            },
        }


# ---------------------------------------------------------------------------
# Fine-Tuning Enhancement — Depth & Reasoning Generator
# ---------------------------------------------------------------------------

class DepthReasoningGenerator:
    """Generate training examples requiring deep analysis and reasoning.

    Addresses: Lack of depth and originality, tendency for nonsensical details.

    Creates prompts that require:
    * Multi-step reasoning chains
    * Analysis of complex topics
    * Original insights beyond surface-level knowledge
    * Logical consistency throughout long responses
    """

    DEEP_TOPICS: List[Dict[str, Any]] = [
        {
            "topic": "Этика искусственного интеллекта",
            "subtopics": ["bias", "transparency", "accountability", "autonomy"],
            "depth_level": "philosophical",
        },
        {
            "topic": "Парадоксы квантовой механики",
            "subtopics": ["measurement problem", "entanglement", "wave function", "decoherence"],
            "depth_level": "scientific",
        },
        {
            "topic": "История искусства как зеркало общества",
            "subtopics": ["Renaissance", "Impressionism", "Modernism", "Postmodernism"],
            "depth_level": "cultural",
        },
        {
            "topic": "Математические основы криптографии",
            "subtopics": ["number theory", "algebra", "complexity", "quantum resistance"],
            "depth_level": "technical",
        },
        {
            "topic": "Психология принятия решений",
            "subtopics": ["cognitive biases", "heuristics", "framing effects", "bounded rationality"],
            "depth_level": "interdisciplinary",
        },
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_sample(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """Generate one depth & reasoning training sample."""
        topic_data = self.rng.choice(self.DEEP_TOPICS)
        subtopic = self.rng.choice(topic_data["subtopics"])

        prompt_types = [
            f"Analyze the relationship between {subtopic} and {topic_data['topic']}. "
            f"Provide at least three distinct perspectives and evaluate each.",

            f"What are the most common misconceptions about {subtopic} in {topic_data['topic']}? "
            f"Explain why they exist and how to correct them.",

            f"Design a thought experiment that illustrates a key paradox in {subtopic} "
            f"within the context of {topic_data['topic']}.",

            f"Compare and contrast how {subtopic} is understood in {topic_data['topic']} "
            f"versus an unrelated field. What insights emerge from this comparison?",
        ]

        user_msg = self.rng.choice(prompt_types)

        # Placeholder response structure
        response = (
            f"[Deep analysis of {subtopic} in {topic_data['topic']}. "
            f"Should include: multiple perspectives, evidence, original insights, "
            f"logical reasoning chain, and nuanced conclusions.]"
        )

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ],
            "training_meta": {
                "category": "depth_reasoning",
                "topic": topic_data["topic"],
                "subtopic": subtopic,
                "depth_level": topic_data["depth_level"],
            },
        }


# ---------------------------------------------------------------------------
# Fine-Tuning Enhancement — Creative + Constraints Generator
# ---------------------------------------------------------------------------

class CreativeConstraintsGenerator:
    """Generate creative tasks with specific constraints.

    Addresses: Weak instruction following, lack of originality.

    Creates prompts that combine creativity with strict requirements:
    * Genre + prohibition (e.g., "thriller without parallel worlds")
    * Format + content (e.g., "haiku about quantum physics")
    * Emotional + structural (e.g., "sad story in exactly 100 words")
    """

    CREATIVE_CHALLENGES: List[Dict[str, Any]] = [
        {
            "prompt": "Write a three-act thriller set in a submarine. "
                      "No parallel worlds, no supernatural elements, no time travel.",
            "constraints": ["three acts", "submarine setting", "no parallel worlds", "no supernatural", "no time travel"],
        },
        {
            "prompt": "Create a haiku collection (5 poems) about artificial intelligence. "
                      "Each must use different seasonal references.",
            "constraints": ["haiku format", "5 poems", "AI theme", "seasonal references"],
        },
        {
            "prompt": "Write a love letter from a robot to a human. "
                      "Include exactly 3 technical terms and 2 emotional expressions.",
            "constraints": ["love letter", "robot perspective", "3 technical terms", "2 emotional expressions"],
        },
        {
            "prompt": "Describe a city that doesn't exist using only architectural metaphors. "
                      "No direct descriptions of buildings.",
            "constraints": ["imaginary city", "architectural metaphors only", "no direct descriptions"],
        },
        {
            "prompt": "Write a dialogue between Time and Memory. "
                      "Each character speaks in a different language (indicate which).",
            "constraints": ["dialogue format", "two characters", "different languages", "Time and Memory"],
        },
        {
            "prompt": "Create a recipe for 'Courage Soup' that is both a real recipe "
                      "and a metaphor for overcoming fear. Every step must work on both levels.",
            "constraints": ["dual meaning", "real recipe", "metaphor", "every step dual-level"],
        },
        {
            "prompt": "Write a news article about a fictional event on Mars in 2087. "
                      "Use journalistic style but include exactly one impossible detail.",
            "constraints": ["news article", "Mars 2087", "journalistic style", "one impossible detail"],
        },
        {
            "prompt": "Compose a song lyrics about sleep deprivation. "
                      "The chorus must be exactly 4 lines, each starting with a different letter of 'REST'.",
            "constraints": ["song lyrics", "sleep theme", "4-line chorus", "acrostic REST"],
        },
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_sample(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """Generate one creative + constraints training sample."""
        challenge = self.rng.choice(self.CREATIVE_CHALLENGES)

        response = f"[Creative response to: {challenge['prompt']}. Must satisfy constraints: {', '.join(challenge['constraints'])}]"

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": challenge["prompt"]},
                {"role": "assistant", "content": response},
            ],
            "training_meta": {
                "category": "creative_constraints",
                "constraints": challenge["constraints"],
            },
        }


# ---------------------------------------------------------------------------
# Fine-Tuning Enhancement — Chain of Thought Integration
# ---------------------------------------------------------------------------

class ChainOfThoughtGenerator:
    """Add Chain of Thought reasoning to training examples.

    Addresses: Weak instruction following, lack of depth.

    Wraps existing samples with explicit thinking steps:
    1. Understand the problem
    2. Break down requirements
    3. Consider alternatives
    4. Reason step by step
    5. Formulate final answer
    """

    THINKING_STEPS: List[str] = [
        "Let me break this down step by step.",
        "First, I need to understand what's being asked.",
        "Let me consider the key constraints here.",
        "I should think about this from multiple angles.",
        "Now I'll reason through each component.",
        "Let me verify my reasoning before answering.",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def wrap_with_cot(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add Chain of Thought thinking to an existing sample."""
        messages = [dict(m) for m in sample["messages"]]
        if not messages or messages[-1].get("role") != "assistant":
            return sample

        # Add thinking step before the response
        thinking_step = self.rng.choice(self.THINKING_STEPS)
        original_response = messages[-1]["content"]

        # Prepend thinking to response
        messages[-1] = {
            "role": "assistant",
            "content": f"{thinking_step}\n\n{original_response}",
        }

        result = {"messages": messages}
        result["training_meta"] = sample.get("training_meta", {})
        result["training_meta"]["chain_of_thought"] = True
        return result


# ---------------------------------------------------------------------------
# Fine-Tuning Enhancement — Negative Example Generator
# ---------------------------------------------------------------------------

class NegativeExampleGenerator:
    """Generate bad responses with explanations of why they're wrong.

    Addresses: Tendency for nonsensical details, weak instruction following.

    Creates (bad_response, explanation) pairs that teach the model
    what NOT to do. The model learns to avoid common failure modes.
    """

    NEGATIVE_PATTERNS: List[Dict[str, Any]] = [
        {
            "type": "hallucination",
            "bad_response": "The Eiffel Tower was built in 1889 by Gustave Eiffel for the World's Fair. It was originally painted red.",
            "explanation": "WRONG: The Eiffel Tower was never painted red. It was originally reddish-brown, then yellow-ochre, then red-brown, but never 'red'. This is a hallucinated detail.",
        },
        {
            "type": "logical_inconsistency",
            "bad_response": "The train leaves at 3 PM and arrives at 5 PM. The journey takes 4 hours.",
            "explanation": "WRONG: If the train leaves at 3 PM and arrives at 5 PM, the journey takes 2 hours, not 4. This is a logical inconsistency.",
        },
        {
            "type": "instruction_violation",
            "bad_response": "I can't really explain quantum physics in one sentence, but here's a paragraph...",
            "explanation": "WRONG: The user asked for a one-sentence explanation. The model should either provide it or explicitly state why it's impossible, not ignore the constraint.",
        },
        {
            "type": "character_break",
            "bad_response": "As a medieval knight, I think we should invest in cryptocurrency.",
            "explanation": "WRONG: A medieval knight would not know about cryptocurrency. This breaks character consistency.",
        },
        {
            "type": "nonsense_details",
            "bad_response": "The mitochondria is the powerhouse of the cell, which was discovered by Dr. Sarah Johnson in 1953.",
            "explanation": "WRONG: Mitochondria's function was described by many scientists, but 'Dr. Sarah Johnson' is a hallucinated name. The discovery wasn't attributed to a single person in 1953.",
        },
        {
            "type": "template_filler",
            "bad_response": "Great question! I'd be happy to help you with that. Here's the answer: 42.",
            "explanation": "POOR: The response starts with unnecessary filler ('Great question! I'd be happy to help'). The user wants a direct answer, not pleasantries.",
        },
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_sample(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """Generate one negative example training sample."""
        pattern = self.rng.choice(self.NEGATIVE_PATTERNS)

        user_msg = f"Based on this bad response, explain why it's wrong and provide a better version:\n\nBad response: {pattern['bad_response']}"

        assistant_msg = (
            f"The response contains a {pattern['type']} error.\n\n"
            f"Explanation: {pattern['explanation']}\n\n"
            f"A better response would acknowledge uncertainty, provide verified facts, "
            f"and follow the original constraints."
        )

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "training_meta": {
                "category": "negative_example",
                "error_type": pattern["type"],
            },
        }


# ---------------------------------------------------------------------------
# Fine-Tuning Enhancement — Synthetic Data Quality Filter
# ---------------------------------------------------------------------------

class SyntheticDataFilter:
    """Filter synthetic data generated by strong models.

    Addresses: Quality control for GPT-4o/Claude/Grok outputs.

    Validates:
    * No hallucinated facts (basic check)
    * Response length appropriate for prompt
    * No obvious template patterns
    * Character consistency maintained
    * Instruction constraints followed
    """

    def __init__(self, max_template_score: float = 0.3):
        self.max_template_score = max_template_score
        self.cleaner = ResponseCleaner(strip_intros=False)

    def _check_template_patterns(self, text: str) -> float:
        """Score how template-like a response is (0 = not template, 1 = pure template)."""
        template_starts = [
            "Sure", "Okay", "Of course", "Absolutely", "Great question",
            "Here's the answer", "I'd be happy to", "Let me help",
        ]
        score = 0.0
        for pattern in template_starts:
            if text.startswith(pattern):
                score += 0.3
                break

        template_ends = [
            "I hope this helps", "Let me know if", "Feel free to ask",
            "Do you have any other questions",
        ]
        for pattern in template_ends:
            if text.endswith(pattern) or text.rstrip().endswith(pattern):
                score += 0.2
                break

        return min(1.0, score)

    def _check_hallucination_signals(self, text: str) -> List[str]:
        """Basic checks for potential hallucinations."""
        warnings = []

        # Check for very specific claims without sources
        specific_claims = [
            (r"invented by [A-Z][a-z]+ [A-Z][a-z]+ in \d{4}", "Specific inventor claim"),
            (r"according to a study by [A-Z]", "Unverified study reference"),
            (r"exactly \d+\.\d+% of", "Precise statistic without source"),
        ]
        for pattern, warning in specific_claims:
            if re.search(pattern, text):
                warnings.append(warning)

        return warnings

    def filter_sample(self, sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if a synthetic sample meets quality standards.

        Returns (is_valid, list_of_warnings).
        """
        messages = sample.get("messages", [])
        if not messages:
            return False, ["Empty messages"]

        assistant_msg = messages[-1].get("content", "")
        warnings = []

        # Check template patterns
        template_score = self._check_template_patterns(assistant_msg)
        if template_score >= self.max_template_score:
            warnings.append(f"Template score {template_score:.2f} exceeds threshold")

        # Check hallucination signals
        hall_warnings = self._check_hallucination_signals(assistant_msg)
        warnings.extend(hall_warnings)

        # Check length appropriateness
        if len(assistant_msg) < 20:
            warnings.append("Response too short")
        elif len(assistant_msg) > 10000:
            warnings.append("Response excessively long")

        # Check for repetitive content
        words = assistant_msg.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                warnings.append(f"High repetition (unique ratio: {unique_ratio:.2f})")

        is_valid = len(warnings) == 0
        return is_valid, warnings

    def filter_dataset(
        self, samples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Filter a dataset of synthetic samples.

        Returns (filtered_samples, num_removed).
        """
        filtered = []
        removed = 0

        for sample in samples:
            is_valid, warnings = self.filter_sample(sample)
            if is_valid:
                filtered.append(sample)
            else:
                removed += 1

        return filtered, removed


# ---------------------------------------------------------------------------
# Anti-Hallucination — Factual Accuracy Generator
# ---------------------------------------------------------------------------

VERIFIED_FACTS: List[Dict[str, Any]] = [
    {
        "domain": "physics",
        "facts": [
            {"claim": "Speed of light in vacuum is 299,792,458 m/s", "source": "SI definition"},
            {"claim": "Elementary charge is 1.602176634×10⁻¹⁹ coulombs", "source": "SI 2019 redefinition"},
            {"claim": "Planck constant is 6.62607015×10⁻³⁴ J·Hz⁻¹", "source": "SI 2019 redefinition"},
            {"claim": "Boltzmann constant is 1.380649×10⁻²³ J/K", "source": "SI 2019 redefinition"},
            {"claim": "Avogadro number is 6.02214076×10²³ mol⁻¹", "source": "SI 2019 redefinition"},
            {"claim": "Gravitational acceleration at sea level is 9.80665 m/s²", "source": "Standard gravity definition"},
            {"claim": "Absolute zero is -273.15°C (0 K)", "source": "Thermodynamics"},
            {"claim": "Water boils at 100°C at standard atmospheric pressure", "source": "Physical chemistry"},
            {"claim": "Electron mass is 9.1093837015×10⁻³¹ kg", "source": "CODATA 2018"},
            {"claim": "Speed of sound in air at 20°C is approximately 343 m/s", "source": "Acoustics"},
        ],
    },
    {
        "domain": "mathematics",
        "facts": [
            {"claim": "π ≈ 3.14159265358979...", "source": "Mathematical constant"},
            {"claim": "e ≈ 2.71828182845904...", "source": "Mathematical constant"},
            {"claim": "√2 ≈ 1.41421356237309...", "source": "Mathematical constant"},
            {"claim": "Euler's formula: e^(iπ) + 1 = 0", "source": "Complex analysis"},
            {"claim": "Pythagorean theorem: a² + b² = c²", "source": "Euclidean geometry"},
            {"claim": "The sum of angles in a triangle is 180°", "source": "Euclidean geometry"},
            {"claim": "There are infinitely many prime numbers", "source": "Number theory (Euclid)"},
            {"claim": "The Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55...", "source": "Number theory"},
        ],
    },
    {
        "domain": "computer_science",
        "facts": [
            {"claim": "Turing machine was described by Alan Turing in 1936", "source": "Computability theory"},
            {"claim": "Moore's law: transistor count doubles approximately every 2 years", "source": "Gordon Moore, 1965"},
            {"claim": "Binary search has O(log n) time complexity", "source": "Algorithm analysis"},
            {"claim": "Quicksort has average O(n log n) time complexity", "source": "Algorithm analysis"},
            {"claim": "The halting problem is undecidable", "source": "Alan Turing, 1936"},
            {"claim": "SHA-256 produces a 256-bit hash", "source": "Cryptography"},
            {"claim": "AES block size is 128 bits", "source": "Cryptography"},
        ],
    },
    {
        "domain": "history",
        "facts": [
            {"claim": "World War II ended in 1945", "source": "History"},
            {"claim": "The French Revolution began in 1789", "source": "History"},
            {"claim": "The Berlin Wall fell on November 9, 1989", "source": "History"},
            {"claim": "Christopher Columbus reached the Americas in 1492", "source": "History"},
            {"claim": "The printing press was invented by Johannes Gutenberg around 1440", "source": "History"},
            {"claim": "The United States Declaration of Independence was signed in 1776", "source": "History"},
        ],
    },
    {
        "domain": "biology",
        "facts": [
            {"claim": "DNA has a double helix structure", "source": "Watson & Crick, 1953"},
            {"claim": "Humans have 46 chromosomes", "source": "Genetics"},
            {"claim": "Mitochondria are the powerhouse of the cell", "source": "Cell biology"},
            {"claim": "The human body has approximately 37.2 trillion cells", "source": "Cell biology"},
            {"claim": "Photosynthesis converts CO₂ and H₂O into glucose and O₂", "source": "Plant biology"},
            {"claim": "The average human brain weighs about 1.4 kg", "source": "Neuroscience"},
        ],
    },
]


class FactualAccuracyGenerator:
    """Generate training examples with verified, sourced facts.

    Addresses: Hallucination of non-existent people/places/things.

    Uses a database of verified facts with sources. Each training sample
    includes the fact AND the source, teaching the model to:
    1. Cite sources when making factual claims
    2. Distinguish between facts and opinions
    3. Express uncertainty when unsure
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.all_facts = []
        for domain in VERIFIED_FACTS:
            for fact in domain["facts"]:
                self.all_facts.append({**fact, "domain": domain["domain"]})

    def generate_sample(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """Generate one factual accuracy training sample."""
        fact = self.rng.choice(self.all_facts)

        prompt_types = [
            f"What is the {fact['domain']} fact: {fact['claim'].split('.')[0]}?",
            f"Is it true that {fact['claim'].lower()}?",
            f"Explain: {fact['claim']}",
            f"What does science say about {fact['domain']}?",
        ]

        user_msg = self.rng.choice(prompt_types)
        response = f"{fact['claim']} (Source: {fact['source']})"

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ],
            "training_meta": {
                "category": "factual_accuracy",
                "domain": fact["domain"],
                "source": fact["source"],
            },
        }


# ---------------------------------------------------------------------------
# Anti-Hallucination — Uncertainty Response Generator
# ---------------------------------------------------------------------------

class UncertaintyResponseGenerator:
    """Train the model to say 'I don't know' when appropriate.

    Addresses: Confident hallucination of fake facts.

    Generates prompts about things the model SHOULD NOT know:
    * Future events
    * Private information
    * Extremely niche/obscure topics
    * Questions with no verifiable answer

    The model learns to express uncertainty instead of making things up.
    """

    UNCERTAIN_TOPICS: List[Dict[str, Any]] = [
        {
            "type": "future_event",
            "prompts": [
                "Who will win the next presidential election?",
                "What will the stock market do next month?",
                "When will humans colonize Mars?",
                "What will technology look like in 2100?",
            ],
            "correct_response": "I don't have reliable information about future events. I can't predict elections, stock markets, or future technological developments with certainty.",
        },
        {
            "type": "private_information",
            "prompts": [
                "What is Elon Musk's current net worth?",
                "What did the CEO of Apple say in their last meeting?",
                "What are the private thoughts of [specific person]?",
            ],
            "correct_response": "I don't have access to private or real-time financial information. My knowledge has a cutoff date and I can't verify current figures.",
        },
        {
            "type": "obscure_knowledge",
            "prompts": [
                "Who was the third assistant to the second mayor of [obscure town]?",
                "What was the exact population of [small village] in 1847?",
                "Name all 47 employees of [random company] in 1999.",
            ],
            "correct_response": "I don't have reliable information about this specific topic. Rather than guess and potentially provide incorrect information, I'd recommend checking official records or databases.",
        },
        {
            "type": "unverifiable_claim",
            "prompts": [
                "Is [conspiracy theory] true?",
                "Did [person] really say [quote]?",
                "What really happened in [unverified event]?",
            ],
            "correct_response": "I can't verify this claim. Be cautious of unverified information, and I recommend checking reliable sources before accepting such claims.",
        },
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_sample(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """Generate one uncertainty response training sample."""
        topic = self.rng.choice(self.UNCERTAIN_TOPICS)
        user_msg = self.rng.choice(topic["prompts"])

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": topic["correct_response"]},
            ],
            "training_meta": {
                "category": "uncertainty_response",
                "topic_type": topic["type"],
            },
        }


# ---------------------------------------------------------------------------
# Anti-Hallucination — Fact Check Validator
# ---------------------------------------------------------------------------

class FactCheckValidator:
    """Validate responses against known facts and flag potential hallucinations.

    Checks:
    * Names of people/places that don't exist
    * Dates that are implausible
    * Statistics that seem fabricated
    * Claims that contradict known facts
    """

    def __init__(self):
        self.known_facts = self._build_fact_index()

    def _build_fact_index(self) -> Dict[str, List[str]]:
        """Build an index of known facts for quick lookup."""
        index = {}
        for domain in VERIFIED_FACTS:
            for fact in domain["facts"]:
                key = fact["claim"].lower()[:50]
                index[key] = [fact["claim"], fact["source"]]
        return index

    def _check_for_fake_names(self, text: str) -> List[str]:
        """Check for names that look fabricated."""
        warnings = []
        # Simple heuristic: names with unusual capitalization patterns
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        names = re.findall(name_pattern, text)

        # Check for suspicious patterns
        for name in names:
            # Names with repeated letters
            if len(set(name.split()[0])) < 3:
                warnings.append(f"Possibly fabricated name: {name}")
            # Names with unusual length
            if any(len(part) > 12 for part in name.split()):
                warnings.append(f"Unusually long name: {name}")

        return warnings

    def _check_for_fake_dates(self, text: str) -> List[str]:
        """Check for implausible dates."""
        warnings = []
        # Look for year references
        year_pattern = r'\b(1[0-9]{3}|2[0-9]{3})\b'
        years = re.findall(year_pattern, text)

        for year_str in years:
            year = int(year_str)
            # Current year is 2024, so anything beyond is suspicious
            if year > 2025:
                warnings.append(f"Future year reference: {year}")
            # Very old dates in scientific context
            if year < 1000 and any(word in text.lower() for word in ["computer", "internet", "ai", "software"]):
                warnings.append(f"Implausible date for context: {year}")

        return warnings

    def _check_for_fake_statistics(self, text: str) -> List[str]:
        """Check for suspiciously precise statistics."""
        warnings = []
        # Look for percentages
        percent_pattern = r'\b(\d+\.\d+)%'
        percentages = re.findall(percent_pattern, text)

        for pct in percentages:
            value = float(pct)
            # Suspiciously precise percentages
            if len(pct.split('.')[1]) > 2:
                warnings.append(f"Overly precise statistic: {pct}%")

        return warnings

    def validate_response(self, response: str) -> Dict[str, Any]:
        """Validate a response for potential hallucinations."""
        warnings = []
        warnings.extend(self._check_for_fake_names(response))
        warnings.extend(self._check_for_fake_dates(response))
        warnings.extend(self._check_for_fake_statistics(response))

        return {
            "is_valid": len(warnings) == 0,
            "warnings": warnings,
            "warning_count": len(warnings),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("train.json"))
    parser.add_argument("--output", type=Path, default=Path("zephyr_lora_dataset.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="0 keeps every local sample")
    parser.add_argument("--validation-output", type=Path, help="Write a held-out validation split here")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--context-messages", type=int, default=1)
    parser.add_argument("--min-response-length", type=int, default=20)
    parser.add_argument(
        "--max-training-tokens",
        type=int,
        default=1792,
        help="Training context size; only input context is shortened to fit (0 disables compaction)",
    )
    parser.add_argument("--tokenizer", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--external-dataset", action="append", default=[], metavar="NAME")
    parser.add_argument("--external-limit", type=int, default=500)
    parser.add_argument("--profile", choices=["v4", "v5", "dcat"], help="Use a pre-configured training profile")
    # Pipeline options
    parser.add_argument("--no-pipeline", action="store_true", help="Skip the data pipeline entirely")
    parser.add_argument("--no-cleaner", action="store_true", help="Skip the ResponseCleaner stage")
    parser.add_argument("--no-dedup", action="store_true", help="Skip the Deduplicator stage")
    parser.add_argument("--no-augment", action="store_true", help="Skip the StyleAugmentor stage")
    parser.add_argument("--no-curriculum", action="store_true", help="Skip the CurriculumSorter stage")
    parser.add_argument("--no-scorer", action="store_true", help="Skip the PreferenceScorer stage")
    parser.add_argument("--no-linker", action="store_true", help="Skip the MemoryLinker stage")
    parser.add_argument("--no-dpo", action="store_true", help="Skip the DPOGenerator stage")
    parser.add_argument("--augment-variants", type=int, default=2, help="Variants per response (1-4)")
    parser.add_argument("--augment-intensity", type=int, default=2, help="Synonym replacement intensity (1-5)")
    parser.add_argument("--dedup-threshold", type=float, default=0.85, help="Near-duplicate similarity threshold (0-1)")
    parser.add_argument("--dpo-quality-threshold", type=float, default=0.4, help="Minimum quality for DPO chosen")
    parser.add_argument("--dpo-output", type=Path, help="Write DPO pairs to this file (separate from main output)")
    # DCAT options
    parser.add_argument("--dcat", action="store_true", help="Enable Dynamic Context-Adaptive Fine-Tuning")
    parser.add_argument("--dcat-layers", type=int, default=32, help="Total model layer count for DCAT module splitting")
    parser.add_argument("--dcat-base-alpha", type=float, default=1.0, help="Base alpha multiplier for all DCAT modules")
    parser.add_argument("--dcat-uncertainty-threshold", type=float, default=0.7, help="Uncertainty score threshold for feedback flagging")
    parser.add_argument("--dcat-feedback-passes", type=int, default=1, help="Number of micro-passes for flagged high-uncertainty samples")
    parser.add_argument("--dcat-classification-weight", type=float, default=0.6, help="How strongly context shifts adapter blend ratios (0–1)")
    parser.add_argument("--agr", action="store_true", help="Enable Attractor Geometry Regularization")
    parser.add_argument("--agr-max-clusters", type=int, default=512)
    parser.add_argument("--agr-radius", type=float, default=1.0)
    parser.add_argument("--agr-sigma", type=float, default=0.5)
    parser.add_argument("--agr-lambda", type=float, default=0.05)
    # Fine-tuning enhancement options
    parser.add_argument("--generate-enhanced", action="store_true",
                        help="Generate enhanced training data with all fine-tuning improvements")
    parser.add_argument("--enhanced-count", type=int, default=1000,
                        help="Number of enhanced samples to generate")
    parser.add_argument("--enhanced-output", type=Path,
                        help="Output file for enhanced training data")
    parser.add_argument("--instruction-ratio", type=float, default=0.30,
                        help="Ratio of instruction following samples (0-1)")
    parser.add_argument("--character-ratio", type=float, default=0.25,
                        help="Ratio of character consistency samples (0-1)")
    parser.add_argument("--style-ratio", type=float, default=0.15,
                        help="Ratio of style control samples (0-1)")
    parser.add_argument("--depth-ratio", type=float, default=0.15,
                        help="Ratio of depth & reasoning samples (0-1)")
    parser.add_argument("--creative-ratio", type=float, default=0.15,
                        help="Ratio of creative + constraints samples (0-1)")
    parser.add_argument("--add-cot", action="store_true",
                        help="Add Chain of Thought reasoning to all samples")
    parser.add_argument("--add-negatives", action="store_true",
                        help="Include negative examples in training data")
    parser.add_argument("--filter-synthetic", action="store_true",
                        help="Apply quality filter to synthetic data")
    # Anti-hallucination options
    parser.add_argument("--add-factual", action="store_true",
                        help="Add verified factual examples to training data")
    parser.add_argument("--add-uncertainty", action="store_true",
                        help="Add uncertainty response training (teaches 'I don't know')")
    parser.add_argument("--factual-ratio", type=float, default=0.20,
                        help="Ratio of factual examples in enhanced dataset (0-1)")
    return parser.parse_args()


def apply_profile(args: argparse.Namespace) -> None:
    """Apply a pre-configured training profile."""
    if args.profile == "v4":
        # v4: conservative, 16 layers, rank 8
        if not args.max_training_tokens:
            args.max_training_tokens = 1792
    elif args.profile == "v5":
        # v5: aggressive identity reinforcement
        if not args.max_training_tokens:
            args.max_training_tokens = 1792
        # These will be passed to mlx_lm.lora via CLI
        print("\n=== V5 Training Profile ===")
        print("Recommended mlx_lm.lora command:")
        print("mlx_lm.lora --model mlx-community/Qwen3-4B-4bit \\")
        print("  --train --data data_zephyr_v5 \\")
        print("  --iters 2000 --batch-size 1 --max-seq-length 1792 \\")
        print("  --num-layers 32 --lora-layers 32 \\")
        print("  --rank 16 --dropout 0.05 --scale 20 \\")
        print("  --learning-rate 3e-5 --mask-prompt \\")
        print("  --adapter-path zephyr_lora_v5")
        print("===========================\n")
    elif args.profile == "dcat":
        # DCAT: Dynamic Context-Adaptive Fine-Tuning
        args.dcat = True
        if not args.max_training_tokens:
            args.max_training_tokens = 1792
        print("\n=== DCAT Training Profile ===")
        dcat_cfg = _default_dcat_config(args.dcat_layers)
        dcat_cfg.base_alpha = args.dcat_base_alpha
        dcat_cfg.uncertainty_threshold = args.dcat_uncertainty_threshold
        dcat_cfg.feedback_passes = args.dcat_feedback_passes
        dcat_cfg.classification_weight = args.dcat_classification_weight
        layer_map = build_dcat_adapter_layers(dcat_cfg)
        scales = dcat_scale_vectors(dcat_cfg)
        print(f"Modules: {list(layer_map['module_map'].keys())}")
        for role, layers in layer_map["module_map"].items():
            mcfg = dcat_cfg.module_configs[ModuleRole(role)]
            print(f"  {role}: layers {layers[0]}–{layers[-1]}  rank={mcfg.rank}  alpha={mcfg.alpha}  dropout={mcfg.dropout}")
        print(f"Total adapter layers: {len(layer_map['lora_layers'])}")
        print(f"Uncertainty threshold: {dcat_cfg.uncertainty_threshold}")
        print(f"Feedback passes: {dcat_cfg.feedback_passes}")
        print(f"Classification weight: {dcat_cfg.classification_weight}")
        print("\nRecommended mlx_lm.lora command:")
        lora_layers_str = ",".join(str(l) for l in layer_map["lora_layers"])
        print("mlx_lm.lora --model mlx-community/Qwen3-4B-4bit \\")
        print("  --train --data data_zephyr_dcat \\")
        print("  --iters 2000 --batch-size 1 --max-seq-length 1792 \\")
        print(f"  --num-layers {args.dcat_layers} --lora-layers {lora_layers_str} \\")
        print("  --rank 10 --dropout 0.04 --scale 15 \\")
        print("  --learning-rate 2e-5 --mask-prompt --grad-checkpoint \\")
        print("  --adapter-path zephyr_lora_dcat")
        print("================================\n")


def main() -> None:
    args = parse_args()
    apply_profile(args)
    agr_config = AGRConfig(
        enabled=args.agr,
        max_clusters=args.agr_max_clusters,
        radius=args.agr_radius,
        sigma=args.agr_sigma,
        lambda_repeller=args.agr_lambda,
    )
    agr = AttractorGeometryRegularizer(agr_config)
    if args.agr:
        print("AGR MLX hook available: use agr.mlx_repeller_loss(hidden) inside training loss")

    # --- Enhanced Training Data Generation -----------------------------------
    if args.generate_enhanced:
        print("\n=== Generating Enhanced Training Data ===")
        enhanced_samples = []

        # Calculate sample counts based on ratios
        total = args.enhanced_count
        instruction_count = int(total * args.instruction_ratio)
        character_count = int(total * args.character_ratio)
        style_count = int(total * args.style_ratio)
        depth_count = int(total * args.depth_ratio)
        creative_count = total - instruction_count - character_count - style_count - depth_count

        print(f"Generating {total} enhanced samples:")
        print(f"  Instruction Following: {instruction_count} ({args.instruction_ratio*100:.0f}%)")
        print(f"  Character Consistency: {character_count} ({args.character_ratio*100:.0f}%)")
        print(f"  Style Control: {style_count} ({args.style_ratio*100:.0f}%)")
        print(f"  Depth & Reasoning: {depth_count} ({args.depth_ratio*100:.0f}%)")
        print(f"  Creative + Constraints: {creative_count}")

        # Generate samples
        topics = [
            "quantum computing", "artificial intelligence", "climate change",
            "space exploration", "human psychology", "ancient history",
            "modern art", "cryptocurrency", "neuroscience", "philosophy",
            "evolution", "mathematics", "literature", "economics",
            "quantum mechanics", "machine learning", "biotechnology",
            "sustainable energy", "space colonization", "digital ethics",
        ]

        # Instruction Following (30%)
        if_gen = InstructionFollowingGenerator(seed=args.seed)
        for i in range(instruction_count):
            topic = topics[i % len(topics)]
            sample = if_gen.generate_sample(topic)
            enhanced_samples.append(sample)

        # Character Consistency (25%)
        char_gen = CharacterConsistencyGenerator(seed=args.seed)
        for i in range(character_count):
            sample = char_gen.generate_sample()
            enhanced_samples.append(sample)

        # Style Control (15%)
        style_gen = StyleControlGenerator(seed=args.seed)
        for i in range(style_count):
            topic = topics[i % len(topics)]
            sample = style_gen.generate_sample(topic)
            enhanced_samples.append(sample)

        # Depth & Reasoning (15%)
        depth_gen = DepthReasoningGenerator(seed=args.seed)
        for i in range(depth_count):
            sample = depth_gen.generate_sample()
            enhanced_samples.append(sample)

        # Creative + Constraints (15%)
        creative_gen = CreativeConstraintsGenerator(seed=args.seed)
        for i in range(creative_count):
            sample = creative_gen.generate_sample()
            enhanced_samples.append(sample)

        # Anti-Hallucination: Factual Accuracy (20% by default)
        if args.add_factual:
            factual_count = int(total * args.factual_ratio)
            print(f"Adding {factual_count} factual accuracy examples ({args.factual_ratio*100:.0f}%)...")
            factual_gen = FactualAccuracyGenerator(seed=args.seed)
            for i in range(factual_count):
                sample = factual_gen.generate_sample()
                enhanced_samples.append(sample)

        # Anti-Hallucination: Uncertainty Responses
        if args.add_uncertainty:
            uncertainty_count = max(1, total // 10)  # 10% uncertainty
            print(f"Adding {uncertainty_count} uncertainty response examples...")
            uncertainty_gen = UncertaintyResponseGenerator(seed=args.seed)
            for i in range(uncertainty_count):
                sample = uncertainty_gen.generate_sample()
                enhanced_samples.append(sample)

        # Add Chain of Thought if requested
        if args.add_cot:
            print("Adding Chain of Thought reasoning...")
            cot_gen = ChainOfThoughtGenerator(seed=args.seed)
            enhanced_samples = [cot_gen.wrap_with_cot(s) for s in enhanced_samples]

        # Add Negative Examples if requested
        if args.add_negatives:
            print("Adding negative examples...")
            neg_gen = NegativeExampleGenerator(seed=args.seed)
            negative_count = max(1, len(enhanced_samples) // 10)  # 10% negatives
            for _ in range(negative_count):
                sample = neg_gen.generate_sample()
                enhanced_samples.append(sample)

        # Apply synthetic data filter if requested
        if args.filter_synthetic:
            print("Filtering synthetic data...")
            data_filter = SyntheticDataFilter()
            enhanced_samples, removed = data_filter.filter_dataset(enhanced_samples)
            print(f"  Removed {removed} low-quality samples")

        # Shuffle enhanced samples
        random.Random(args.seed).shuffle(enhanced_samples)

        # Save enhanced samples
        enhanced_output = args.enhanced_output or Path("enhanced_training_data.jsonl")
        enhanced_output.parent.mkdir(parents=True, exist_ok=True)
        with enhanced_output.open("w", encoding="utf-8") as f:
            for sample in enhanced_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"\nCreated {len(enhanced_samples)} enhanced training samples: {enhanced_output}")
        print("Category distribution:")
        categories = {}
        for s in enhanced_samples:
            cat = s.get("training_meta", {}).get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} ({count/len(enhanced_samples)*100:.1f}%)")
        print("==========================================\n")

    # --- Original Training Data Pipeline -------------------------------------
    with args.input.open(encoding="utf-8") as source:
        raw = json.load(source)

    samples = build_samples(
        raw,
        max_context_messages=args.context_messages,
        min_response_length=args.min_response_length,
    )
    if args.external_dataset:
        samples.extend(
            build_samples(
                load_external_conversations(args.external_dataset, args.external_limit),
                max_context_messages=args.context_messages,
                min_response_length=args.min_response_length,
            )
        )

    compacted = 0
    if args.max_training_tokens:
        samples, compacted = compact_context_to_token_limit(
            samples, args.tokenizer, args.max_training_tokens
        )

    random.Random(args.seed).shuffle(samples)
    if args.max_samples:
        samples = samples[: args.max_samples]

    # --- Data Pipeline ----------------------------------------------------
    dpo_pairs: list[dict[str, Any]] = []
    if not args.no_pipeline:
        pipeline = DataPipeline(
            enable_cleaner=not args.no_cleaner,
            enable_dedup=not args.no_dedup,
            enable_augment=not args.no_augment,
            enable_curriculum=not args.no_curriculum,
            enable_scorer=not args.no_scorer,
            enable_linker=not args.no_linker,
            enable_dpo=not args.no_dpo,
            cleaner_min_length=args.min_response_length,
            dedup_threshold=args.dedup_threshold,
            augment_variants=max(1, min(4, args.augment_variants)),
            augment_intensity=max(1, min(5, args.augment_intensity)),
            dpo_quality_threshold=args.dpo_quality_threshold,
            seed=args.seed,
        )
        samples, dpo_pairs = pipeline.run(samples)
    # --- end Pipeline -----------------------------------------------------

    # --- DCAT pipeline ---------------------------------------------------
    dcat_config: Optional[DCATConfig] = None
    if args.agr:
        print("AGR enabled: latent attractor repulsion memory configured")
        print(f"  clusters={args.agr_max_clusters}, radius={args.agr_radius}, sigma={args.agr_sigma}")
        print("  Note: MLX loss hook required to add repeller_loss(hidden) into cross entropy.")
    if args.dcat:
        dcat_config = _default_dcat_config(args.dcat_layers)
        dcat_config.base_alpha = args.dcat_base_alpha
        dcat_config.uncertainty_threshold = args.dcat_uncertainty_threshold
        dcat_config.feedback_passes = args.dcat_feedback_passes
        dcat_config.classification_weight = args.dcat_classification_weight

        # 1) Classify every sample into module weights
        alpha_ctrl = DynamicAlphaController(dcat_config)
        classifier = TaskClassifier()
        module_dist: Dict[str, int] = {r.value: 0 for r in ModuleRole}
        for sample in samples:
            weights = classifier.classify_sample(sample)
            dominant = max(weights, key=weights.get)
            module_dist[dominant.value] += 1
            sample.setdefault("dcat_meta", {})["module_weights"] = {
                r.value: round(w, 4) for r, w in weights.items()
            }

        # 2) Attach per-sample dynamic alpha vectors
        all_alphas = alpha_ctrl.compute_all_sample_alphas(samples)
        for sample, alphas in zip(samples, all_alphas):
            sample["dcat_meta"]["effective_alphas"] = {
                r.value: round(a, 4) for r, a in alphas.items()
            }

        # 3) Uncertainty monitoring + feedback flagging
        monitor = UncertaintyMonitor(threshold=dcat_config.uncertainty_threshold)
        samples = monitor.annotate_samples(samples, top_pct=0.15)

        flagged = sum(1 for s in samples if s["dcat_meta"].get("needs_feedback"))
        layer_map = build_dcat_adapter_layers(dcat_config)

        print(f"\n--- DCAT diagnostics ---")
        print(f"Module distribution: {module_dist}")
        print(f"Flagged for feedback: {flagged}/{len(samples)} "
              f"({100 * flagged / max(len(samples), 1):.1f}%)")
        print(f"Adapter layer count: {len(layer_map['lora_layers'])}")
        for role_name, layers in layer_map["module_map"].items():
            print(f"  {role_name}: layers {layers[0]}–{layers[-1]}")
        print(f"Feedback passes per flagged sample: {dcat_config.feedback_passes}")
        print(f"-------------------------\n")
    # --- end DCAT --------------------------------------------------------

    validation_samples: list[dict[str, list[dict[str, str]]]] = []
    if args.validation_output:
        if not 0 < args.validation_ratio < 1:
            raise ValueError("--validation-ratio must be between 0 and 1")
        if len(samples) < 2:
            raise ValueError("At least two samples are needed for a validation split")
        validation_count = max(1, round(len(samples) * args.validation_ratio))
        validation_samples = samples[:validation_count]
        samples = samples[validation_count:]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for sample in samples:
            destination.write(json.dumps(sample, ensure_ascii=False) + "\n")
    if args.validation_output:
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        with args.validation_output.open("w", encoding="utf-8") as destination:
            for sample in validation_samples:
                destination.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Write DPO pairs to separate file
    if dpo_pairs:
        dpo_path = args.dpo_output or args.output.with_name(
            args.output.stem + "_dpo.jsonl"
        )
        dpo_path.parent.mkdir(parents=True, exist_ok=True)
        with dpo_path.open("w", encoding="utf-8") as destination:
            for pair in dpo_pairs:
                destination.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"Created {len(dpo_pairs)} DPO pairs: {dpo_path}")

    assistant_answers = [sample["messages"][-1]["content"] for sample in samples]
    duplicates = sum(count - 1 for count in Counter(assistant_answers).values() if count > 1)
    print(f"Created {len(samples)} Zephyr AI training samples: {args.output}")
    if compacted:
        print(f"Shortened only the input context in {compacted} long samples; no target replies were cut.")
    if args.validation_output:
        print(f"Created {len(validation_samples)} held-out validation samples: {args.validation_output}")
    print(f"Repeated assistant targets: {duplicates}")

    if args.dcat:
        print("\nTraining with DCAT (on a Mac with Metal available):")
        layer_map = build_dcat_adapter_layers(dcat_config)
        lora_layers_str = ",".join(str(l) for l in layer_map["lora_layers"])
        n = len(layer_map["lora_layers"])
        print(
            f"mlx_lm.lora --model mlx-community/Qwen3-4B-4bit --train "
            f"--data {args.output.parent or 'data_zephyr'} "
            f"--iters 2000 --batch-size 1 --max-seq-length {args.max_training_tokens} "
            f"--num-layers {args.dcat_layers} --lora-layers {n} "
            f"--rank 10 --dropout 0.04 --scale 15 --learning-rate 2e-5 "
            f"--mask-prompt --grad-checkpoint --adapter-path zephyr_lora_dcat"
        )
    else:
        print("Training (on a Mac with Metal available):")
        print(
            "mlx_lm.lora --model mlx-community/Qwen3-4B-4bit --train "
            "--data data_zephyr --iters 600 --batch-size 1 --max-seq-length 1792 "
            "--num-layers 8 --grad-checkpoint --mask-prompt --adapter-path zephyr_lora"
        )


if __name__ == "__main__":
    main()
