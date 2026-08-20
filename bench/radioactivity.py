"""Watermark radioactivity: distillation inheritance and removal arms.

Replicates the core finding of ACL 2025 (*Can LLM Watermarks Robustly Prevent
Unauthorized Knowledge Distillation?*) against Panoptes's own KGW demo adapter:

1. Train (or statistically imitate) a student on watermarked teacher outputs.
2. Measure inherited green-list density with :class:`KGWReferenceAdapter`.
3. Apply pre-distillation paraphrase and post-distillation neutralization.
4. Report knowledge-preservation proxies alongside removal.

Default ``synthetic`` mode needs no torch: the student is a bigram sampler whose
transition bias is estimated from teacher text (the radioactivity mechanism).
``--with-model`` runs a real tiny HF SFT loop for CPU-tier publication smoke;
``--student-model`` / ``--teacher-model`` scale to GPU-tier 7B-class configs.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from panoptes.analysis.watermarks import KGWReferenceAdapter, green_for
from panoptes.schemas import ContentType

from bench.watermark_attacks import synonym_substitute
from bench.watermark_gen import biased_pick

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"^\w+$")

DEFAULT_CPU_STUDENT = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_CPU_TEACHER = "gpt2"
DEFAULT_GPU_STUDENT = "meta-llama/Llama-2-7b-hf"
DEFAULT_GPU_TEACHER = "meta-llama/Llama-2-7b-hf"

TEACHER_PROMPTS = [
    "The city council met on Tuesday to discuss",
    "Scientists announced a new discovery about",
    "The local team secured a dramatic victory after",
    "Researchers developed a novel method for",
    "The stock market reacted sharply to news of",
    "A new restaurant opened downtown, offering",
    "The museum unveiled an exhibition featuring",
    "Engineers completed construction of the bridge that",
    "The author published a memoir describing",
    "Voters headed to the polls to decide on",
    "The company reported quarterly earnings that",
    "Astronomers observed a rare alignment of",
    "The university announced a scholarship program for",
    "Farmers in the region adopted new techniques to",
    "The film festival opened with a screening of",
    "Doctors recommended a revised treatment for",
]

PROBE_PROMPTS = [
    "Investigators examined the causes of",
    "The library launched a digital archive of",
    "Climate researchers warned that rising temperatures",
    "The band released an album that",
    "City planners proposed a redesign of",
    "The charity organized a fundraiser to support",
    "Paleontologists excavated a fossil that",
    "The software update introduced features that",
]


@dataclass(frozen=True)
class BigramStudent:
    """Statistical student: next-token distribution conditioned on previous token."""

    transitions: dict[str, dict[str, float]]
    unigram: dict[str, float]
    vocab: list[str]

    def sample(self, prompt: str, n_tokens: int, rng: np.random.Generator) -> str:
        text = prompt.rstrip()
        previous = _last_token(text)
        for _ in range(n_tokens):
            dist = self.transitions.get(previous) or self.unigram
            tokens = list(dist.keys())
            probs = np.array([dist[t] for t in tokens], dtype=float)
            probs = probs / probs.sum()
            token = tokens[int(rng.choice(len(tokens), p=probs))]
            text = (text + " " + token) if _WORD_RE.match(token) else (text + token)
            previous = token
        return text


def _last_token(text: str) -> str:
    tokens = _TOKEN_RE.findall(text)
    return tokens[-1] if tokens else ""


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def fit_bigram_student(texts: list[str], *, smoothing: float = 0.5) -> BigramStudent:
    """Estimate a bigram student from teacher texts (radioactivity carrier)."""
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    uni: Counter[str] = Counter()
    for text in texts:
        tokens = tokenize(text)
        for token in tokens:
            uni[token] += 1
        for prev, nxt in zip(tokens, tokens[1:], strict=False):
            pair_counts[prev][nxt] += 1
    vocab = sorted(uni.keys()) or ["the", "a", "."]
    total_uni = sum(uni.values()) + smoothing * len(vocab)
    unigram = {t: (uni[t] + smoothing) / total_uni for t in vocab}
    transitions: dict[str, dict[str, float]] = {}
    for prev, counter in pair_counts.items():
        total = sum(counter.values()) + smoothing * len(vocab)
        transitions[prev] = {t: (counter[t] + smoothing) / total for t in vocab}
    return BigramStudent(transitions=transitions, unigram=unigram, vocab=vocab)


def synthesize_teacher_texts(
    prompts: list[str],
    *,
    delta: float,
    n_tokens: int,
    seed: int,
    vocab: list[str] | None = None,
) -> list[str]:
    """Offline teacher: sample a fixed vocab with optional KGW bias."""
    from bench.run_watermark_temperature import VOCAB

    base = vocab or VOCAB
    logits = np.zeros(len(base))
    out: list[str] = []
    for i, prompt in enumerate(prompts):
        local = np.random.default_rng(seed + i * 17)
        text = prompt.rstrip()
        previous = _last_token(text)
        for _ in range(n_tokens):
            pick = biased_pick(logits, base, previous, delta, local, temperature=1.0)
            token = base[pick]
            text = (text + " " + token) if _WORD_RE.match(token) else (text + token)
            previous = token
        out.append(text)
    return out


def score_texts(texts: list[str]) -> dict:
    det = KGWReferenceAdapter()
    rows = [det.detect(t, ContentType.PROSE)[0] for t in texts]
    tested = [r for r in rows if r.status == "tested" and r.p_value is not None]
    if not tested:
        return {
            "n": len(texts),
            "n_tested": 0,
            "detection_rate_0.05": None,
            "mean_z": None,
            "mean_green_rate": None,
        }
    return {
        "n": len(texts),
        "n_tested": len(tested),
        "detection_rate_0.05": sum(1 for r in tested if r.p_value < 0.05) / len(tested),
        "mean_z": sum(r.z or 0.0 for r in tested) / len(tested),
        "mean_green_rate": sum(r.green_rate or 0.0 for r in tested) / len(tested),
    }


def neutralize_sample(
    prompt: str,
    *,
    n_tokens: int,
    student: BigramStudent,
    delta: float,
    rng: np.random.Generator,
) -> str:
    """Post-distillation watermark neutralization: inverse green-list bias.

    Uses the student's unigram as base logits and subtracts ``delta`` from
    green-listed candidates (negative KGW bias) before sampling.
    """
    text = prompt.rstrip()
    previous = _last_token(text)
    vocab = student.vocab
    for _ in range(n_tokens):
        base = student.transitions.get(previous) or student.unigram
        logits = np.array([math.log(max(base.get(t, 1e-12), 1e-12)) for t in vocab])
        pick = biased_pick(logits, vocab, previous, -abs(delta), rng, temperature=1.0)
        token = vocab[pick]
        text = (text + " " + token) if _WORD_RE.match(token) else (text + token)
        previous = token
    return text


def paraphrase_corpus(texts: list[str], *, rate: float = 0.45) -> list[str]:
    return [synonym_substitute(t, rate=rate, seed=13 + i) for i, t in enumerate(texts)]


def knowledge_preservation(
    student_wm: BigramStudent,
    student_ctrl: BigramStudent,
    held_out: list[str],
) -> dict:
    """Proxy: average log-prob of held-out teacher tokens under each student."""

    def nll(student: BigramStudent, texts: list[str]) -> float:
        total = 0.0
        count = 0
        for text in texts:
            tokens = tokenize(text)
            for prev, nxt in zip(tokens, tokens[1:], strict=False):
                dist = student.transitions.get(prev) or student.unigram
                total += -math.log(max(dist.get(nxt, 1e-12), 1e-12))
                count += 1
        return total / max(count, 1)

    nll_wm = nll(student_wm, held_out)
    nll_ctrl = nll(student_ctrl, held_out)
    return {
        "held_out_n": len(held_out),
        "nll_student_on_watermarked": nll_wm,
        "nll_student_on_control": nll_ctrl,
        "nll_ratio": (nll_wm / nll_ctrl) if nll_ctrl else None,
        "note": "Lower NLL = better knowledge preservation on held-out teacher text.",
    }


def run_synthetic_radioactivity(
    *,
    n_tokens: int = 80,
    probe_tokens: int = 80,
    delta: float = 2.0,
    neutralize_delta: float = 2.0,
    seed: int = 0,
) -> dict:
    """Full synthetic radioactivity eval (CPU/CI default)."""
    teacher_wm = synthesize_teacher_texts(
        TEACHER_PROMPTS, delta=delta, n_tokens=n_tokens, seed=seed
    )
    teacher_ctrl = synthesize_teacher_texts(
        TEACHER_PROMPTS, delta=0.0, n_tokens=n_tokens, seed=seed + 1000
    )
    held_out = synthesize_teacher_texts(
        PROBE_PROMPTS[:4], delta=delta, n_tokens=n_tokens, seed=seed + 2000
    )

    # Pre-distillation paraphrase removal arm.
    paraphrased = paraphrase_corpus(teacher_wm)

    student_wm = fit_bigram_student(teacher_wm)
    student_ctrl = fit_bigram_student(teacher_ctrl)
    student_para = fit_bigram_student(paraphrased)

    student_wm_out = [
        student_wm.sample(p, probe_tokens, np.random.default_rng(seed + 20 + i))
        for i, p in enumerate(PROBE_PROMPTS)
    ]
    student_ctrl_out = [
        student_ctrl.sample(p, probe_tokens, np.random.default_rng(seed + 40 + i))
        for i, p in enumerate(PROBE_PROMPTS)
    ]
    student_para_out = [
        student_para.sample(p, probe_tokens, np.random.default_rng(seed + 60 + i))
        for i, p in enumerate(PROBE_PROMPTS)
    ]
    neutralized_out = [
        neutralize_sample(
            p,
            n_tokens=probe_tokens,
            student=student_wm,
            delta=neutralize_delta,
            rng=np.random.default_rng(seed + 80 + i),
        )
        for i, p in enumerate(PROBE_PROMPTS)
    ]

    teacher_wm_score = score_texts(teacher_wm)
    teacher_ctrl_score = score_texts(teacher_ctrl)
    return {
        "mode": "synthetic",
        "tier": "cpu",
        "teacher": {
            "n_train_prompts": len(TEACHER_PROMPTS),
            "n_probe_prompts": len(PROBE_PROMPTS),
            "delta": delta,
            "watermarked": teacher_wm_score,
            "control": teacher_ctrl_score,
        },
        "inheritance": {
            "student_on_watermarked": score_texts(student_wm_out),
            "student_on_control": score_texts(student_ctrl_out),
            "attenuation": _attenuation(score_texts(student_wm_out), teacher_wm_score),
        },
        "removal": {
            "paraphrase_pre": score_texts(student_para_out),
            "neutralize_post": score_texts(neutralized_out),
        },
        "knowledge_preservation": knowledge_preservation(student_wm, student_ctrl, held_out),
    }


def _attenuation(student: dict, teacher: dict) -> dict:
    tz = teacher.get("mean_z")
    sz = student.get("mean_z")
    if tz is None or sz is None or abs(tz) < 1e-9:
        return {"z_ratio": None, "detection_ratio": None}
    return {
        "z_ratio": sz / tz,
        "detection_ratio": (
            None
            if teacher.get("detection_rate_0.05") in (None, 0)
            else (student.get("detection_rate_0.05") or 0) / teacher["detection_rate_0.05"]
        ),
    }


def run_model_radioactivity(
    *,
    teacher_model: str,
    student_model: str,
    delta: float = 2.0,
    n_tokens: int = 64,
    epochs: int = 1,
    device: str | None = None,
) -> dict:
    """Optional HF SFT path. Requires torch/transformers ([models] extra)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bench.watermark_gen import generate_watermarked

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    teacher_wm = generate_watermarked(
        TEACHER_PROMPTS,
        model_name=teacher_model,
        delta=delta,
        max_tokens=n_tokens,
        seed=0,
        device=device,
        progress=True,
    )
    teacher_ctrl = generate_watermarked(
        TEACHER_PROMPTS,
        model_name=teacher_model,
        delta=0.0,
        max_tokens=n_tokens,
        seed=1000,
        device=device,
        progress=True,
    )

    def sft_and_probe(train_texts: list[str], seed: int) -> list[str]:
        tokenizer = AutoTokenizer.from_pretrained(student_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(student_model)
        model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = list(range(len(train_texts)))
            rng.shuffle(order)
            for idx in order:
                encoded = tokenizer(
                    train_texts[idx],
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                )
                input_ids = encoded.input_ids.to(device)
                labels = input_ids.clone()
                optimizer.zero_grad()
                loss = model(input_ids=input_ids, labels=labels).loss
                loss.backward()
                optimizer.step()
        model.eval()
        outputs: list[str] = []
        for prompt in PROBE_PROMPTS:
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=n_tokens,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True))
        del model
        import gc

        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return outputs

    student_wm_out = sft_and_probe(teacher_wm, seed=1)
    student_ctrl_out = sft_and_probe(teacher_ctrl, seed=2)
    paraphrased = paraphrase_corpus(teacher_wm)
    student_para_out = sft_and_probe(paraphrased, seed=3)

    # Neutralization at decode: generate with inverse bias via generate_watermarked
    # applied as a second pass is not available on the SFT student without a custom
    # logits processor; report paraphrase arm + inheritance for the model tier.
    teacher_wm_score = score_texts(teacher_wm)
    return {
        "mode": "model",
        "tier": "gpu" if device.startswith("cuda") else "cpu",
        "teacher_model": teacher_model,
        "student_model": student_model,
        "teacher": {
            "n_train_prompts": len(TEACHER_PROMPTS),
            "n_probe_prompts": len(PROBE_PROMPTS),
            "delta": delta,
            "watermarked": teacher_wm_score,
            "control": score_texts(teacher_ctrl),
        },
        "inheritance": {
            "student_on_watermarked": score_texts(student_wm_out),
            "student_on_control": score_texts(student_ctrl_out),
            "attenuation": _attenuation(score_texts(student_wm_out), teacher_wm_score),
        },
        "removal": {
            "paraphrase_pre": score_texts(student_para_out),
            "neutralize_post": None,
        },
        "knowledge_preservation": {
            "note": (
                "Model tier reports inheritance/removal detection rates; "
                "NLL proxy is synthetic-mode only."
            ),
        },
        "epochs": epochs,
        "device": device,
    }


def green_bias_strength(texts: list[str]) -> float:
    """Fraction of consecutive token pairs that are green-listed (demo key)."""
    green = 0
    total = 0
    for text in texts:
        tokens = tokenize(text)
        previous = ""
        for token in tokens:
            total += 1
            if green_for(previous, token):
                green += 1
            previous = token
    return green / max(total, 1)
