"""Unit tests for watermark radioactivity (no model download)."""

from __future__ import annotations

import numpy as np
import pytest
from panoptes.analysis.watermarks import green_for

from bench.radioactivity import (
    fit_bigram_student,
    green_bias_strength,
    neutralize_sample,
    paraphrase_corpus,
    run_synthetic_radioactivity,
    score_texts,
    synthesize_teacher_texts,
    tokenize,
)
from bench.run_radioactivity import build_card


def test_synthesize_teacher_watermark_raises_green_rate() -> None:
    prompts = ["The researchers reported that", "A local team won after"]
    wm = synthesize_teacher_texts(prompts, delta=2.0, n_tokens=60, seed=0)
    ctrl = synthesize_teacher_texts(prompts, delta=0.0, n_tokens=60, seed=1)
    assert green_bias_strength(wm) > green_bias_strength(ctrl)
    assert score_texts(wm)["mean_green_rate"] > score_texts(ctrl)["mean_green_rate"]


def test_bigram_student_inherits_green_bias() -> None:
    prompts = [
        "The city council met on Tuesday to discuss",
        "Scientists announced a new discovery about",
        "Researchers developed a novel method for",
        "The stock market reacted sharply to news of",
    ]
    wm = synthesize_teacher_texts(prompts, delta=2.0, n_tokens=80, seed=0)
    ctrl = synthesize_teacher_texts(prompts, delta=0.0, n_tokens=80, seed=100)
    student_wm = fit_bigram_student(wm)
    student_ctrl = fit_bigram_student(ctrl)
    probe = ["Investigators examined the causes of", "The library launched a digital archive of"]
    out_wm = [student_wm.sample(p, 80, np.random.default_rng(i)) for i, p in enumerate(probe)]
    out_ctrl = [
        student_ctrl.sample(p, 80, np.random.default_rng(10 + i)) for i, p in enumerate(probe)
    ]
    assert score_texts(out_wm)["mean_z"] > score_texts(out_ctrl)["mean_z"]


def test_neutralization_reduces_green_rate() -> None:
    prompts = ["The researchers reported that", "A local team won after"]
    wm = synthesize_teacher_texts(prompts * 4, delta=2.0, n_tokens=80, seed=0)
    student = fit_bigram_student(wm)
    plain = [
        student.sample("Investigators examined the causes of", 80, np.random.default_rng(i))
        for i in range(4)
    ]
    neutralized = [
        neutralize_sample(
            "Investigators examined the causes of",
            n_tokens=80,
            student=student,
            delta=2.0,
            rng=np.random.default_rng(100 + i),
        )
        for i in range(4)
    ]
    assert score_texts(neutralized)["mean_green_rate"] < score_texts(plain)["mean_green_rate"]


def test_paraphrase_perturbs_tokens() -> None:
    texts = ["The study said the new report was important for many people."]
    out = paraphrase_corpus(texts, rate=1.0)
    assert out[0] != texts[0]
    assert tokenize(out[0])


def test_synthetic_radioactivity_card_shape() -> None:
    result = run_synthetic_radioactivity(n_tokens=50, probe_tokens=50, delta=2.0, seed=0)
    assert result["inheritance"]["student_on_watermarked"]["mean_z"] is not None
    assert result["removal"]["neutralize_post"]["mean_z"] is not None
    card = build_card(result)
    assert card["schema"] == "panoptes-radioactivity-card-v1"
    assert card["mode"] == "synthetic"
    # Inheritance should beat control
    assert (
        result["inheritance"]["student_on_watermarked"]["mean_z"]
        > result["inheritance"]["student_on_control"]["mean_z"]
    )


def test_green_for_consistency_on_tokenized_pairs() -> None:
    text = "the cat sat on the mat"
    tokens = tokenize(text)
    previous = ""
    for token in tokens:
        assert green_for(previous, token) in (True, False)
        previous = token


@pytest.mark.slow
def test_tiny_model_radioactivity_smoke() -> None:
    """Opt-in smoke test: real HF SFT on a tiny model. Needs torch/transformers
    and network for the first weight download; skipped by default (-m "not slow")."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from bench.radioactivity import run_model_radioactivity

    result = run_model_radioactivity(
        teacher_model="gpt2",
        student_model="HuggingFaceTB/SmolLM2-135M",
        delta=2.0,
        n_tokens=32,
        epochs=1,
        device="cpu",
    )
    assert result["mode"] == "model"
    assert result["inheritance"]["student_on_watermarked"]["n_tested"] >= 0
