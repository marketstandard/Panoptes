"""Generate the paper's SVG figures from the signed artifacts.

Reads backend/artifacts/*.json and writes standalone SVGs into
research/figures/, then injects the same markup into frontend/public/paper.html
between <!-- FIG:name --> ... <!-- /FIG:name --> markers so the paper always
renders the numbers in the signed artifacts. Deterministic: same artifacts,
byte-identical figures.

Usage: python research/make_figures.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "backend" / "artifacts"
FIGURES = ROOT / "research" / "figures"
PAPER = ROOT / "frontend" / "public" / "paper.html"

INK = "#1a2332"
MUTED = "#5b6b7f"
BLUE = "#1d4ed8"
TEAL = "#0f766e"
AMBER = "#b45309"
GREEN = "#15803d"
RED = "#b91c1c"
GRID = "#d4d9e2"


def _load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def _legend(x: float, y: float, entries: list[tuple[str, str, str]], line_height: int = 15) -> str:
    """Boxed legend. entries: (kind, color, label); kind in line|dash|point|band."""
    width = max(len(label) for _, _, label in entries) * 5.6 + 34
    height = line_height * len(entries) + 8
    parts = [
        f'<rect x="{x}" y="{y}" width="{width:.0f}" height="{height}" fill="#ffffff" fill-opacity="0.92" stroke="{GRID}"/>'
    ]
    ey = y + 12
    for kind, color, label in entries:
        if kind == "line":
            parts.append(f'<line x1="{x + 6}" y1="{ey - 3}" x2="{x + 24}" y2="{ey - 3}" stroke="{color}" stroke-width="2"/>')
        elif kind == "dash":
            parts.append(f'<line x1="{x + 6}" y1="{ey - 3}" x2="{x + 24}" y2="{ey - 3}" stroke="{color}" stroke-width="1.4" stroke-dasharray="4 3"/>')
        elif kind == "point":
            parts.append(f'<circle cx="{x + 15}" cy="{ey - 3}" r="4" fill="{color}"/>')
        elif kind == "band":
            parts.append(f'<rect x="{x + 6}" y="{ey - 8}" width="18" height="9" fill="{color}" fill-opacity="0.25"/>')
        parts.append(f'<text x="{x + 30}" y="{ey}" font-size="10" fill="{INK}">{label}</text>')
        ey += line_height
    return "".join(parts)


def _axes(w: int, h: int, pad: dict, xticks: list, yticks: list, xlabel: str, ylabel: str) -> str:
    """Render axis lines, tick labels, and axis titles. xticks/yticks are
    (position_pixels, label) pairs."""
    parts = [
        f'<line x1="{pad["l"]}" y1="{h - pad["b"]}" x2="{w - pad["r"]}" y2="{h - pad["b"]}" stroke="{INK}" stroke-width="1"/>',
        f'<line x1="{pad["l"]}" y1="{pad["t"]}" x2="{pad["l"]}" y2="{h - pad["b"]}" stroke="{INK}" stroke-width="1"/>',
    ]
    for x, label in xticks:
        parts.append(f'<line x1="{x:.1f}" y1="{h - pad["b"]}" x2="{x:.1f}" y2="{h - pad["b"] + 4}" stroke="{INK}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{h - pad["b"] + 16}" text-anchor="middle" font-size="10" fill="{MUTED}">{label}</text>')
    for y, label in yticks:
        parts.append(f'<line x1="{pad["l"] - 4}" y1="{y:.1f}" x2="{pad["l"]}" y2="{y:.1f}" stroke="{INK}" stroke-width="1"/>')
        parts.append(f'<text x="{pad["l"] - 7}" y="{y + 3:.1f}" text-anchor="end" font-size="10" fill="{MUTED}">{label}</text>')
    parts.append(f'<text x="{(pad["l"] + w - pad["r"]) / 2:.1f}" y="{h - 6}" text-anchor="middle" font-size="11" fill="{INK}">{xlabel}</text>')
    parts.append(f'<text x="14" y="{(pad["t"] + h - pad["b"]) / 2:.1f}" text-anchor="middle" font-size="11" fill="{INK}" transform="rotate(-90 14 {(pad["t"] + h - pad["b"]) / 2:.1f})">{ylabel}</text>')
    return "".join(parts)


def _svg(w: int, h: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'role="img" style="max-width:100%;height:auto;font-family:Georgia,serif">{body}</svg>'
    )


def fig_reliability() -> str:
    cal = _load("baseline-calibration.json")
    bins = cal["reliability_bins"]
    w, h, pad = 460, 318, {"l": 52, "r": 18, "t": 16, "b": 58}
    x = lambda v: pad["l"] + v * (w - pad["l"] - pad["r"])
    y = lambda v: h - pad["b"] - v * (h - pad["t"] - pad["b"])
    max_n = max(b["n"] for b in bins)
    body = [_axes(w, h, pad, [(x(t), f"{t:.1f}") for t in (0, .2, .4, .6, .8, 1)], [(y(t), f"{t:.1f}") for t in (0, .2, .4, .6, .8, 1)], "mean predicted probability", "observed AI frequency")]
    body.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 4"/>')
    for b in bins:
        r = 3 + 9 * b["n"] / max_n
        body.append(f'<circle cx="{x(b["mean_predicted"]):.1f}" cy="{y(b["observed"]):.1f}" r="{r:.1f}" fill="{BLUE}" fill-opacity="0.55" stroke="{BLUE}"/>')
    body.append(_legend(pad["l"] + 8, pad["t"] + 4, [
        ("dash", MUTED, "perfect calibration"),
        ("point", BLUE, "held-out bins (area ∝ n)"),
    ]))
    body.append(f'<text x="{(pad["l"] + w - pad["r"]) / 2:.1f}" y="{h - 26}" text-anchor="middle" font-size="10.5" fill="{MUTED}">ECE {cal["metrics"]["ece"]:.3f} · Brier {cal["metrics"]["brier"]:.3f} · held-out folds, n={cal["corpus"]["n_records"]}</text>')
    return _svg(w, h, "".join(body))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def fig_power() -> str:
    cal = _load("baseline-calibration.json")
    n_now = cal["corpus"]["n_records"]
    try:
        n_defactify = _load("defactify-summary.json")["n_records"]
    except FileNotFoundError:
        n_defactify = None
    w, h, pad = 460, 300, {"l": 52, "r": 18, "t": 16, "b": 44}
    n_min, n_max = 20, 75000
    lo, hi = math.log10(n_min), math.log10(n_max)
    x = lambda n: pad["l"] + (math.log10(max(n, n_min)) - lo) / (hi - lo) * (w - pad["l"] - pad["r"])
    y = lambda p: h - pad["b"] - p * (h - pad["t"] - pad["b"])
    power = lambda n: _normal_cdf(0.05 * math.sqrt(n) - 1.959964)
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x(10 ** (lo + i / 159 * (hi - lo))):.1f},{y(power(10 ** (lo + i / 159 * (hi - lo)))):.1f}"
        for i in range(160)
    )
    ticks = [100, 1000, 3140, 10000, 75000]
    body = [_axes(w, h, pad, [(x(t), f"{t:,}") for t in ticks], [(y(t), f"{t:.1f}") for t in (0, .2, .4, .6, .8, 1)], "eligible corpus size n (log scale)", "power")]

    body.append(f'<line x1="{pad["l"]}" y1="{y(0.8):.1f}" x2="{w - pad["r"]}" y2="{y(0.8):.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 4"/>')
    body.append(f'<path d="{path}" stroke="{BLUE}" stroke-width="2" fill="none"/>')
    body.append(f'<circle cx="{x(n_now):.1f}" cy="{y(power(n_now)):.1f}" r="4.5" fill="{AMBER}"/>')
    body.append(f'<text x="{x(n_now) + 8:.1f}" y="{y(power(n_now)) + 14:.1f}" font-size="10" fill="{AMBER}">corpus: n={n_now}, power {power(n_now) * 100:.1f}%</text>')
    legend = [
        ("line", BLUE, "power to detect a 5-pt lift (α = 0.05)"),
        ("dash", MUTED, "80% power target (n ≈ 3,140)"),
        ("point", AMBER, f"project corpus (n = {n_now})"),
    ]
    if n_defactify:
        body.append(f'<circle cx="{x(n_defactify):.1f}" cy="{y(power(n_defactify)):.1f}" r="4.5" fill="{GREEN}"/>')
        body.append(f'<text x="{x(n_defactify) - 8:.1f}" y="{y(power(n_defactify)) + 14:.1f}" text-anchor="end" font-size="10" fill="{GREEN}">Defactify: n={n_defactify:,}</text>')
        legend.append(("point", GREEN, f"Defactify bench (n = {n_defactify:,})"))
    body.append(_legend(pad["l"] + 8, pad["t"] + 4, legend))
    return _svg(w, h, "".join(body))


def fig_training() -> str:
    card = _load("panoptes-v0-card.json")
    curve = card["training_curve_seed13"]
    w, h, pad = 460, 300, {"l": 52, "r": 52, "t": 16, "b": 44}
    epochs = [r["epoch"] for r in curve]
    losses = [r["train_loss"] for r in curve]
    eces = [r["val_ece"] for r in curve]
    e_max = max(epochs)
    loss_max = max(losses) * 1.08
    ece_max = max(eces) * 1.15
    x = lambda e: pad["l"] + e / e_max * (w - pad["l"] - pad["r"])
    yl = lambda v: h - pad["b"] - v / loss_max * (h - pad["t"] - pad["b"])
    yr = lambda v: h - pad["b"] - v / ece_max * (h - pad["t"] - pad["b"])
    loss_path = " ".join(f"{'M' if i == 0 else 'L'}{x(e):.1f},{yl(v):.1f}" for i, (e, v) in enumerate(zip(epochs, losses)))
    ece_path = " ".join(f"{'M' if i == 0 else 'L'}{x(e):.1f},{yr(v):.1f}" for i, (e, v) in enumerate(zip(epochs, eces)))
    step = max(1, round(e_max / 5))
    body = [_axes(w, h, pad, [(x(t), str(t)) for t in range(0, e_max + 1, step)], [(yl(t), f"{t:.2f}") for t in [0, loss_max / 2, loss_max / 1.08 * 0.9]], "epoch (seed 13)", "train loss (evidential)")]
    # right axis: validation ECE
    body.append(f'<line x1="{w - pad["r"]}" y1="{pad["t"]}" x2="{w - pad["r"]}" y2="{h - pad["b"]}" stroke="{INK}" stroke-width="1"/>')
    for t in (0, ece_max / 2, ece_max / 1.15 * 0.9):
        body.append(f'<line x1="{w - pad["r"]}" y1="{yr(t):.1f}" x2="{w - pad["r"] + 4}" y2="{yr(t):.1f}" stroke="{INK}" stroke-width="1"/>')
        body.append(f'<text x="{w - pad["r"] + 7}" y="{yr(t) + 3:.1f}" font-size="10" fill="{TEAL}">{t:.2f}</text>')
    body.append(f'<text x="{w - 12}" y="{(pad["t"] + h - pad["b"]) / 2:.1f}" text-anchor="middle" font-size="11" fill="{TEAL}" transform="rotate(90 {w - 12} {(pad["t"] + h - pad["b"]) / 2:.1f})">validation ECE</text>')
    body.append(f'<path d="{loss_path}" stroke="{BLUE}" stroke-width="2" fill="none"/>')
    body.append(f'<path d="{ece_path}" stroke="{TEAL}" stroke-width="2" fill="none"/>')
    body.append(_legend(pad["l"] + 8, pad["t"] + 4, [
        ("line", BLUE, "train loss (evidential, left)"),
        ("line", TEAL, "grouped-validation ECE (right)"),
    ]))
    return _svg(w, h, "".join(body))


def fig_coverage() -> str:
    card = _load("cards/logistic-tier0.json")
    curve = card["evaluation"]["coverage_curve"]
    w, h, pad = 460, 300, {"l": 52, "r": 18, "t": 16, "b": 44}
    x = lambda c: pad["l"] + c * (w - pad["l"] - pad["r"])
    y = lambda a: h - pad["b"] - a * (h - pad["t"] - pad["b"])
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(r['coverage']):.1f},{y(r['accuracy']):.1f}" for i, r in enumerate(curve))
    body = [_axes(w, h, pad, [(x(t), f"{t:.1f}") for t in (0, .2, .4, .6, .8, 1)], [(y(t), f"{t:.1f}") for t in (0, .2, .4, .6, .8, 1)], "coverage (fraction kept)", "accuracy on kept cases")]
    body.append(f'<path d="{path}" stroke="{BLUE}" stroke-width="2" fill="none"/>')
    for r in curve:
        body.append(f'<circle cx="{x(r["coverage"]):.1f}" cy="{y(r["accuracy"]):.1f}" r="3" fill="{BLUE}"/>')
    first, last = curve[0], curve[-1]
    body.append(f'<text x="{x(last["coverage"]):.1f}" y="{y(last["accuracy"]) - 8:.1f}" text-anchor="end" font-size="10" fill="{MUTED}">strictest threshold: coverage {last["coverage"]:.2f}, accuracy {last["accuracy"]:.3f}</text>')
    body.append(f'<text x="{x(first["coverage"]):.1f}" y="{y(first["accuracy"]) + 14:.1f}" text-anchor="start" font-size="10" fill="{MUTED}">no abstention: accuracy {first["accuracy"]:.3f}</text>')
    body.append(_legend(pad["l"] + 8, pad["t"] + 4, [
        ("line", BLUE, "accuracy as low-confidence cases are abstained"),
    ]))
    return _svg(w, h, "".join(body))


def fig_corpus() -> str:
    cs = _load("corpus-summary.json")
    cohorts = [c for c in cs["cohorts"] if c["kind"] == "text"]
    cohorts.sort(key=lambda c: (c["family"] != "human", c["family"]))
    metrics = [("long_words", "long-word rate"), ("unique_ratio", "unique-token ratio")]
    w, h, pad = 460, 300, {"l": 52, "r": 18, "t": 34, "b": 64}
    n_groups = len(cohorts)
    group_w = (w - pad["l"] - pad["r"]) / n_groups
    bar_w = group_w / (len(metrics) + 1)
    v_max = max(c["features"][m]["mean"] for c in cohorts for m, _ in metrics) * 1.15
    y = lambda v: h - pad["b"] - v / v_max * (h - pad["t"] - pad["b"])
    body = [_axes(w, h, pad, [], [(y(t), f"{t:.2f}") for t in (0, .2, .4, .6)], "", "cohort mean feature rate")]
    colors = {metrics[0][0]: BLUE, metrics[1][0]: TEAL}
    for gi, c in enumerate(cohorts):
        gx = pad["l"] + gi * group_w
        for mi, (m, _) in enumerate(metrics):
            v = c["features"][m]["mean"]
            bx = gx + bar_w * (mi + 0.5)
            body.append(f'<rect x="{bx:.1f}" y="{y(v):.1f}" width="{bar_w * 0.82:.1f}" height="{h - pad["b"] - y(v):.1f}" fill="{colors[m]}" fill-opacity="0.8"/>')
        label = c["family"].replace("-max", "").replace("-extra-high", "")
        body.append(f'<text x="{gx + group_w / 2:.1f}" y="{h - pad["b"] + 12}" text-anchor="end" font-size="9" fill="{MUTED}" transform="rotate(-28 {gx + group_w / 2:.1f} {h - pad["b"] + 12})">{label}</text>')
    lx = pad["l"] + 6
    for m, label in metrics:
        body.append(f'<rect x="{lx}" y="10" width="9" height="9" fill="{colors[m]}" fill-opacity="0.8"/>')
        body.append(f'<text x="{lx + 13}" y="18" font-size="10" fill="{INK}">{label}</text>')
        lx += 130
    return _svg(w, h, "".join(body))


def fig_posterior() -> str:
    w, h, pad = 460, 300, {"l": 52, "r": 18, "t": 16, "b": 44}
    x = lambda lo: pad["l"] + (lo + 2) / 4 * (w - pad["l"] - pad["r"])
    y = lambda p: h - pad["b"] - p * (h - pad["t"] - pad["b"])
    body = [_axes(w, h, pad, [(x(t), f"{10 ** t:g}") for t in (-2, -1, 0, 1, 2)], [(y(t), f"{t:.1f}") for t in (0, .2, .4, .6, .8, 1)], "prior odds (log scale)", "posterior probability")]
    curves = ((0.5, MUTED), (1.0, "#9aa7b8"), (2.0, TEAL), (5.0, BLUE))
    for lr, color in curves:
        pts = []
        for i in range(121):
            lo = -2 + i / 120 * 4
            odds = 10 ** lo * lr
            pts.append(f"{'M' if i == 0 else 'L'}{x(lo):.1f},{y(odds / (1 + odds)):.1f}")
        dash = ' stroke-dasharray="4 4"' if lr == 1.0 else ""
        body.append(f'<path d="{" ".join(pts)}" stroke="{color}" stroke-width="2" fill="none"{dash}/>')
    body.append(_legend(pad["l"] + 8, pad["t"] + 4, [
        ("line", BLUE, "LR = 5 (strong evidence)"),
        ("line", TEAL, "LR = 2"),
        ("dash", "#9aa7b8", "LR = 1 (no evidence)"),
        ("line", MUTED, "LR = 0.5 (evidence for human)"),
    ]))
    return _svg(w, h, "".join(body))


def fig_defactify() -> str:
    """Per-model AUROC and ECE on the Defactify grouped holdout (OOF), with
    95% CI whiskers; chance reference annotated with the Roy et al. baselines."""
    models = [
        ("shipped heuristic (corpus-cal.)", "cards/defactify-external-validation.json", MUTED),
        ("logistic tier-0", "cards/logistic-tier0-defactify.json", BLUE),
        ("GBM tier-1", "cards/gbm-tier1-defactify.json", TEAL),
        ("Panoptes-v0 (seq. branch)", "panoptes-v0-card.json", AMBER),
    ]
    rows = []
    for label, path, color in models:
        card = _load(path)
        if "evaluation" in card:
            metrics, ci = card["evaluation"]["metrics"], card["evaluation"]["auroc_ci95"]
        else:
            m = card["metrics"]
            metrics, ci = m, [m["auroc_ci95_lo"], m["auroc_ci95_hi"]]
        rows.append({"label": label, "auroc": metrics["auroc"], "ece": metrics["ece"], "ci": ci, "color": color})

    w, h, pad = 460, 300, {"l": 150, "r": 18, "t": 30, "b": 44}
    n = len(rows)
    row_h = (h - pad["t"] - pad["b"]) / n
    x = lambda v: pad["l"] + (v - 0.4) / 0.65 * (w - pad["l"] - pad["r"])
    body = [_axes(w, h, pad, [(x(t), f"{t:.2f}") for t in (0.4, 0.6, 0.8, 1.0)], [], "AUROC (out-of-fold, 95% CI)", "")]
    body.append(f'<line x1="{x(0.5):.1f}" y1="{pad["t"]}" x2="{x(0.5):.1f}" y2="{h - pad["b"]}" stroke="{RED}" stroke-width="1.2" stroke-dasharray="5 4"/>')
    body.append(f'<text x="{x(0.5) + 4:.1f}" y="{pad["t"] + 10}" font-size="9.5" fill="{RED}">chance — Roy et al. 2026 detection baselines score 53–58% accuracy</text>')
    for i, row in enumerate(rows):
        cy = pad["t"] + row_h * (i + 0.5)
        body.append(f'<text x="{pad["l"] - 8}" y="{cy + 3:.1f}" text-anchor="end" font-size="10" fill="{INK}">{row["label"]}</text>')
        body.append(f'<rect x="{x(0.4):.1f}" y="{cy - row_h * 0.26:.1f}" width="{x(row["auroc"]) - x(0.4):.1f}" height="{row_h * 0.52:.1f}" fill="{row["color"]}" fill-opacity="0.75"/>')
        lo, hi = row["ci"]
        body.append(f'<line x1="{x(lo):.1f}" y1="{cy:.1f}" x2="{x(hi):.1f}" y2="{cy:.1f}" stroke="{INK}" stroke-width="1.4"/>')
        for cap in (lo, hi):
            body.append(f'<line x1="{x(cap):.1f}" y1="{cy - 5:.1f}" x2="{x(cap):.1f}" y2="{cy + 5:.1f}" stroke="{INK}" stroke-width="1.4"/>')
        body.append(f'<text x="{x(row["auroc"]) + 6:.1f}" y="{cy - 6:.1f}" font-size="9.5" fill="{MUTED}">AUROC {row["auroc"]:.3f} · ECE {row["ece"]:.3f}</text>')
    return _svg(w, h, "".join(body))


def fig_attribution() -> str:
    """Per-family F1 for the exploratory 7-class attribution experiment, with
    the Roy et al. 5–9% attribution-accuracy band for reference."""
    card = _load("cards/attribution-defactify.json")
    attribution = card["attribution"]
    best = attribution["contenders"][attribution["best_by_macro_f1"]]
    per_family = best["per_family_f1"]
    ref = attribution["external_reference"]
    families = list(per_family)
    w, h, pad = 460, 300, {"l": 52, "r": 18, "t": 30, "b": 64}
    n = len(families)
    group_w = (w - pad["l"] - pad["r"]) / n
    v_max = max(max(per_family.values()), ref["high"]) * 1.18
    y = lambda v: h - pad["b"] - v / v_max * (h - pad["t"] - pad["b"])
    body = [_axes(w, h, pad, [], [(y(t), f"{t:.2f}") for t in (0, 0.2, 0.4, 0.6)], "", "per-family F1 (out-of-fold)")]
    body.append(f'<rect x="{pad["l"]}" y="{y(ref["high"]):.1f}" width="{w - pad["l"] - pad["r"]}" height="{y(ref["low"]) - y(ref["high"]):.1f}" fill="{RED}" fill-opacity="0.14"/>')
    for i, family in enumerate(families):
        v = per_family[family]
        bx = pad["l"] + i * group_w + group_w * 0.18
        body.append(f'<rect x="{bx:.1f}" y="{y(v):.1f}" width="{group_w * 0.64:.1f}" height="{h - pad["b"] - y(v):.1f}" fill="{BLUE}" fill-opacity="0.8"/>')
        body.append(f'<text x="{bx + group_w * 0.32:.1f}" y="{y(v) - 5:.1f}" text-anchor="middle" font-size="9.5" fill="{INK}">{v:.2f}</text>')
        body.append(f'<text x="{pad["l"] + i * group_w + group_w / 2:.1f}" y="{h - pad["b"] + 12}" text-anchor="end" font-size="9" fill="{MUTED}" transform="rotate(-28 {pad["l"] + i * group_w + group_w / 2:.1f} {h - pad["b"] + 12})">{family}</text>')
    body.append(_legend(pad["l"] + 8, pad["t"] + 2, [
        ("point", BLUE, f'{attribution["best_by_macro_f1"]} (macro-F1 {best["macro_f1"]:.3f})'),
        ("band", RED, f'Roy et al. 2026 attribution accuracy {int(ref["low"] * 100)}–{int(ref["high"] * 100)}%'),
    ]))
    return _svg(w, h, "".join(body))


FIGS = {
    "reliability": fig_reliability,
    "power": fig_power,
    "training": fig_training,
    "coverage": fig_coverage,
    "corpus": fig_corpus,
    "posterior": fig_posterior,
    "defactify": fig_defactify,
    "attribution": fig_attribution,
}


def inject(paper: str, name: str, svg: str) -> str:
    pattern = re.compile(rf"(<!-- FIG:{name} -->).*?(<!-- /FIG:{name} -->)", re.DOTALL)
    if not pattern.search(paper):
        raise SystemExit(f"paper.html is missing the FIG:{name} marker pair")
    return pattern.sub(lambda m: m.group(1) + "\n" + svg + "\n" + m.group(2), paper)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paper = PAPER.read_text(encoding="utf-8")
    for name, build in FIGS.items():
        svg = build()
        (FIGURES / f"fig-{name}.svg").write_text(svg, encoding="utf-8", newline="\n")
        paper = inject(paper, name, svg)
        print(f"fig-{name}.svg ({len(svg)} bytes)")
    PAPER.write_text(paper, encoding="utf-8", newline="\n")
    print("paper.html figures injected")


if __name__ == "__main__":
    sys.exit(main())
