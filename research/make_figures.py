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
GRID = "#d4d9e2"


def _load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


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
    w, h, pad = 460, 300, {"l": 52, "r": 18, "t": 16, "b": 44}
    x = lambda v: pad["l"] + v * (w - pad["l"] - pad["r"])
    y = lambda v: h - pad["b"] - v * (h - pad["t"] - pad["b"])
    max_n = max(b["n"] for b in bins)
    body = [_axes(w, h, pad, [(x(t), f"{t:.1f}") for t in (0, .2, .4, .6, .8, 1)], [(y(t), f"{t:.1f}") for t in (0, .2, .4, .6, .8, 1)], "mean predicted probability", "observed AI frequency")]
    body.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 4"/>')
    for b in bins:
        r = 3 + 9 * b["n"] / max_n
        body.append(f'<circle cx="{x(b["mean_predicted"]):.1f}" cy="{y(b["observed"]):.1f}" r="{r:.1f}" fill="{BLUE}" fill-opacity="0.55" stroke="{BLUE}"/>')
    body.append(f'<text x="{x(0.03):.0f}" y="{y(0.97):.0f}" font-size="10.5" fill="{MUTED}">ECE {cal["metrics"]["ece"]:.3f} · Brier {cal["metrics"]["brier"]:.3f} · held-out folds, n={cal["corpus"]["n_records"]}</text>')
    return _svg(w, h, "".join(body))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def fig_power() -> str:
    cal = _load("baseline-calibration.json")
    n_now = cal["corpus"]["n_records"]
    w, h, pad = 460, 300, {"l": 52, "r": 18, "t": 16, "b": 44}
    n_max = 5000
    x = lambda n: pad["l"] + (n - 20) / (n_max - 20) * (w - pad["l"] - pad["r"])
    y = lambda p: h - pad["b"] - p * (h - pad["t"] - pad["b"])
    power = lambda n: _normal_cdf(0.05 * math.sqrt(n) - 1.959964)
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x(20 + i / 119 * (n_max - 20)):.1f},{y(power(20 + i / 119 * (n_max - 20))):.1f}"
        for i in range(120)
    )
    body = [_axes(w, h, pad, [(x(t), str(t)) for t in (500, 1500, 2500, 3500, 4500)], [(y(t), f"{t:.1f}") for t in (0, .2, .4, .6, .8, 1)], "eligible corpus size n", "power")]

    body.append(f'<line x1="{pad["l"]}" y1="{y(0.8):.1f}" x2="{w - pad["r"]}" y2="{y(0.8):.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 4"/>')
    body.append(f'<text x="{w - pad["r"] - 4}" y="{y(0.8) - 5:.1f}" text-anchor="end" font-size="10" fill="{MUTED}">80% power (n ≈ 3,140)</text>')
    body.append(f'<path d="{path}" stroke="{BLUE}" stroke-width="2" fill="none"/>')
    body.append(f'<circle cx="{x(n_now):.1f}" cy="{y(power(n_now)):.1f}" r="4.5" fill="{AMBER}"/>')
    body.append(f'<text x="{x(n_now) + 8:.1f}" y="{y(power(n_now)) + 14:.1f}" font-size="10" fill="{AMBER}">today: n={n_now}, power {power(n_now) * 100:.1f}%</text>')
    return _svg(w, h, "".join(body))


def fig_training() -> str:
    card = _load("panoptes-v0-card.json")
    curve = card["training_curve_seed13"]
    w, h, pad = 460, 300, {"l": 52, "r": 18, "t": 16, "b": 44}
    epochs = [r["epoch"] for r in curve]
    losses = [r["train_loss"] for r in curve]
    eces = [r["val_ece"] for r in curve]
    e_max, v_max = max(epochs), max(max(losses), max(eces))
    x = lambda e: pad["l"] + e / e_max * (w - pad["l"] - pad["r"])
    y = lambda v: h - pad["b"] - v / v_max * (h - pad["t"] - pad["b"])
    loss_path = " ".join(f"{'M' if i == 0 else 'L'}{x(e):.1f},{y(v):.1f}" for i, (e, v) in enumerate(zip(epochs, losses)))
    ece_path = " ".join(f"{'M' if i == 0 else 'L'}{x(e):.1f},{y(v):.1f}" for i, (e, v) in enumerate(zip(epochs, eces)))
    body = [_axes(w, h, pad, [(x(t), str(t)) for t in range(0, e_max + 1, 20)], [(y(t), f"{t:.2f}") for t in (0, .2, .4, .6)], "epoch (seed 13)", "loss / ECE")]
    body.append(f'<path d="{loss_path}" stroke="{BLUE}" stroke-width="2" fill="none"/>')
    body.append(f'<path d="{ece_path}" stroke="{TEAL}" stroke-width="2" fill="none"/>')
    body.append(f'<text x="{x(e_max * 0.55):.0f}" y="{y(v_max * 0.9):.0f}" font-size="10.5" fill="{BLUE}">train loss (evidential)</text>')
    body.append(f'<text x="{x(e_max * 0.55):.0f}" y="{y(v_max * 0.82):.0f}" font-size="10.5" fill="{TEAL}">grouped-validation ECE</text>')
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
    return _svg(w, h, "".join(body))


def fig_corpus() -> str:
    cs = _load("corpus-summary.json")
    cohorts = [c for c in cs["cohorts"] if c["kind"] == "text"]
    cohorts.sort(key=lambda c: (c["family"] != "human", c["family"]))
    metrics = [("long_words", "long-word rate"), ("unique_ratio", "unique-token ratio")]
    w, h, pad = 460, 300, {"l": 52, "r": 18, "t": 16, "b": 64}
    n_groups = len(cohorts)
    group_w = (w - pad["l"] - pad["r"]) / n_groups
    bar_w = group_w / (len(metrics) + 1)
    v_max = max(c["features"][m]["mean"] for c in cohorts for m, _ in metrics) * 1.15
    y = lambda v: h - pad["b"] - v / v_max * (h - pad["t"] - pad["b"])
    body = [_axes(w, h, pad, [], [(y(t), f"{t:.2f}") for t in (0, .2, .4, .6)], "", "mean (text cohorts)")]
    colors = {metrics[0][0]: BLUE, metrics[1][0]: TEAL}
    for gi, c in enumerate(cohorts):
        gx = pad["l"] + gi * group_w
        for mi, (m, _) in enumerate(metrics):
            v = c["features"][m]["mean"]
            bx = gx + bar_w * (mi + 0.5)
            body.append(f'<rect x="{bx:.1f}" y="{y(v):.1f}" width="{bar_w * 0.82:.1f}" height="{h - pad["b"] - y(v):.1f}" fill="{colors[m]}" fill-opacity="0.8"/>')
        label = c["family"].replace("-max", "").replace("-extra-high", "")
        anchor = "end" if c["family"] != "human" else "start"
        body.append(f'<text x="{gx + group_w / 2:.1f}" y="{h - pad["b"] + 12}" text-anchor="{anchor}" font-size="9" fill="{MUTED}" transform="rotate(-28 {gx + group_w / 2:.1f} {h - pad["b"] + 12})">{label}</text>')
    lx = pad["l"] + 6
    for m, label in metrics:
        body.append(f'<rect x="{lx}" y="{pad["t"]}" width="9" height="9" fill="{colors[m]}" fill-opacity="0.8"/>')
        body.append(f'<text x="{lx + 13}" y="{pad["t"] + 8}" font-size="10" fill="{INK}">{label}</text>')
        lx += 130
    return _svg(w, h, "".join(body))


def fig_posterior() -> str:
    w, h, pad = 460, 300, {"l": 52, "r": 18, "t": 16, "b": 44}
    x = lambda lo: pad["l"] + (lo + 2) / 4 * (w - pad["l"] - pad["r"])
    y = lambda p: h - pad["b"] - p * (h - pad["t"] - pad["b"])
    body = [_axes(w, h, pad, [(x(t), f"{10 ** t:g}") for t in (-2, -1, 0, 1, 2)], [(y(t), f"{t:.1f}") for t in (0, .2, .4, .6, .8, 1)], "prior odds (log scale)", "posterior probability")]
    for lr, color in ((0.5, MUTED), (1.0, "#9aa7b8"), (2.0, TEAL), (5.0, BLUE)):
        pts = []
        for i in range(121):
            lo = -2 + i / 120 * 4
            odds = 10 ** lo * lr
            pts.append(f"{'M' if i == 0 else 'L'}{x(lo):.1f},{y(odds / (1 + odds)):.1f}")
        dash = ' stroke-dasharray="4 4"' if lr == 1.0 else ""
        body.append(f'<path d="{" ".join(pts)}" stroke="{color}" stroke-width="2" fill="none"{dash}/>')
        body.append(f'<text x="{x(1.62):.1f}" y="{y((10 ** 1.62 * lr) / (1 + 10 ** 1.62 * lr)) - 6:.1f}" font-size="10" fill="{color}">LR={lr:g}</text>')
    return _svg(w, h, "".join(body))


FIGS = {
    "reliability": fig_reliability,
    "power": fig_power,
    "training": fig_training,
    "coverage": fig_coverage,
    "corpus": fig_corpus,
    "posterior": fig_posterior,
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
