"""Training and development-only evaluation for the neural pilot.

The window encoder is trained on window-level examples (each window inherits
its document's label) with one of the preregistered objectives. Development
evaluation runs the encoder over every development window, aggregates windows
to document scores, and reports overall and worst-cohort (per-domain) metrics.
Model selection and early stopping use development only; no final-test label is
ever loaded here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from bench.neural.aggregate import aggregate_documents, log_odds, sigmoid
from bench.neural.data import WindowedCorpus
from bench.neural.objectives import GroupDRO


@dataclass
class PilotConfig:
    lr: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    grad_accum: int = 2
    max_epochs: int = 2
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    seed: int = 13
    dro_step_size: float = 0.05
    eval_batch_size: int = 32
    max_windows: int = 32


@dataclass
class EncoderOutput:
    """Flat window-level forward results plus per-document slices."""

    window_logits: np.ndarray  # [n_windows, 2]
    window_embeds: np.ndarray  # [n_windows, hidden]
    doc_slices: list[tuple[int, int]]  # per-doc (start, end) into flat arrays
    doc_spans: list[list[tuple[int, int]]]
    doc_n_tokens: list[int]
    labels: np.ndarray
    domains: list[str]
    families: list[str]
    groups: list[str]

    def doc_probabilities_logit_mean(self) -> np.ndarray:
        doc_logits = [self.window_logits[s:e] for s, e in self.doc_slices]
        return aggregate_documents(doc_logits, self.doc_spans, self.doc_n_tokens)


def _batch_iter(n: int, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


@torch.no_grad()
def encode_corpus(model, corpus: WindowedCorpus, device: str, batch_size: int = 32) -> EncoderOutput:
    """Run the window encoder over every window; collect logits + embeddings."""
    model.eval()
    flat = corpus.flat()
    input_ids = flat["input_ids"]
    attention_mask = flat["attention_mask"]
    n = len(input_ids)
    logits_out: list[np.ndarray] = []
    embeds_out: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        ids = torch.from_numpy(input_ids[start : start + batch_size]).to(device)
        mask = torch.from_numpy(attention_mask[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits, cls = model(ids, mask)
        logits_out.append(logits.float().cpu().numpy())
        embeds_out.append(cls.float().cpu().numpy())
    window_logits = np.concatenate(logits_out, axis=0)
    window_embeds = np.concatenate(embeds_out, axis=0)

    doc_slices: list[tuple[int, int]] = []
    doc_spans: list[list[tuple[int, int]]] = []
    cursor = 0
    for wins in corpus.windows:
        doc_slices.append((cursor, cursor + len(wins)))
        doc_spans.append([(w.token_start, w.token_end) for w in wins])
        cursor += len(wins)
    return EncoderOutput(
        window_logits=window_logits,
        window_embeds=window_embeds,
        doc_slices=doc_slices,
        doc_spans=doc_spans,
        doc_n_tokens=list(corpus.n_tokens),
        labels=np.asarray(corpus.labels),
        domains=list(corpus.domains),
        families=list(corpus.families),
        groups=list(corpus.groups),
    )


def cohort_metrics(labels: np.ndarray, probs: np.ndarray, cohorts: list[str]) -> dict:
    """Overall + per-cohort + worst-cohort AUROC/Brier for development selection."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    cohorts_arr = np.array([str(c) for c in cohorts])
    out: dict = {"n": int(len(labels)), "prevalence": float(labels.mean())}
    try:
        out["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["auroc"] = float("nan")
    out["brier"] = float(np.mean((probs - labels) ** 2))

    per: dict[str, dict] = {}
    for c in sorted(set(cohorts_arr.tolist())):
        m = cohorts_arr == c
        y = labels[m]
        if len(set(y.tolist())) < 2:
            per[c] = {"n": int(m.sum()), "auroc": float("nan"), "brier": float(np.mean((probs[m] - y) ** 2))}
            continue
        per[c] = {
            "n": int(m.sum()),
            "auroc": float(roc_auc_score(y, probs[m])),
            "brier": float(np.mean((probs[m] - y) ** 2)),
        }
    out["per_cohort"] = per
    aurocs = [v["auroc"] for v in per.values() if not np.isnan(v["auroc"])]
    briers = [v["brier"] for v in per.values()]
    out["worst_cohort_auroc"] = float(np.min(aurocs)) if aurocs else float("nan")
    out["worst_cohort_brier"] = float(np.max(briers)) if briers else float("nan")
    out["n_cohorts"] = len(per)
    return out


def train_window_encoder(
    model: nn.Module,
    train_corpus: WindowedCorpus,
    dev_corpus: WindowedCorpus,
    objective: str,
    objective_payload,
    config: PilotConfig,
    device: str,
    log_prefix: str = "",
) -> tuple[nn.Module, dict]:
    """Train the window encoder with development-only checkpoint selection.

    Returns the model restored to its best-development state and a history dict.
    """
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    flat = train_corpus.flat()
    input_ids = flat["input_ids"]
    attention_mask = flat["attention_mask"]
    labels = flat["label"]
    group_keys = flat["group_key"]
    n = len(input_ids)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps_per_epoch = int(np.ceil(n / (config.batch_size * config.grad_accum)))
    total_steps = steps_per_epoch * config.max_epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(total_steps * config.warmup_ratio), total_steps
    )
    ce = nn.CrossEntropyLoss(reduction="none")

    balanced_w = None
    if objective == "group_balanced" and objective_payload is not None:
        balanced_w = np.asarray(objective_payload, dtype=np.float32)

    history = {"objective": objective, "epochs": [], "best": None}
    best_state = None
    best_key = None
    global_step = 0
    for epoch in range(config.max_epochs):
        model.train()
        running = 0.0
        seen = 0
        opt.zero_grad(set_to_none=True)
        for bi, batch_idx in enumerate(_batch_iter(n, config.batch_size, rng, shuffle=True)):
            ids = torch.from_numpy(input_ids[batch_idx]).to(device)
            mask = torch.from_numpy(attention_mask[batch_idx]).to(device)
            y = torch.from_numpy(labels[batch_idx]).to(device)
            gk = [group_keys[int(i)] for i in batch_idx]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits, _ = model(ids, mask)
                per = ce(logits, y)
                if objective == "group_dro" and isinstance(objective_payload, GroupDRO):
                    loss = objective_payload.loss(per, gk)
                elif balanced_w is not None:
                    w = torch.from_numpy(balanced_w[batch_idx]).to(device)
                    loss = (per * w).sum() / w.sum()
                else:
                    loss = per.mean()
                loss = loss / config.grad_accum
            loss.backward()
            running += float(per.detach().mean().cpu()) * len(batch_idx)
            seen += len(batch_idx)
            if (bi + 1) % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
        train_loss = running / max(seen, 1)

        # Development-only checkpoint selection.
        dev_out = encode_corpus(model, dev_corpus, device, config.eval_batch_size)
        dev_probs = dev_out.doc_probabilities_logit_mean()
        dev_metrics = cohort_metrics(dev_out.labels, dev_probs, dev_out.domains)
        key = (dev_metrics["worst_cohort_auroc"], dev_metrics["auroc"])
        history["epochs"].append(
            {
                "epoch": epoch + 1,
                "train_window_loss": float(train_loss),
                "dev": dev_metrics,
            }
        )
        print(
            f"{log_prefix}[epoch {epoch + 1}] train_loss {train_loss:.4f} | "
            f"dev worst-cohort AUROC {dev_metrics['worst_cohort_auroc']:.4f} | "
            f"dev AUROC {dev_metrics['auroc']:.4f} | dev Brier {dev_metrics['brier']:.4f}",
            flush=True,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["best"] = {
                "epoch": epoch + 1,
                "dev": dev_metrics,
                "selection_key": {"worst_cohort_auroc": key[0], "auroc": key[1]},
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, history


def train_summary_head(
    head: nn.Module,
    train_output: EncoderOutput,
    dev_output: EncoderOutput,
    config: PilotConfig,
    device: str,
    lr: float = 1e-3,
    max_epochs: int = 10,
    log_prefix: str = "",
) -> tuple[nn.Module, dict]:
    """Train the hierarchical summary head on frozen window embeddings."""
    from bench.neural.aggregate import sigmoid as _sig

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=config.weight_decay)
    ce = nn.CrossEntropyLoss()

    def doc_batches(output: EncoderOutput, batch_docs: int, shuffle: bool):
        n_docs = len(output.doc_slices)
        order = rng.permutation(n_docs) if shuffle else np.arange(n_docs)
        hidden = output.window_embeds.shape[1]
        for start in range(0, n_docs, batch_docs):
            sel = order[start : start + batch_docs]
            maxw = max(
                min(output.doc_slices[d][1] - output.doc_slices[d][0], config.max_windows) for d in sel
            )
            embeds = np.zeros((len(sel), maxw, hidden), dtype=np.float32)
            mask = np.zeros((len(sel), maxw), dtype=bool)
            y = np.zeros(len(sel), dtype=np.int64)
            for r, d in enumerate(sel):
                s, e = output.doc_slices[d]
                w = min(e - s, config.max_windows)
                embeds[r, :w] = output.window_embeds[s : s + w]
                mask[r, :w] = True
                y[r] = int(output.labels[d])
            yield (
                torch.from_numpy(embeds).to(device),
                torch.from_numpy(mask).to(device),
                torch.from_numpy(y).to(device),
            )

    def evaluate(output: EncoderOutput) -> dict:
        head.eval()
        probs_list: list[float] = []
        with torch.no_grad():
            for embeds, mask, _y in doc_batches(output, 64, shuffle=False):
                logits = head(embeds, mask)
                lo = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
                probs_list.extend(_sig(lo).tolist())
        probs = np.array(probs_list, dtype=np.float64)
        return cohort_metrics(output.labels, probs, output.domains)

    best_state = None
    best_key = None
    history = {"epochs": [], "best": None}
    for epoch in range(max_epochs):
        head.train()
        running = 0.0
        seen = 0
        for embeds, mask, y in doc_batches(train_output, 64, shuffle=True):
            logits = head(embeds, mask)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu()) * len(y)
            seen += len(y)
        dev_metrics = evaluate(dev_output)
        key = (dev_metrics["worst_cohort_auroc"], dev_metrics["auroc"])
        history["epochs"].append({"epoch": epoch + 1, "train_loss": running / max(seen, 1), "dev": dev_metrics})
        print(
            f"{log_prefix}[summary epoch {epoch + 1}] train_loss {running / max(seen, 1):.4f} | "
            f"dev worst-cohort AUROC {dev_metrics['worst_cohort_auroc']:.4f}",
            flush=True,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            history["best"] = {"epoch": epoch + 1, "dev": dev_metrics}
    if best_state is not None:
        head.load_state_dict(best_state)
    head.to(device)
    return head, history


@torch.no_grad()
def summary_head_probabilities(head: nn.Module, output: EncoderOutput, device: str, max_windows: int = 32, batch_docs: int = 64) -> np.ndarray:
    """Apply a trained summary head to a corpus's frozen window embeddings."""
    from bench.neural.aggregate import sigmoid as _sig

    head.eval()
    hidden = output.window_embeds.shape[1]
    n_docs = len(output.doc_slices)
    probs = np.zeros(n_docs, dtype=np.float64)
    for start in range(0, n_docs, batch_docs):
        sel = np.arange(start, min(start + batch_docs, n_docs))
        maxw = max(min(output.doc_slices[d][1] - output.doc_slices[d][0], max_windows) for d in sel)
        embeds = np.zeros((len(sel), maxw, hidden), dtype=np.float32)
        mask = np.zeros((len(sel), maxw), dtype=bool)
        for r, d in enumerate(sel):
            s, e = output.doc_slices[d]
            w = min(e - s, max_windows)
            embeds[r, :w] = output.window_embeds[s : s + w]
            mask[r, :w] = True
        logits = head(torch.from_numpy(embeds).to(device), torch.from_numpy(mask).to(device))
        lo = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
        probs[start : start + len(sel)] = _sig(lo)
    return probs


def measure_latency(model: nn.Module, corpus: WindowedCorpus, device: str, n_docs: int = 200, batch_size: int = 32) -> dict:
    """Inference latency per document and peak memory on a dev subset."""
    flat = corpus.flat()
    n_windows_total = len(flat["input_ids"])
    # take the first n_docs documents' windows
    doc_slices = []
    cursor = 0
    for d, wins in enumerate(corpus.windows):
        doc_slices.append((cursor, cursor + len(wins)))
        cursor += len(wins)
        if d + 1 >= n_docs:
            break
    n_w = doc_slices[-1][1]
    ids = flat["input_ids"][:n_w]
    mask = flat["attention_mask"][:n_w]
    model.eval()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n_w, batch_size):
            i = torch.from_numpy(ids[start : start + batch_size]).to(device)
            m = torch.from_numpy(mask[start : start + batch_size]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                model(i, m)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_docs_actual = len(doc_slices)
    out = {
        "n_docs": n_docs_actual,
        "n_windows": int(n_w),
        "total_sec": round(elapsed, 3),
        "ms_per_doc": round(1000.0 * elapsed / max(n_docs_actual, 1), 2),
        "windows_per_sec": round(n_w / max(elapsed, 1e-9), 1),
    }
    if device == "cuda":
        out["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return out
