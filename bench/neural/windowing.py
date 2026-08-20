"""Tokenizer-offset document windowing with overlap bookkeeping.

Long documents exceed a transformer's context window, so each document is
split into overlapping token windows. Every window records the half-open token
span ``[token_start, token_end)`` it covers in the document's full token
sequence and the matching character span ``[char_start, char_end)``. Those
spans let the aggregation layer correct for overlap (so a token covered by two
windows is not double-counted) and let the runtime map a document score back to
character offsets.

Windowing is a pure function of the text and tokenizer; it never touches labels,
which keeps the data firewall intact.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    """One token window of a document.

    ``input_ids``/``attention_mask`` include the model's special tokens and are
    padded/truncated to ``max_length``. ``token_start``/``token_end`` index the
    document's special-token-free token sequence; ``char_start``/``char_end``
    index the raw text.
    """

    input_ids: list[int]
    attention_mask: list[int]
    token_start: int
    token_end: int
    char_start: int
    char_end: int


def document_windows(
    text: str,
    tokenizer,
    max_length: int,
    overlap: int,
    max_windows: int = 32,
) -> list[Window]:
    """Split ``text`` into overlapping token windows.

    ``max_length`` is the total per-window token budget including special
    tokens; ``overlap`` is the number of content tokens shared between
    consecutive windows. The stride is ``(max_length - n_special) - overlap``.
    Windows stop at the end of the document and are capped at ``max_windows``
    (excess tail is truncated and recorded by the caller via coverage).
    """
    if max_length < 8:
        raise ValueError("max_length must leave room for content tokens")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(map(int, o)) for o in enc["offset_mapping"]]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    # ModernBERT uses cls/sep; fall back to bos/eos if a model differs.
    if cls_id is None:
        cls_id = tokenizer.bos_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id
    n_special = (1 if cls_id is not None else 0) + (1 if sep_id is not None else 0)
    content_budget = max_length - n_special
    if content_budget <= 0:
        raise ValueError("max_length too small for special tokens")
    if overlap >= content_budget:
        raise ValueError("overlap must be smaller than the content budget")
    stride = content_budget - overlap

    n = len(ids)
    if n == 0:
        # Degenerate (whitespace-only) text: emit a single empty-content window
        # so downstream batching never sees a zero-window document.
        input_ids = ([cls_id] if cls_id is not None else []) + (
            [sep_id] if sep_id is not None else []
        )
        return [
            Window(
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids),
                token_start=0,
                token_end=0,
                char_start=0,
                char_end=len(text),
            )
        ]

    windows: list[Window] = []
    start = 0
    while start < n and len(windows) < max_windows:
        end = min(start + content_budget, n)
        chunk = ids[start:end]
        input_ids = (
            ([cls_id] if cls_id is not None else [])
            + chunk
            + ([sep_id] if sep_id is not None else [])
        )
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        windows.append(
            Window(
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids),
                token_start=start,
                token_end=end,
                char_start=char_start,
                char_end=char_end,
            )
        )
        if end >= n:
            break
        start += stride
    return windows


def pad_windows(windows: list[Window], pad_id: int, max_length: int) -> list[Window]:
    """Right-pad every window to ``max_length`` with ``pad_id``."""
    padded: list[Window] = []
    for w in windows:
        pad = max_length - len(w.input_ids)
        if pad < 0:
            ids = w.input_ids[:max_length]
            mask = w.attention_mask[:max_length]
        else:
            ids = w.input_ids + [pad_id] * pad
            mask = w.attention_mask + [0] * pad
        padded.append(
            Window(
                input_ids=ids,
                attention_mask=mask,
                token_start=w.token_start,
                token_end=w.token_end,
                char_start=w.char_start,
                char_end=w.char_end,
            )
        )
    return padded
