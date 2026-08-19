"""Neural document classifier: window encoder + heads.

The :class:`WindowEncoder` wraps a Hugging Face base encoder and produces, for
each window, a 2-class logit (human/AI) and the ``[CLS]`` window embedding. The
:class:`HierarchicalSummaryHead` pools a document's window embeddings with a
single Transformer layer plus a learned-query attention pool to produce a
document-level logit; it is the preregistered alternative to plain
overlap-corrected logit averaging.

Both heads expose logits for the AI class; aggregation converts window logits
to a document score (see :mod:`bench.neural.aggregate`).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class WindowEncoder(nn.Module):
    """A base transformer encoder with a window-level classification head.

    Pass ``hf_name`` to load pretrained weights; pass a prebuilt ``encoder``
    (with ``hidden_size``) to inject a model, e.g. a tiny config-built model in
    tests.
    """

    def __init__(
        self,
        hf_name: str | None = None,
        num_labels: int = 2,
        dropout: float = 0.1,
        encoder: nn.Module | None = None,
        hidden_size: int | None = None,
    ):
        super().__init__()
        self.hf_name = hf_name
        if encoder is not None:
            if hidden_size is None:
                raise ValueError("hidden_size is required when injecting an encoder")
            self.encoder = encoder
            hidden = int(hidden_size)
        else:
            if hf_name is None:
                raise ValueError("hf_name or encoder is required")
            # Load in full precision. transformers>=5 defaults to the checkpoint
            # dtype (DeBERTa-v3 ships fp16), which both mismatches the fp32 head
            # and diverges to NaN under half-precision training. fp32 master
            # weights + bf16 autocast at the call site are stable and fast.
            self.encoder = AutoModel.from_pretrained(hf_name, dtype=torch.float32)
            hidden = int(self.encoder.config.hidden_size)
        self.hidden_size = hidden
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS]; DeBERTa and ModernBERT are BERT-style
        logits = self.classifier(self.dropout(cls))
        return logits, cls

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HierarchicalSummaryHead(nn.Module):
    """Pool window embeddings into a document logit.

    One Transformer encoder layer contextualizes the window embeddings, a
    learned query attention-pools them (masking padding windows), and a linear
    layer emits the document's 2-class logits. Trained on frozen window
    embeddings so it is compared against logit averaging on an equal footing.
    """

    def __init__(
        self,
        hidden: int,
        num_labels: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden % nhead != 0:
            nhead = 4 if hidden % 4 == 0 else 1
        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.pool_query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, window_embeds, window_mask):
        # window_embeds: [B, W, H]; window_mask: [B, W] with True = real window.
        pad_mask = ~window_mask.bool()
        # A document with all-pad windows would NaN; guarantee at least one.
        pad_mask[:, 0] = False
        h = self.layer(window_embeds, src_key_padding_mask=pad_mask)
        scores = (h * self.pool_query).sum(-1)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        doc = (h * attn.unsqueeze(-1)).sum(1)
        return self.classifier(self.dropout(doc))
