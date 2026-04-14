"""
small causal byte-level LM used to drive entropy-based patch boundaries (blt).

training convention: logits[t] is the distribution over the next byte byte_ids[t + 1]
given causal context byte_ids[:, : t + 1]. loss uses logits[:, :-1] vs targets byte_ids[:, 1:].
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from blt.base_transformer import BaseTransformer, BaseTransformerArgs

BYTE_VOCAB = 256


def default_entropy_lm_args() -> BaseTransformerArgs:
    """small defaults; swap for paper-scale entropy model when training."""
    return BaseTransformerArgs(
        dim=256,
        n_layers=4,
        n_heads=8,
        vocab_size=BYTE_VOCAB,
        max_seqlen=2048,
        ffn_dim_multiplier=4.0,
    )


class EntropyLM(nn.Module):
    """
    byte embedding + BaseTransformer trunk. forward returns [B, T, 256] logits.
    """

    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        if args.vocab_size != BYTE_VOCAB:
            raise ValueError(
                f"vocab_size must be {BYTE_VOCAB} for raw bytes, got {args.vocab_size}"
            )
        self.args = args
        self.byte_embed = nn.Embedding(BYTE_VOCAB, args.dim, padding_idx=None)
        self.trunk = BaseTransformer(args)
        nn.init.normal_(self.byte_embed.weight, mean=0.0, std=0.02)

    def forward(
        self, byte_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if byte_ids.dtype not in (torch.long, torch.int32):
            byte_ids = byte_ids.long()
        x = self.byte_embed(byte_ids)
        return self.trunk(x, mask=mask)

    def per_position_entropy(
        self, byte_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """shannon entropy h_t of softmax(logits[t]) over 256 bytes, shape [B, T]."""
        logits = self.forward(byte_ids, mask=mask)
        log_p = F.log_softmax(logits.float(), dim=-1)
        p = log_p.exp()
        h = -(p * log_p).sum(dim=-1)
        return h

    def lm_loss(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """next-byte ce: predict byte_ids[:, 1:] from logits[:, :-1]."""
        logits = self.forward(byte_ids)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, BYTE_VOCAB),
            byte_ids[:, 1:].reshape(-1),
        )
