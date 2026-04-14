"""
todo scaffold for lm transformer implementation.

this file should mirror only the model-level parts we need from
`blt/bytelatent/transformer.py`, not training/distributed infrastructure.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from pydantic import ConfigDict
from torch import nn
from torch.nn import functional as F

from src.model.base_transformer import BaseTransformer, BaseTransformerArgs


class LMTransformerArgs(BaseTransformerArgs):
    """
    todo: keep this args class small and focused on lm-level behavior.
    """

    model_config = ConfigDict(extra="forbid")
    seed: int = 42
    vocab_size: int = -1
    weight_tying: bool = False
    sliding_window: int | None = None
    attn_bias_type: str | None = "causal"
    attn_impl: str | None = "sdpa"


class LMTransformer(BaseTransformer):
    """
    todo checklist copied from upstream intent, in implementation order:
    1) wire lm-specific modules: token embedding, final norm, output projection.
    2) validate args at init time (vocab_size > 0, head dimensions sane).
    3) optionally tie output weights to embedding weights when weight_tying is true.
    4) in forward(), embed token ids -> hidden states.
    5) construct or pass through causal/local-block mask.
    6) call BaseTransformer.forward(...) with the chosen mask + attn implementation.
    7) project to logits [batch, seq_len, vocab_size].
    8) if target is provided, return next-token cross entropy; else return logits.
    9) add reset/init method parity for norm/output/embedding.
    10) keep behavior deterministic with dtype/device-safe mask handling.

    todo on interface parity:
    - keep forward signature close to upstream:
      token_values, target=None, tok_idx=None, mask=None, attn_impl=None
    - keep output semantics identical:
      target is None -> logits, target is tensor -> scalar loss

    todo on what to intentionally defer in this file:
    - huggingface hub mixins
    - fused kernel specific branches not needed for first working version
    - distributed/tp/fsdp wiring
    """

    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)

        # validate all args
        if args.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {args.vocab_size}")
        if args.n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {args.n_heads}")
        head_dim = args.head_dim if args.head_dim is not None else args.dim // args.n_heads
        if args.dim % args.n_heads != 0 and args.head_dim is None:
            raise ValueError(
                f"dim ({args.dim}) must be divisible by n_heads ({args.n_heads}) when head_dim is not set"
            )
        if head_dim * args.n_heads != args.dim:
            raise ValueError(
                f"head_dim * n_heads must equal dim, got {head_dim} * {args.n_heads} != {args.dim}"
            )

        
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        self.vocab_size = args.vocab_size

        # todo: replace placeholder modules with final upstream-parity definitions.
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # todo: validate weight tying behavior against base module naming.
        if args.weight_tying:
            self.output.weight = self.tok_embeddings.weight

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
        attn_impl: str | None = None,
    ) -> torch.Tensor:
        if token_values.dtype not in (torch.long, torch.int32, torch.int64):
            token_values = token_values.long()

        h = self.tok_embeddings(token_values)
        h = super().forward(h, mask=mask)

        logits = self.output(self.norm(h))
        if target is not None:
            return F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                target.reshape(-1),
            )
        return logits

        # we just return logits

    def reset_parameters(self, init_std: float | None = None) -> None:
        del init_std  # kept for future parity with upstream signature.
        # we don't really need to worry too much about initialization here, so we can just use xavier uniform
        nn.init.xavier_uniform_(self.tok_embeddings.weight)
        nn.init.ones_(self.norm.weight)
        nn.init.xavier_uniform_(self.output.weight)
