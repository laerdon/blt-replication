# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
this file has the latent-side transformer pieces.

crossattention is the tiny block that lets one stream read from another stream.
in blt that usually means byte states reading patch states, or patch states
reading byte states. it does not use rope because the q side and kv side can
come from different sequence layouts.

globaltransformer is the patch-level transformer. the caller gives it patch
embeds instead of token ids, then this class makes the mask if needed, projects
the embed size if needed, runs the normal transformer stack, and hands the
cache back unchanged for now.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias

from bytelatent.base_transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    flex_attention_comp,
    repeat_kv,
)
from bytelatent.model.utils import create_causal_mask

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm

    RMSNorm = FusedRMSNorm
except (ImportError, ModuleNotFoundError):
    logging.debug("apex not found. using nn.rmsnorm")
    RMSNorm = nn.RMSNorm


class CrossAttention(nn.Module):
    """
    cross attention between one hidden stream and another.

    this is used when decoder byte states need to read encoder patch states,
    or vice versa. no rope.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        # this assumes n_heads is divisible by n_kv_heads, same as before.
        # bad configs will break later when the heads get shaped for attention.
        self.heads_per_group = n_heads // n_kv_heads

        # todo(srikar): can u look at q norm vs fused kv norm?
        self.cross_attn_norm_q = nn.RMSNorm(dim, eps=norm_eps)
        self.cross_attn_norm_kv = RMSNorm(dim, eps=norm_eps)

        # q gets all heads. k/v can have fewer heads, then we copy them below.
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def _make_qkv(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        batch_size: int,
        q_len: int,
        kv_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # normalize first, then make q/k/v. this matches the rest of this model.
        q = self.wq(self.cross_attn_norm_q(x))
        kv_normed = self.cross_attn_norm_kv(kv)
        k = self.wk(kv_normed)
        v = self.wv(kv_normed)

        # flex attention wants [batch, heads, seq, head_dim], but linear gives
        # [batch, seq, heads * head_dim], so split heads first.
        q = q.view(batch_size, q_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, kv_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, kv_len, self.n_kv_heads, self.head_dim)

        # grouped-query attention: each kv head may be shared by many q heads.
        if self.heads_per_group != 1:
            k = repeat_kv(k, self.heads_per_group, dim=2)
            v = repeat_kv(v, self.heads_per_group, dim=2)

        return q, k, v

    def _run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[Union[BlockMask, AttentionBias, str]],
        batch_size: int,
        q_len: int,
    ) -> torch.Tensor:
        # todo(ezra): can u look at letting xformers masks pass through here too?
        assert mask is None or isinstance(mask, BlockMask)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = flex_attention_comp(q, k, v, block_mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous()
        return attn_out.view(batch_size, q_len, self.n_heads * self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
    ) -> torch.Tensor:
        batch_size, q_len, _ = x.shape
        kv_len = kv.shape[1]

        q, k, v = self._make_qkv(x, kv, batch_size, q_len, kv_len)
        attn_out = self._run_attention(q, k, v, mask, batch_size, q_len)

        # keep the old residual behavior: this module returns x + attention.
        return x + self.wo(attn_out)

    def init_weights(self, base_std: Optional[float], factor: float = 1.0):
        std = base_std or (self.dim ** (-0.5)) / factor

        # same init for all four projections, written as one loop so it is less
        # copy-pastey but still does the same thing.
        for proj in (self.wq, self.wk, self.wv, self.wo):
            nn.init.trunc_normal_(
                proj.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )

        self.cross_attn_norm_q.reset_parameters()
        self.cross_attn_norm_kv.reset_parameters()


class GlobalTransformer(BaseTransformer):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__(args)

        self.dropout = args.dropout
        self.eos_id = args.eos_id
        self.dim_token_emb = args.dim_token_emb

        # local/patch embeddings may not be the same width as the global model.
        self.token_embedding_projection = None
        needs_projection = (
            args.dim_token_emb is not None and args.dim_token_emb != self.dim
        )
        if needs_projection:
            self.token_embedding_projection = nn.Linear(
                args.dim_token_emb,
                args.dim,
                bias=False,
            )

    def _mask_or_default(
        self,
        tokens: torch.Tensor,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]],
    ) -> Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]]:
        if mask is not None:
            return mask

        # callers can pass a prepared mask, but global forward can also build
        # its own causal/block mask from token boundaries.
        return create_causal_mask(
            tokens.shape[1],
            self.attn_impl,
            self.attn_bias_type,
            tokens=tokens,
            eos_id=self.eos_id,
        )

    def _project_embeds_if_needed(self, h: torch.Tensor) -> torch.Tensor:
        if self.token_embedding_projection is None or h.shape[-1] == self.dim:
            return h
        return self.token_embedding_projection(h)

    def forward(
        self,
        tokens: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """
        run the global transformer on already-built patch embeddings.
        """
        h = embeds

        mask = self._mask_or_default(tokens, mask)
        h = self._project_embeds_if_needed(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # base transformer owns rope, layer loop, attention impl selection, etc.
        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=self.attn_impl)

        # todo(laerdon): can u look at cache stuff for global decoding?
        return h, cache

    def init_weights(self):
        super().init_weights()

        if self.token_embedding_projection is None:
            return

        std = self.dim_token_emb ** (-0.5)
        nn.init.trunc_normal_(
            self.token_embedding_projection.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
