import abc
import logging
import math
import os
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from xformers.ops import AttentionBias, fmha

from bytelatent.tokenizers.constants import EOS_ID

log = logging.getLogger()

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm
    RMSNorm = FusedRMSNorm
except Exception:
    log.debug("no apex, using torch RMSNorm")
    RMSNorm = nn.RMSNorm

_SKIP_FLEX = int(os.environ.get("BLT_ALLOW_MISSING_FLEX_ATTENTION", 0)) != 0
flex_attention_comp = None if _SKIP_FLEX else torch.compile(flex_attention)


class InitStdFactor(str, Enum):
    DISABLED = "disabled"
    GLOBAL_DEPTH = "global_depth"
    CURRENT_DEPTH = "current_depth"
    DIM_RATIO = "dim_ratio"


class BaseTransformerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim: int = 512
    n_layers: int = 8
    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None

    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0
    rope_use_fp32_in_outer_product: bool = False

    init_base_std: float | None = None
    init_std_factor: InitStdFactor = InitStdFactor.DISABLED

    max_seqlen: int = 1024

    attn_impl: str | None = "sdpa"
    attn_bias_type: str | None = None
    eos_id: int | None = EOS_ID


def cross_entropy(pred, target, **kw):
    logp = F.log_softmax(pred.flatten(end_dim=-2).float(), dim=-1)
    return F.nll_loss(logp, target.flatten(end_dim=-1), **kw)


def _trunc_init(t, std):
    nn.init.trunc_normal_(t, mean=0.0, std=std, a=-3 * std, b=3 * std)


def _default_std(dim, init_std, factor):
    return init_std if init_std is not None else (dim ** -0.5) / factor


def _gqa_expand(x, n_rep):
    if n_rep == 1:
        return x
    B, S, H, D = x.shape
    return x.unsqueeze(3).expand(B, S, H, n_rep, D).reshape(B, S, H * n_rep, D)


def _resolve_head_config(args):
    assert args.head_dim is not None or args.n_heads is not None, \
        "need head_dim or n_heads"
    head_dim = args.head_dim if args.head_dim is not None else args.dim // args.n_heads
    n_heads = args.n_heads if args.n_heads is not None else args.dim // args.head_dim
    n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else n_heads
    return head_dim, n_heads, n_kv_heads


def _build_rope_cache(head_dim, max_len, theta, fp32_outer=False):
    half = head_dim // 2
    idx = torch.arange(0, head_dim, 2)[:half].float()
    inv_f = 1.0 / (theta ** (idx / head_dim))
    pos = torch.arange(max_len, device=inv_f.device)
    if fp32_outer:
        pos = pos.float()
    angles = torch.outer(pos, inv_f).float()
    c, s = angles.cos(), angles.sin()
    return torch.stack((c, -s, s, c), dim=-1).view(*angles.shape, 2, 2)


def _broadcast_rope(freqs, x, seq_dim):
    assert 0 <= seq_dim < x.ndim
    expected = (x.shape[seq_dim], x.shape[-3], 2, 2)
    assert freqs.shape == expected, f"rope shape mismatch {freqs.shape} vs {x.shape}"
    keep = {seq_dim, x.ndim - 3}
    shape = [d if i in keep else 1 for i, d in enumerate(x.shape[:-2])] + [2, 2]
    return freqs.view(*shape)


def apply_rope(xq, xk, seq_dim, freqs):
    q_pair = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    k_pair = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    fb = _broadcast_rope(freqs, q_pair, seq_dim).float()
    out_q = (q_pair * fb).sum(dim=5).flatten(3)
    out_k = (k_pair * fb).sum(dim=5).flatten(3)
    return out_q.type_as(xq), out_k.type_as(xk)


class RotaryEmbedding(nn.Module):
    def __init__(self, theta, head_dim, max_seqlen=1024, rope_use_fp32_in_outer_product=False):
        super().__init__()
        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.rope_use_fp32_in_outer_product = rope_use_fp32_in_outer_product
        cache = _build_rope_cache(head_dim, max_seqlen, theta, rope_use_fp32_in_outer_product)
        self.register_buffer("freqs_cis", cache, persistent=False)

    def reset_parameters(self):
        self.freqs_cis[...] = _build_rope_cache(
            self.head_dim, self.max_seqlen, self.theta, self.rope_use_fp32_in_outer_product
        )

    def forward(self, seqlen=None, tok_idx=None):
        assert seqlen is not None or tok_idx is not None, "need one of seqlen / tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        return self.freqs_cis[:seqlen]


def causal_mask(b, h, q, kv):
    return q >= kv


def _doc_starts(lengths):
    starts = lengths.cumsum(0).roll(1)
    starts[0] = 0
    return starts


def _doc_positions(lengths):
    assert lengths.ndim == 1
    doc_id = torch.repeat_interleave(lengths)
    starts_per_tok = _doc_starts(lengths)[doc_id]
    pos = torch.arange(lengths.sum(), device=lengths.device) - starts_per_tok
    return doc_id, pos


def generate_doc_mask_mod(mask_mod, lengths, kv_lengths=None):
    if kv_lengths is None:
        kv_lengths = lengths
    q_doc, q_pos = _doc_positions(lengths)
    k_doc, k_pos = _doc_positions(kv_lengths)
    q_max = lengths.sum() - 1
    k_max = kv_lengths.sum() - 1

    def inner(b, h, q_idx, kv_idx):
        qi = torch.minimum(q_max, q_idx)
        ki = torch.minimum(k_max, kv_idx)
        valid = (q_idx <= q_max) & (kv_idx <= k_max)
        same = q_doc[qi] == k_doc[ki]
        return same & mask_mod(b, h, q_pos[qi], k_pos[ki]) & valid

    return inner


def lengths_to_start_ids(lengths):
    return _doc_starts(lengths)


def lengths_to_local_ids(lengths):
    return _doc_positions(lengths)


def _maybe_reshape_for_bias(bias, *ts):
    if isinstance(bias, fmha.attn_bias.BlockDiagonalCausalMask):
        return [t.reshape(1, -1, *t.shape[2:]) for t in ts]
    return list(ts)


class Attention(nn.Module):
    def __init__(self, dim, head_dim, n_heads, n_kv_heads, rope_theta):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.rope_theta = rope_theta
        self.heads_per_group = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def _project_qkv(self, x):
        B, S, _ = x.shape
        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)
        return xq, xk, xv

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        B, S, _ = x.shape

        xq, xk, xv = self._project_qkv(x)
        xq, xk = apply_rope(xq, xk, 1, freq_cis[:S])

        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = _gqa_expand(xk, self.heads_per_group)
        xv = _gqa_expand(xv, self.heads_per_group)

        if attn_impl == "sdpa":
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = isinstance(mask, str) and mask == "causal"
            m = mask if isinstance(mask, torch.Tensor) else None
            q, k, v = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, attn_mask=m)
            out = out.transpose(1, 2).contiguous()
        elif attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            q, k, v = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
            out = flex_attention_comp(q, k, v, block_mask=mask)
            out = out.transpose(1, 2).contiguous()
        elif attn_impl == "xformers":
            assert mask is None or isinstance(mask, AttentionBias)
            saved = xq.shape
            q, k, v = _maybe_reshape_for_bias(mask, xq, xk, xv)
            out = fmha.memory_efficient_attention(q, k, v, attn_bias=mask).view(saved)
        else:
            raise NotImplementedError(f"unknown attn_impl {attn_impl}")

        return self.wo(out.reshape(B, S, self.n_heads * self.head_dim))

    def reset_parameters(self, init_std=None, factor=1.0):
        std = _default_std(self.dim, init_std, factor)
        for w in (self.wq, self.wk, self.wv, self.wo):
            _trunc_init(w.weight, std)


def _swiglu_hidden(dim_in, multiple_of, ffn_mult):
    h = int(2 * dim_in / 3)
    if ffn_mult is not None:
        h = int(ffn_mult * h)
    return multiple_of * math.ceil(h / multiple_of)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier, mp_size=1):
        super().__init__()
        hidden_dim = _swiglu_hidden(hidden_dim, multiple_of, ffn_dim_multiplier)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self, init_std=None, factor=1.0):
        std_in = _default_std(self.dim, init_std, factor)
        std_out = _default_std(self.hidden_dim, init_std, factor)
        _trunc_init(self.w1.weight, std_in)
        _trunc_init(self.w3.weight, std_in)
        _trunc_init(self.w2.weight, std_out)


class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        head_dim, n_heads, n_kv_heads = _resolve_head_config(args)
        assert args.n_heads % n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.attention = Attention(
            dim=args.dim,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        attn_out = self.attention(
            self.attention_norm(x), freq_cis,
            tok_idx=tok_idx, mask=mask, attn_impl=attn_impl,
        )
        h = x + attn_out
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class SequenceModelWithOutput(abc.ABC):
    @abc.abstractmethod
    def get_output_seq_len(self) -> int:
        ...


class BaseTransformer(nn.Module, SequenceModelWithOutput):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.attn_impl = args.attn_impl
        self.attn_bias_type = args.attn_bias_type
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.eos_id = args.eos_id

        rope_head_dim, _, _ = _resolve_head_config(args)
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=rope_head_dim,
            max_seqlen=args.max_seqlen,
            rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
        )

        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])

    def get_output_seq_len(self):
        return self.max_seqlen

    def forward(self, h, tok_idx=None, mask=None, attn_impl="sdpa"):
        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)
        for layer in self.layers:
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        return h

    def _depth_factor(self, depth):
        f = self.init_std_factor
        if f == InitStdFactor.CURRENT_DEPTH:
            return (2 * (depth + 1)) ** 0.5
        if f == InitStdFactor.GLOBAL_DEPTH:
            return (2 * (len(self.layers) + 1)) ** 0.5
        if f == InitStdFactor.DIM_RATIO:
            return self.dim / 4096
        return 1.0

    def init_weights(self):
        self.rope_embeddings.reset_parameters()
        for depth, layer in enumerate(self.layers):
            layer.init_weights(self.init_base_std, self._depth_factor(depth))
