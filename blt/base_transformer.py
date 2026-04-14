import abc
import logging
import os
from enum import Enum
from typing import Optional, Tuple, Union

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn import functional as F

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        _mask_mod_signature,
        flex_attention,
    )
except ImportError:
    BlockMask = None  # type: ignore[misc, assignment]
    _mask_mod_signature = None  # type: ignore[misc, assignment]
    flex_attention = None  # type: ignore[misc, assignment]

try:
    from xformers.ops import AttentionBias, fmha
except ImportError:
    AttentionBias = None  # type: ignore[misc, assignment]
    fmha = None  # type: ignore[misc, assignment]

# from bytelatent.tokenizers.constants import EOS_ID

logger = logging.getLogger()

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm

    RMSNorm = FusedRMSNorm
except (ImportError, ModuleNotFoundError):
    logging.debug("Apex not found. Using nn.RMSNorm")
    RMSNorm = nn.RMSNorm

if flex_attention is not None and int(os.environ.get("BLT_ALLOW_MISSING_FLEX_ATTENTION", False)) == 0:
    flex_attention_comp = torch.compile(flex_attention)
else:
    flex_attention_comp = None

"""
we take the same code imports from the original. Overall we want to use the same bells and whistles as BLT's og implementation
so that we can spend as little money as possible.

We will also not be creating our own attention kernels or anything like that. Similarly to BLT's og implemntation,
we will use xformers for efficient attention when we can and nn SDPA when we can't.

We will also be using Pydantic to define our model arguments.

Overall, we want to implement a few important classes here:


- Base Transformer
- Attention
- a FFNN
- a Transformer Block
- RoPE

also, similarly to the BLT original implementation, we could also implement the sequence stacked format.
TODO do we want to implement this?
"""

# --- ALL THE ARGUMENT CLASSES ---

class BaseTransformerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int | None = None
    n_kv_heads: int | None = None
    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256
    vocab_size: int = 32000  # default llama-2-ish size
    max_seqlen: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
# in practice, we use GQA, so this is why we have n_kv_heads.
# but since we set to None this kinda means we just do traditional MHA.


# --- modules and stuff ---


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


# TODO i need to understand this better
class RoPE(nn.Module):
    """
    rotary position embeddings applied to q and k after projection.
    q, k: [batch, n_heads, n_ctx, head_dim]
    """

    def __init__(self, head_dim: int, max_seqlen: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("[rope] head_dim must be even")
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_ctx = q.shape[2]
        if n_ctx > self.max_seqlen:
            raise ValueError(f"[rope] sequence length {n_ctx} exceeds max_seqlen {self.max_seqlen}")
        t = torch.arange(n_ctx, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(q.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=q.dtype).view(1, 1, n_ctx, self.head_dim)
        sin = emb.sin().to(dtype=q.dtype).view(1, 1, n_ctx, self.head_dim)
        q_out = (q * cos) + (_rotate_half(q) * sin)
        k_out = (k * cos) + (_rotate_half(k) * sin)
        return q_out, k_out


class Attention(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.args = args
        n_heads = args.n_heads
        if n_heads is None:
            raise ValueError("[attention] n_heads is required")
        head_dim = args.head_dim if args.head_dim is not None else args.dim // n_heads
        self.rope = RoPE(
            head_dim=head_dim,
            max_seqlen=args.max_seqlen,
            theta=args.rope_theta,
        )
        self.q_proj = nn.Linear(args.dim, args.dim)
        self.k_proj = nn.Linear(args.dim, args.dim)
        self.v_proj = nn.Linear(args.dim, args.dim)
        self.out_proj = nn.Linear(args.dim, args.dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        steps:
        1. assign all of our dimensions to variables
        2. get the q k v's using our linear projections
        3. bring the QKV into a good view,
        4. apply rope to q and k, then sdpa.
        """

        b, n_ctx, _ = x.shape
        n_heads = self.args.n_heads
        if n_heads is None:
            raise ValueError("[attention] n_heads is required")
        head_dim = self.args.head_dim if self.args.head_dim is not None else self.args.dim // n_heads

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [b, n_ctx, dim] -> [b, n_heads, n_ctx, head_dim] for sdpa
        q = q.view(b, n_ctx, n_heads, head_dim).transpose(1, 2)
        k = k.view(b, n_ctx, n_heads, head_dim).transpose(1, 2)
        v = v.view(b, n_ctx, n_heads, head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        # scaling + softmax + matmul@v live here; use is_causal only when no explicit mask
        if mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(b, n_ctx, self.args.dim)
        return self.out_proj(out)

class FFN(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.args = args
        mult = args.ffn_dim_multiplier if args.ffn_dim_multiplier is not None else 4.0
        hidden = int(args.dim * mult)
        self.fc1 = nn.Linear(args.dim, hidden)
        self.fc2 = nn.Linear(hidden, args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.args = args
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attention = Attention(args)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn = FFN(args)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class BaseTransformer(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.args = args
        self.transformer_blocks = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.lm_head = nn.Linear(args.dim, args.vocab_size)
        self.apply(self.init_weights)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x, mask)
        return self.lm_head(x)

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm) and getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)

    # we only need an init_weights function here because
    # we initialize everything here.