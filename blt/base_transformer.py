import abc
import logging
import os
from enum import Enum
from typing import Optional, Tuple, Union

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    flex_attention,
)
from xformers.ops import AttentionBias, fmha

# from bytelatent.tokenizers.constants import EOS_ID

logger = logging.getLogger()

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm

    RMSNorm = FusedRMSNorm
except (ImportError, ModuleNotFoundError):
    logging.debug("Apex not found. Using nn.RMSNorm")
    RMSNorm = nn.RMSNorm

if int(os.environ.get("BLT_ALLOW_MISSING_FLEX_ATTENTION", False)) == 0:
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

class AttentionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: int = None
    ffn_dim_multiplier: float = 4.0
    multiple_of: int = 256

class BaseTransformerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim: int = 512
    n_layers: int = 8
    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    ffn_dim_multiplier: float | None = None
    multiple_of: int = 256
# in practice, we use GQA, so this is why we have n_kv_heads.
# but since we set to None this kinda means we just do traditional MHA.


# --- modules and stuff ---

class RoPE(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.args = args
        self.w = nn.Parameter(torch.randn(args.dim // 2))
        self.register_buffer("cos_cached", None)
        self.register_buffer("sin_cached", None)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        if self.cos_cached is None or self.sin_cached is None or self.cos_cached.shape[0] != seq_len:
            self.cos_cached = torch.cos(self.w)
            self.sin_cached = torch.sin(self.w)
        cos = self.cos_cached[None, :seq_len, None, :]
        sin = self.sin_cached[None, :seq_len, None, :]
        return x * cos + x * sin

class Attention(nn.Module):
    def __init__(self, args: AttentionArgs):
        super().__init__()
        self.args = args
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
        4. apply the mask, do the SDPA.
        """

        b, n_ctx, _ = x.shape
        n_heads = self.args.n_heads
        if n_heads is None:
            raise ValueError("[attention] n_heads is required")
        head_dim = getattr(self.args, "head_dim", None)
        if head_dim is None:
            head_dim = self.args.dim // n_heads

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [b, n_ctx, dim] -> [b, n_heads, n_ctx, head_dim] for sdpa
        q = q.view(b, n_ctx, n_heads, head_dim).transpose(1, 2)
        k = k.view(b, n_ctx, n_heads, head_dim).transpose(1, 2)
        v = v.view(b, n_ctx, n_heads, head_dim).transpose(1, 2)

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
        self.fc1 = nn.Linear(args.dim, args.dim * args.ffn_dim_multiplier)
        self.fc2 = nn.Linear(args.dim * args.ffn_dim_multiplier, args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.args = args
        self.attention = Attention(args)
        self.ffn = FFN(args)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(x, mask) + x
        x = self.ffn(x) + x
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
    # we only need an init_weights function here because
    # we initialize everything here.