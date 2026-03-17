"""async patch processor for parallel processing of patches"""

import asyncio
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor


class AsyncPatchProcessor:
    """
    processes patches asynchronously for improved throughput.
    useful for handling variable-length patches in parallel.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        device: str = "cpu",
    ):
        self.max_workers = max_workers
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_patch_async(
        self,
        patch: torch.Tensor,
        processor_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        process a single patch asynchronously.
        
        args:
            patch: patch tensor to process
            processor_fn: function to apply to patch
        returns:
            processed patch
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            processor_fn,
            patch,
        )
        return result
    
    async def process_patches_parallel(
        self,
        patches: List[torch.Tensor],
        processor_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        process multiple patches in parallel.
        
        args:
            patches: list of patch tensors
            processor_fn: function to apply to each patch
        returns:
            list of processed patches
        """
        tasks = [
            self.process_patch_async(patch, processor_fn)
            for patch in patches
        ]
        results = await asyncio.gather(*tasks)
        return results
    
    def process_patches_sync(
        self,
        patches: List[torch.Tensor],
        processor_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        synchronous wrapper for async patch processing.
        
        args:
            patches: list of patch tensors
            processor_fn: function to apply to each patch
        returns:
            list of processed patches
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.process_patches_parallel(patches, processor_fn)
        )
    
    def close(self):
        """cleanup executor resources"""
        self.executor.shutdown(wait=True)


class AsyncAttentionLayer(nn.Module):
    """
    attention layer with async patch processing capabilities.
    processes attention for different patches in parallel.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_async: bool = True,
        max_workers: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_async = use_async
        
        assert d_model % num_heads == 0, "[ERROR] d_model must be divisible by num_heads"
        
        self.head_dim = d_model // num_heads
        
        # projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        if use_async:
            self.async_processor = AsyncPatchProcessor(max_workers=max_workers)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        split into multiple attention heads.
        
        args:
            x: [batch_size, seq_len, d_model]
        returns:
            [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        merge attention heads back.
        
        args:
            x: [batch_size, num_heads, seq_len, head_dim]
        returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        compute scaled dot-product attention.
        
        args:
            q: [batch_size, num_heads, seq_len, head_dim]
            k: [batch_size, num_heads, seq_len, head_dim]
            v: [batch_size, num_heads, seq_len, head_dim]
            mask: optional attention mask
        returns:
            [batch_size, num_heads, seq_len, head_dim]
        """
        # compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        
        # apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        forward pass with optional async processing.
        
        args:
            x: [batch_size, seq_len, d_model]
            mask: optional attention mask
        returns:
            [batch_size, seq_len, d_model]
        """
        # project to q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # split into heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # compute attention
        attn_output = self.compute_attention(q, k, v, mask)
        
        # merge heads
        output = self.merge_heads(attn_output)
        
        # final projection
        output = self.out_proj(output)
        
        return output
    
    def __del__(self):
        """cleanup async processor"""
        if hasattr(self, 'async_processor'):
            self.async_processor.close()
