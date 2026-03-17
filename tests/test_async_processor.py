"""tests for async patch processor"""

import torch
from blt.models.patch_processor import AsyncPatchProcessor, AsyncAttentionLayer


def test_async_processor_initialization():
    """test async processor can be initialized"""
    processor = AsyncPatchProcessor(max_workers=2)
    assert processor.max_workers == 2
    processor.close()
    print("[PASS] async processor initialization")


def test_async_attention_layer():
    """test async attention layer"""
    layer = AsyncAttentionLayer(
        d_model=128,
        num_heads=4,
        use_async=True,
    )
    
    # create input
    x = torch.randn(2, 10, 128)
    
    output = layer(x)
    
    assert output.shape == x.shape
    print("[PASS] async attention layer forward pass")


def test_attention_with_mask():
    """test attention with mask"""
    layer = AsyncAttentionLayer(
        d_model=128,
        num_heads=4,
    )
    
    # create input and mask
    x = torch.randn(2, 10, 128)
    mask = torch.ones(2, 1, 10, 10)
    mask[:, :, :, 5:] = 0  # mask out last 5 positions
    
    output = layer(x, mask)
    
    assert output.shape == x.shape
    print("[PASS] attention with mask")


def test_head_splitting():
    """test attention head splitting"""
    layer = AsyncAttentionLayer(
        d_model=128,
        num_heads=4,
    )
    
    x = torch.randn(2, 10, 128)
    split = layer.split_heads(x)
    
    assert split.shape == (2, 4, 10, 32)  # [batch, heads, seq, head_dim]
    
    merged = layer.merge_heads(split)
    assert merged.shape == x.shape
    print("[PASS] head splitting and merging")


if __name__ == "__main__":
    test_async_processor_initialization()
    test_async_attention_layer()
    test_attention_with_mask()
    test_head_splitting()
    print("\n[PASS] all async processor tests passed")
