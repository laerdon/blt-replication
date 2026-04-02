"""tests for byte encoder"""

import torch
from blt.models.byte_encoder import ByteEncoder


def test_byte_encoder_initialization():
    """test byte encoder can be initialized"""
    encoder = ByteEncoder(
        d_model=128,
        max_patch_size=8,
        min_patch_size=2,
    )
    assert encoder.d_model == 128
    assert encoder.max_patch_size == 8
    assert encoder.min_patch_size == 2
    print("[PASS] byte encoder initialization")


def test_entropy_computation():
    """test entropy computation"""
    encoder = ByteEncoder(d_model=128)
    
    # create random logits
    logits = torch.randn(1, 256)
    entropy = encoder.compute_entropy(logits)
    
    assert entropy.shape == (1,)
    assert 0 <= entropy.item() <= 1
    print("[PASS] entropy computation")


def test_patch_segmentation():
    """test segmentation into patches"""
    encoder = ByteEncoder(
        d_model=128,
        max_patch_size=8,
        min_patch_size=2,
    )
    
    # create sample byte sequence
    byte_sequence = torch.randint(0, 256, (50,), dtype=torch.long)
    
    patches, patch_sizes = encoder.segment_into_patches(byte_sequence)
    
    assert len(patches) == len(patch_sizes)
    assert all(2 <= size <= 8 for size in patch_sizes)
    assert sum(patch_sizes) == 50
    print(f"[PASS] patch segmentation - {len(patches)} patches")


def test_patch_encoding():
    """test encoding single patch"""
    encoder = ByteEncoder(d_model=128)
    
    # create sample patch
    patch = torch.randint(0, 256, (5,), dtype=torch.long)
    
    patch_emb = encoder.encode_patch(patch)
    
    assert patch_emb.shape == (128,)
    print("[PASS] patch encoding")


def test_forward_pass():
    """test full forward pass"""
    encoder = ByteEncoder(d_model=128)
    
    # create sample byte sequence
    byte_sequence = torch.randint(0, 256, (50,), dtype=torch.long)
    
    patch_embeddings, patch_sizes = encoder(byte_sequence)
    
    assert patch_embeddings.shape[1] == 128
    assert len(patch_sizes) == patch_embeddings.shape[0]
    print(f"[PASS] forward pass - output shape: {patch_embeddings.shape}")


def test_batch_encoding():
    """test batch encoding"""
    encoder = ByteEncoder(d_model=128)
    
    # create batch of sequences
    sequences = [
        torch.randint(0, 256, (30,), dtype=torch.long),
        torch.randint(0, 256, (40,), dtype=torch.long),
        torch.randint(0, 256, (25,), dtype=torch.long),
    ]
    
    batch_embeddings, batch_patch_sizes = encoder.encode_batch(sequences)
    
    assert batch_embeddings.shape[0] == 3
    assert batch_embeddings.shape[2] == 128
    assert len(batch_patch_sizes) == 3
    print(f"[PASS] batch encoding - output shape: {batch_embeddings.shape}")


if __name__ == "__main__":
    test_byte_encoder_initialization()
    test_entropy_computation()
    test_patch_segmentation()
    test_patch_encoding()
    test_forward_pass()
    test_batch_encoding()
    print("\n[PASS] all byte encoder tests passed")
