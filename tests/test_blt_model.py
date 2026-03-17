"""tests for blt model"""

import torch
from blt.models.blt_model import ByteLatentTransformer, BLTTransformerLayer


def test_blt_initialization():
    """test blt model can be initialized"""
    model = ByteLatentTransformer(
        d_model=128,
        num_layers=2,
        num_heads=4,
    )
    assert model.d_model == 128
    assert model.num_layers == 2
    print("[PASS] blt model initialization")


def test_transformer_layer():
    """test single transformer layer"""
    layer = BLTTransformerLayer(
        d_model=128,
        num_heads=4,
        d_ff=512,
    )
    
    # create input
    x = torch.randn(2, 10, 128)
    
    output = layer(x)
    
    assert output.shape == x.shape
    print("[PASS] transformer layer forward pass")


def test_blt_forward():
    """test blt forward pass"""
    model = ByteLatentTransformer(
        d_model=128,
        num_layers=2,
        num_heads=4,
    )
    
    # create sample byte sequence
    byte_sequence = torch.randint(0, 256, (50,), dtype=torch.long)
    
    logits, patch_sizes = model(byte_sequence)
    
    assert logits.shape[1] == 256  # vocab size
    assert len(patch_sizes) == logits.shape[0]
    print(f"[PASS] blt forward pass - output shape: {logits.shape}")


def test_blt_batch_forward():
    """test blt batch forward pass"""
    model = ByteLatentTransformer(
        d_model=128,
        num_layers=2,
        num_heads=4,
    )
    
    # create batch of sequences
    sequences = [
        torch.randint(0, 256, (30,), dtype=torch.long),
        torch.randint(0, 256, (40,), dtype=torch.long),
    ]
    
    logits, patch_sizes = model.forward_batch(sequences)
    
    assert logits.shape[0] == 2  # batch size
    assert logits.shape[2] == 256  # vocab size
    assert len(patch_sizes) == 2
    print(f"[PASS] blt batch forward pass - output shape: {logits.shape}")


def test_blt_generation():
    """test text generation"""
    model = ByteLatentTransformer(
        d_model=128,
        num_layers=2,
        num_heads=4,
    )
    model.eval()
    
    # create prompt
    prompt = torch.randint(0, 256, (10,), dtype=torch.long)
    
    with torch.no_grad():
        generated = model.generate(prompt, max_length=20)
    
    assert generated.shape[0] == 30  # prompt + generated
    print(f"[PASS] text generation - generated {generated.shape[0]} bytes")


def test_model_parameters():
    """test model has trainable parameters"""
    model = ByteLatentTransformer(
        d_model=128,
        num_layers=2,
        num_heads=4,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0
    print(f"[PASS] model has {num_params:,} parameters")


if __name__ == "__main__":
    test_blt_initialization()
    test_transformer_layer()
    test_blt_forward()
    test_blt_batch_forward()
    test_blt_generation()
    test_model_parameters()
    print("\n[PASS] all blt model tests passed")
