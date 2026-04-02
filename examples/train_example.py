"""example training script for blt"""

import torch
from blt.models.blt_model import ByteLatentTransformer
from blt.utils.trainer import AsyncBLTTrainer
from blt.utils.data_loader import ByteDataLoader


def create_sample_data():
    """create sample byte sequences for demonstration"""
    # sample text data
    texts = [
        "hello world! this is a test of the byte latent transformer.",
        "the quick brown fox jumps over the lazy dog.",
        "machine learning models process data in various ways.",
        "byte-level models can handle any text without tokenization.",
        "entropy-based patching allocates compute where needed.",
    ]
    
    # convert to byte sequences
    byte_sequences = []
    for text in texts:
        bytes_data = text.encode('utf-8')
        byte_tensor = torch.tensor(list(bytes_data), dtype=torch.long)
        byte_sequences.append(byte_tensor)
    
    return byte_sequences


def main():
    print("[PASS] initializing blt training example")
    
    # create sample data
    train_sequences = create_sample_data()
    val_sequences = create_sample_data()[:2]
    
    # create data loaders
    train_loader = ByteDataLoader(
        byte_sequences=train_sequences,
        batch_size=2,
        max_seq_len=128,
        shuffle=True,
    )
    
    val_loader = ByteDataLoader(
        byte_sequences=val_sequences,
        batch_size=2,
        max_seq_len=128,
        shuffle=False,
    )
    
    # create model
    model = ByteLatentTransformer(
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        max_patch_size=8,
        min_patch_size=2,
        entropy_threshold=0.5,
        dropout=0.1,
        use_async=True,
        max_workers=2,
    )
    
    print(f"[PASS] model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # create trainer
    trainer = AsyncBLTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        device="cpu",
        checkpoint_dir="./checkpoints",
        log_interval=10,
    )
    
    # train
    print("[PASS] starting training")
    trainer.train(num_epochs=5)
    
    print("[PASS] training example completed")


if __name__ == "__main__":
    main()
