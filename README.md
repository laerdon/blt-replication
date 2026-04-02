# async byte latent transformer (blt)

an asynchronous implementation of the byte latent transformer architecture with entropy-based dynamic patching.

## overview

this implementation provides a byte-level language model that processes raw bytes without tokenization. key features include:

- **entropy-based patching**: dynamically sized patches based on next-byte prediction entropy
- **async processing**: parallel patch processing for improved throughput
- **scalable architecture**: transformer-based model with configurable depth and width
- **no tokenization**: works directly with raw bytes, handling any text or data

## architecture

### byte encoder
- converts raw byte sequences into variable-length patches
- uses entropy of next-byte prediction to determine patch boundaries
- higher entropy (more uncertainty) = longer patches = more compute allocated

### async patch processor
- processes patches in parallel using async/await patterns
- includes async attention layers for efficient computation
- configurable worker pool for parallel processing

### transformer layers
- multi-head self-attention with async capabilities
- feed-forward networks with gelu activation
- layer normalization and residual connections
- positional encoding for patch sequences

## installation

```bash
pip install -e .
```

or install dependencies directly:

```bash
pip install -r requirements.txt
```

## usage

### basic inference

```python
import torch
from blt.models.blt_model import ByteLatentTransformer

# create model
model = ByteLatentTransformer(
    d_model=512,
    num_layers=6,
    num_heads=8,
    max_patch_size=16,
    min_patch_size=4,
    use_async=True,
)

# encode text to bytes
text = "hello world"
byte_tensor = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)

# forward pass
logits, patch_sizes = model(byte_tensor)
print(f"patches: {len(patch_sizes)}, sizes: {patch_sizes}")

# generate text
generated = model.generate(byte_tensor, max_length=50)
```

### training

```python
from blt.models.blt_model import ByteLatentTransformer
from blt.utils.trainer import AsyncBLTTrainer
from blt.utils.data_loader import ByteDataLoader

# prepare data
train_loader = ByteDataLoader(
    data_path="train.txt",
    batch_size=8,
    max_seq_len=512,
)

# create model
model = ByteLatentTransformer(
    d_model=512,
    num_layers=6,
    num_heads=8,
)

# create trainer
trainer = AsyncBLTTrainer(
    model=model,
    train_loader=train_loader,
    learning_rate=1e-4,
    device="cuda",
)

# train
trainer.train(num_epochs=10)
```

## examples

see the `examples/` directory for complete examples:

- `train_example.py`: training script with sample data
- `inference_example.py`: inference and generation example

run examples:

```bash
python examples/train_example.py
python examples/inference_example.py
```

## testing

run tests with pytest:

```bash
python tests/test_byte_encoder.py
python tests/test_blt_model.py
python tests/test_async_processor.py
```

or run all tests:

```bash
python -m pytest tests/
```

## model configuration

key hyperparameters:

- `d_model`: model dimension (default: 512)
- `num_layers`: number of transformer layers (default: 6)
- `num_heads`: number of attention heads (default: 8)
- `d_ff`: feed-forward dimension (default: 2048)
- `max_patch_size`: maximum bytes per patch (default: 16)
- `min_patch_size`: minimum bytes per patch (default: 4)
- `entropy_threshold`: threshold for patch segmentation (default: 0.5)
- `use_async`: enable async processing (default: true)
- `max_workers`: number of async workers (default: 4)

## performance considerations

- async processing provides speedup for variable-length patches
- entropy-based patching allocates more compute to complex regions
- batch processing is supported but patches are variable-length
- gpu acceleration recommended for training

## references

based on the byte latent transformer paper:
- paper: "byte latent transformer: patches scale better than tokens"
- original implementation: https://github.com/facebookresearch/blt

## license

mit license
