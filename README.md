# Byte-ing Off More Than You Can Chew

## 1. Introduction

This repo is our ultra low parameter-scale replication and experimentation on FaceBook Research's [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) (BLT). BLT avoids fixed BPE tokens by operating on bytes, grouping
bytes into entropy patches, running a latent transformer over patch states, and
decoding those patch states back into byte predictions.

## 2. Chosen result

We targeted the paper’s fixed FLOP budget scaling claim. Under a comparable compute budget, BLT’s design should allow more favorable bits-per-byte (BPB) scaling than a tokenizer-style baseline. In the original work this is summarized in Figure 1 (scaling trends for fixed FLOP models across training budgets, including BPE vs. BLT crossover behavior). At ~86M parameters and a ~1.7B-token budget, we can only partially reproduce the full crossover; we instead report final validation BPB and the overall trend of our BPB curve against a matched GPT-2-style baseline.

![validation / scaling style plot from our runs](https://github.com/user-attachments/assets/ff804955-1bdb-468b-948d-539958bd67df)

## 3. GitHub contents


### Reimplemented files

Our reimplemented BLT architecture is concentrated in four files:

- `bytelatent/base_transformer.py`: the shared transformer trunk. This includes
  base args, RMSNorm fallback, RoPE cache/build/apply logic, grouped-query
  attention, SDPA/flex/xformers attention paths, SwiGLU feed-forward blocks,
  transformer blocks, and depth-aware init.
- `bytelatent/model/blt.py`: the top-level BLT module. This owns the
  `ByteLatentTransformer` arg schema, patch length to patch id conversion,
  encoder/global/decoder construction, cross-attention mask construction,
  hash-byte-group embeddings, and
  the full forward path. This is where we experimented with modifying hash n-grams and implementing patch length embeddings.
- `bytelatent/model/local_models.py`: the byte-side local encoder and decoder.
  The encoder builds byte states and optionally updates patch states through
  cross-attention. The decoder combines token states with global patch states,
  either by direct patch-state addition or decoder cross-attention.
- `bytelatent/model/latent_transformer.py`: the patch-side transformer and BLT
  cross-attention block. The global transformer consumes patch embeddings rather
  than token ids; cross-attention lets patch states read byte states or byte
  states read patch states without RoPE, since the two streams have different
  sequence layouts.

We intentionally left distributed/runtime infrastructure mostly intact and
focused on the model code that defines the architecture.

### Patching, representations

The patch-length embedding experiment is implemented in `bytelatent/model/blt.py`.
After the local encoder builds patch states, the model can convert `patch_lengths`
into sinusoidal embeddings with the same hidden size and add them directly to the
patch states before the global transformer. This is controlled by
`use_patch_length_sinusoidal_embedding` and enabled in
`bytelatent/configs/patch_len_embeddings.yaml`.

### GPT-2 baseline

`apps/main/train_distilgpt2.py` is the HuggingFace GPT-2 baseline trainer we made to compare with the BLT architecture. It builds a `GPT2LMHeadModel` from config, streams FineWeb-Edu text, tokenizes with
the `distilgpt2` BPE tokenizer, trains with AdamW and cosine LR, logs BPB using
actual UTF-8 byte counts, estimates cumulative FLOPs, runs validation, supports
checkpoint resume, and supports single-node DDP through `torchrun`.

The main baseline config is:

- `apps/main/configs/distilgpt2_hf_fineweb1p7b.yaml`: 6-layer, 768-dim,
  12-head GPT-2-style model on FineWeb-Edu for 25,940 steps, matching the
  approximate 1.7B-token BLT budget.

The helper launcher is:

- `setup/train_distilgpt2_2xh100.sh`: launches the HF GPT-2 trainer on two local
  GPUs with `torchrun`, defaulting to the config above.

Older GPT-2-shaped configs also exist in `apps/main/configs/distilgpt2_83m*.yaml`.
Those follow the repo's BLT training-config shape and were useful during setup,
but the HF trainer path above is the original baseline path.

### Training configs

The BLT configs inherit from each other using the `config:` key.

- `bytelatent/configs/h100_fineweb_1p7b_bs8.yaml`: main 1.7B-token BLT budget.
  Uses 4,096-byte sequences, two data-parallel ranks in the comments, 25,940
  steps, eval/checkpoint every 1,297 steps, and validation BPB.
- `bytelatent/configs/tiny_enc_baseline.yaml`: small-encoder baseline. Uses
  byte-group hashes `[3, 4, 5]` with 20k hash vocab, shrinks the local encoder
  to 1 layer, and reallocates depth into global/decoder layers.
- `bytelatent/configs/tiny_dec_adversarial.yaml`: small-decoder adversarial run.
  Keeps encoder capacity larger, grows the global path, and shrinks the local
  decoder to 1 layer.
- `bytelatent/configs/large_enc_dec_345_20k.yaml`: large encoder/decoder
  3-4-5 n-gram run with 20k hash vocab and extra global depth.
- `bytelatent/configs/large_enc_dec_345_50k.yaml`: 3-4-5 n-gram variant with
  50k hash vocab.
- `bytelatent/configs/large_enc_dec_3x4_50k.yaml`: best 4-gram-style run. It
  keeps the inherited `[4]` byte group with three hash functions and a ~50k hash
  table, giving several independent 4-gram hash views instead of spreading
  the vocabulary size parameters across 3, 4, and 5-grams.
- `bytelatent/configs/patch_len_embeddings.yaml`: patch representation ablation.
  This uses `use_patch_length_sinusoidal_embedding` so the global transformer sees
  patch length as an extra signal.

### Env setups

To run experiments on Prime Intellect H100s, we needed a repeatable way to turn
a fresh Ubuntu GPU box into a BLT training machine. `setup/create_env_uv.sh`
handles that setup end to end.

At a high level, the script prepares shared SSH access, installs the CUDA/build
toolchain needed for torch and xformers, creates the `uv` virtual environment,
builds xformers against the local GPU/CUDA setup, syncs Python dependencies, and
copies the prepared FineWeb train/validation Arrow shards onto the machine. It
also validates the copied Arrow files and prints a final torch/xformers CUDA
check so we know the instance is ready for training.

The script is meant to be rerunnable because Prime machines were not persistent
enough to rely on manual setup. It assumes sudo access, an NVIDIA driver that
supports CUDA 12.x, and the private key used to copy the prepared FineWeb data.

## 4. Re-implementation details

Approach: we forked and adapted Meta’s open BLT-style codebase, concentrated changes in the model core (`bytelatent/model/blt.py`, `local_models.py`, `latent_transformer.py`, `base_transformer.py`), and left most distributed training plumbing intact. Data: FineWeb-Edu–style mix, ~1.7B training tokens, with offline entropy patching using the gated Hugging Face model [`facebook/blt-entropy`](https://huggingface.co/facebook/blt-entropy) (see `data/DATA.md`). Metrics: validation BPB (UTF-8 byte counts), cumulative FLOPs logging on the baseline. Experiments: byte n-gram hash tables, encoder/decoder width ablations (“tiny enc”, “tiny dec”), optional sinusoidal patch-length embeddings (`patch_len_embeddings.yaml`). Challenges: sub-billion-parameter scale dominates allocation trade-offs; hash vocabulary layout and local vs. global depth matter more than at paper scale; xformers and CUDA 12 builds require a reproducible machine setup (`create_env_uv.sh`).

## 5. Reproduction steps

1. Hardware: NVIDIA GPU(s) with CUDA 12.x (we used H100-class nodes on Prime Intellect; a smaller GPU may run smoke configs but not full budgets at the same speed). Python 3.12 is pinned in `code/pyproject.toml`.
2. Environment: from `code/`, use `uv` to sync dependencies and build `xformers` per `pyproject.toml` (non-trivial; for a fresh Ubuntu GPU box see `code/setup/create_env_uv.sh`, which also documents data layout env vars such as `BLT_DATA_ROOT`).
3. Data: either use the 5M-token validation subset bundled under `data/` where provided, request access for pre-shuffled train shards (see `data/DATA.md`), or run `code/setup/shuffle_split_arrow.sh` after HF access to the entropy model to build preprocessed Arrow under the paths expected by the YAML configs (`data/preprocess/...`, `data/validation/...`).
4. Train BLT (from `code/`, OmegaConf overrides supported):  
   `torchrun --standalone --nnodes=1 --nproc_per_node=<GPUS> -m bytelatent.train config=bytelatent/configs/h100_fineweb_1p7b_bs8.yaml`  
   Adjust `distributed.dp_shard` / batch settings in YAML to match your GPU count (configs comment intended 2× data-parallel ranks for the 1.7B budget). Cheaper smoke: `bytelatent/configs/h100_val_canary.yaml`.
5. Train GPT-2 baseline:  
   `code/setup/train_distilgpt2_2xh100.sh` (wraps `torchrun -m apps.main.train_distilgpt2 config=apps/main/configs/distilgpt2_hf_fineweb1p7b.yaml`). Override `CONFIG` / `NPROC_PER_NODE` / `CUDA_VISIBLE_DEVICES` as needed.

## 6. Results / insights

At this scale, parameter allocation across local encoder, latent transformer, and local decoder matters more than the paper’s headline scaling curve; aggressive shrinking of either local path hurts BPB. A concentrated 4-gram hash setup edged out spreading capacity across 3–5-gram tables with similar total hash budget; patch-length sinusoidal embeddings were a clear ablation. Compared to the original Figure 1 takeaway (BLT overtaking BPE at large scale under FLOP control), someone cloning this repo should expect architecture-faithful training code and matched-token-budget BPB numbers, not an equivalent replication of the paper’s multi-billion-parameter crossover.

| Variant | Config | Tokenization / hash | Params | Val BPB |
| --- | --- | --- | --- | --- |
| GPT-2 baseline | `apps/main/configs/distilgpt2_hf_fineweb1p7b.yaml` | BPE (`distilgpt2`) | ~49M | 1.22 |
| Small encoder baseline | `bytelatent/configs/tiny_enc_baseline.yaml` | 3,4,5-gram | ~86M | 1.27 |
| Small decoder adversarial | `bytelatent/configs/tiny_dec_adversarial.yaml` | 4-gram | ~86M | 1.31 |
| Large enc+dec (20k / 50k hash) | `large_enc_dec_345_20k.yaml` / `large_enc_dec_345_50k.yaml` | 3,4,5-gram | ~86M | 1.21 |
| Large enc+dec (best) | `bytelatent/configs/large_enc_dec_3x4_50k.yaml` | 4-gram (3× hash, ~50k) | ~86M | 1.21 |
| Patch length embeddings | `bytelatent/configs/patch_len_embeddings.yaml` | 4-gram | ~86M | 1.26 |

## 7. Conclusion

This repo delivers a working, low-parameter BLT stack with entropy-patched FineWeb training. The main lesson is that paper-scale scaling claims (Figure 1) do not trivially transfer downward: at tens of millions of parameters, hash design and local/global parameter size balance are primary.

## 8. References

- Pagnoni, A., et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* [arXiv:2412.09871](https://arxiv.org/abs/2412.09871) (2024/2025).
- Original implementation (upstream): [facebookresearch/blt](https://github.com/facebookresearch/blt).
- Entropy model for patching (gated): [facebook/blt-entropy](https://huggingface.co/facebook/blt-entropy) on Hugging Face.
- FineWeb / FineWeb-Edu data ecosystem (as used in our pipeline): see Hugging Face FineWeb / FineWeb-Edu dataset cards linked from course materials or `data/DATA.md`.

## 9. Acknowledgements

CS 4782 final project (Cornell). Equal contribution: Akhil Kagithapu, Srikar Karra, Laerdon Kim, Ezra Koreen. Thanks to course staff for feedback and to Prime Intellect for GPU time used in environment bring-up and long runs.
