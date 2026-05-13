# Byte-ing Off More Than You Can Chew

CS 4782 Final Project. Akhil Kagithapu, Srikar Karra, Laerdon Kim, Ezra Koreen (Equal Contribution)

This repo is our ultra low parameter-scale replication and experimentation on FaceBook Research's [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) (BLT). BLT avoids fixed BPE tokens by operating on bytes, grouping
bytes into entropy patches, running a latent transformer over patch states, and
decoding those patch states back into byte predictions.

Our main target was the BLT scaling-law result: under a fixed FLOP budget, BLT
should keep improving in BPB more favorably than a GPT-2-style tokenizer model.
Because we trained at about 86M parameters instead of paper scale (~5B params), we report
final validation BPB and late-training BPB trends rather than claiming the full
crossover point.

<p align="center">
  <img width="683" height="510" alt="image" src="https://github.com/user-attachments/assets/ff804955-1bdb-468b-948d-539958bd67df" />
</p>

## Reimplemented Files

We reimplemented BLT architecture is concentrated in four files:

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

## Patching And Representations

The patch-length embedding experiment is implemented in `bytelatent/model/blt.py`.
After the local encoder builds patch states, the model can convert `patch_lengths`
into sinusoidal embeddings with the same hidden size and add them directly to the
patch states before the global transformer. This is controlled by
`use_patch_length_sinusoidal_embedding` and enabled in
`bytelatent/configs/patch_len_embeddings.yaml`.

## GPT-2 Baseline

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

## Training Configs

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

## Experiment Summary

| Variant | Config | N-gram | Params | BPB |
| --- | --- | --- | --- | --- |
| GPT-2 Baseline | `apps/main/configs/distilgpt2_hf_fineweb1p7b.yaml` | BPE | 49M | 1.22 |
| Small Encoder baseline | `bytelatent/configs/tiny_enc_baseline.yaml` | 3, 4, 5 | 86M | 1.27 |
| Small Decoder adversarial | `bytelatent/configs/tiny_dec_adversarial.yaml` | 4 | 86M | 1.31 |
| Large Encoder + Decoder | `bytelatent/configs/large_enc_dec_345_20k.yaml` / `large_enc_dec_345_50k.yaml` | 3, 4, 5 | 86M | 1.21 |
| Large Encoder + Decoder best | `bytelatent/configs/large_enc_dec_3x4_50k.yaml` | 4 | 86M | 1.21 |
| Patch length embeddings | `bytelatent/configs/patch_len_embeddings.yaml` | 4 | 86M | 1.26 |

Main takeaways: at this ultra-low scale, parameter allocation matters a lot.
Shrinking either local side too aggressively creates a bottleneck. A concentrated
4-gram hash setup was better than spreading the same idea across multiple
n-gram lengths with smaller tables, and sinusoidal patch-length embeddings gave
us a direct patch-representation ablation.

## Environment Setup

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

