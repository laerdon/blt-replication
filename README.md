# Byte-ing Off More Than You Can Chew

## 1. Introduction

This repo is our ultra low parameter-scale replication and experimentation on FaceBook Research's [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) (BLT). BLT avoids fixed BPE tokens by operating on bytes, grouping
bytes into entropy patches, running a latent transformer over patch states, and
decoding those patch states back into byte predictions.

## 2. Chosen result

We targeted the paper’s fixed FLOP budget scaling claim. Under a comparable compute budget, BLT’s design should allow more favorable bits-per-byte (BPB) scaling than a tokenizer-style baseline. In the original work this is summarized in Figure 1 (scaling trends for fixed FLOP models across training budgets, including BPE vs. BLT crossover behavior). At ~86M parameters and a ~1.7B-token budget, we can only partially reproduce the full crossover; we instead report final validation BPB and the overall trend of our BPB curve against a matched GPT-2-style baseline.

![validation / scaling style plot from our runs](https://github.com/user-attachments/assets/ff804955-1bdb-468b-948d-539958bd67df)

## 3. GitHub contents

| Path | Role |
| --- | --- |
| `code/` | Python package (`bytelatent/`), training (`bytelatent/train.py`), HF GPT-2 baseline (`apps/main/train_distilgpt2.py`), YAML configs, `pyproject.toml` / `uv.lock` |
| `code/setup/` | Environment bootstrap (`create_env_uv.sh`), data shuffle/split (`shuffle_split_arrow.sh`), launch helpers |
| `data/` | Dataset notes (`DATA.md`); prepared Arrow shards may live here or be generated locally |
| `results/` | Saved run configs and metric logs for reported experiments |
| `report/`, `poster/` | Course write-up and poster PDFs |

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

At this scale, parameter allocation across local encoder, latent transformer, and local decoder matters more than the paper’s headline scaling curve; aggressive shrinking of either local path hurts BPB. A concentrated 4-gram hash setup edged out spreading capacity across 3–5-gram tables with similar total hash budget; patch-length sinusoidal embeddings were a clear ablation. Compared to the original Figure 1 takeaway (BLT overtaking BPE at large scale under FLOP control), someone cloning this repo should expect architecture-faithful training code and matched-token-budget BPB numbers, not a statistically equivalent replication of the paper’s multi-billion-parameter crossover.

| Variant | Config | Tokenization / hash | Params | Val BPB |
| --- | --- | --- | --- | --- |
| GPT-2 baseline | `apps/main/configs/distilgpt2_hf_fineweb1p7b.yaml` | BPE (`distilgpt2`) | ~49M | 1.22 |
| Small encoder baseline | `bytelatent/configs/tiny_enc_baseline.yaml` | 3,4,5-gram | ~86M | 1.27 |
| Small decoder adversarial | `bytelatent/configs/tiny_dec_adversarial.yaml` | 4-gram | ~86M | 1.31 |
| Large enc+dec (20k / 50k hash) | `large_enc_dec_345_20k.yaml` / `large_enc_dec_345_50k.yaml` | 3,4,5-gram | ~86M | 1.21 |
| Large enc+dec (best) | `bytelatent/configs/large_enc_dec_3x4_50k.yaml` | 4-gram (3× hash, ~50k) | ~86M | 1.21 |
| Patch length embeddings | `bytelatent/configs/patch_len_embeddings.yaml` | 4-gram | ~86M | 1.26 |

## 7. Conclusion

This repo delivers a working, low-parameter BLT stack with entropy-patched FineWeb training, a fair tokenizer baseline, and documented ablations—useful for teaching and for probing architectural sensitivities. The main lesson is that paper-scale scaling claims (Figure 1) do not trivially transfer downward: at tens of millions of parameters, hash design and local/global balance dominate before the “patches vs. tokens” story shows the same crossover.

## 8. References

- Pagnoni, A., et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* [arXiv:2412.09871](https://arxiv.org/abs/2412.09871) (2024/2025).
- Original implementation (upstream): [facebookresearch/blt](https://github.com/facebookresearch/blt).
- Entropy model for patching (gated): [facebook/blt-entropy](https://huggingface.co/facebook/blt-entropy) on Hugging Face.
- FineWeb / FineWeb-Edu data ecosystem (as used in our pipeline): see Hugging Face FineWeb / FineWeb-Edu dataset cards linked from course materials or `data/DATA.md`.

## 9. Acknowledgements

CS 4782 final project (Cornell). Equal contribution: Akhil Kagithapu, Srikar Karra, Laerdon Kim, Ezra Koreen. Thanks to course staff for feedback and to Prime Intellect for GPU time used in environment bring-up and long runs.
