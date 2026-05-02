import json
import logging
import math
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, lr_scheduler
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerBase

from bytelatent.config_parser import parse_args_with_default

logger = logging.getLogger(__name__)


class ModelArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vocab_size: int | None = None
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 6
    n_head: int = 12
    n_inner: int | None = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False


class TokenizerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name_or_path: str = "distilgpt2"
    cache_dir: str | None = None
    revision: str | None = None
    use_fast: bool = True


class DataArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = "sample-10BT"
    data_files: Any | None = None
    cache_dir: str | None = None
    train_split: str = "train"
    validation_split: str = "train"
    text_column: str = "text"
    streaming: bool = True
    shuffle_buffer_size: int = 10000
    seq_len: int = 1024
    batch_size: int = 32
    validation_batch_size: int = 2
    add_eos: bool = True
    train_examples: int | None = None
    validation_examples: int | None = 2048


class OptimArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: float = 1e-4
    weight_decay: float = 0.033
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95
    clip: float = 1.0
    scheduler: str = "cosine"
    warmup: int = 1297
    lr_min_ratio: float = 0.000001
    cycle_length: float = 1.0
    cosine_theta: float = 1.0
    fused: bool | None = None


class FrequencyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    every: int = 1000
    keep: int = 2


class CheckpointArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str | None = None
    dump: FrequencyArgs = Field(default_factory=FrequencyArgs)
    eval: FrequencyArgs = Field(default_factory=lambda: FrequencyArgs(every=1000, keep=-1))
    resume_from: str | None = None
    save_final: bool = True


class LoggingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    freq: int = 25


class ValidationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    split: str | None = None
    batch_size: int | None = None
    max_n_batches: int | None = 40
    max_n_docs: int | None = None


class EvalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_ppl: bool = True
    every: int | None = None
    validation: ValidationArgs = Field(default_factory=ValidationArgs)


class RuntimeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: str = "auto"
    dtype: str = "bf16"
    compile: bool = False
    matmul_allow_tf32: bool = True
    log_level: str = "INFO"
    # used only when WORLD_SIZE>1 (torchrun); default None -> nccl on cuda
    distributed_backend: str | None = None


class TrainArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dump_dir: str = "./runs/distilgpt2_hf"
    name: str = "distilgpt2_hf"
    seed: int = 777
    steps: int = 1000
    max_steps: int | None = None
    grad_acc_steps: int = 1
    model: ModelArgs = Field(default_factory=ModelArgs)
    tokenizer: TokenizerArgs = Field(default_factory=TokenizerArgs)
    data: DataArgs = Field(default_factory=DataArgs)
    optim: OptimArgs = Field(default_factory=OptimArgs)
    checkpoint: CheckpointArgs = Field(default_factory=CheckpointArgs)
    logging: LoggingArgs = Field(default_factory=LoggingArgs)
    eval: EvalArgs | None = Field(default_factory=EvalArgs)
    runtime: RuntimeArgs = Field(default_factory=RuntimeArgs)


def parse_config(argv: list[str] | None = None) -> TrainArgs:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 1 and "=" not in argv[0]:
        cli_cfg = OmegaConf.create({"config": argv[0]})
    else:
        cli_cfg = OmegaConf.from_dotlist(argv)
    default_cfg = OmegaConf.create(TrainArgs().model_dump(mode="json"))
    cfg = parse_args_with_default(default_cfg=default_cfg, cli_args=cli_cfg)
    return TrainArgs.model_validate(cfg)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda was requested but torch.cuda.is_available() is false")
    return resolved


def get_dist_env() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", "0")),
        )
    return 0, 1, 0


def unwrap_model(model: torch.nn.Module) -> GPT2LMHeadModel:
    if isinstance(model, DDP):
        return model.module
    return model


def maybe_destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "fp32":
        return nullcontext()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported runtime dtype: {dtype_name}")
    return torch.autocast(device_type=device.type, dtype=dtype_map[dtype_name])


def build_lr_lambda(args: OptimArgs, n_steps: int):
    def warmup_scale(step: int) -> float:
        if args.warmup <= 0:
            return 1.0
        return min(1.0, float(step) / float(args.warmup))

    def lr_constant(step: int) -> float:
        return warmup_scale(step)

    def lr_linear(step: int) -> float:
        if step < args.warmup:
            return warmup_scale(step)
        if n_steps <= args.warmup:
            return args.lr_min_ratio
        progress = float(step - args.warmup) / float(n_steps - args.warmup)
        return max(args.lr_min_ratio, progress * args.lr_min_ratio + (1.0 - progress))

    def lr_cosine(step: int) -> float:
        if step < args.warmup:
            return warmup_scale(step)
        if n_steps <= args.warmup:
            return args.lr_min_ratio
        progress = min(1.0, float(step - args.warmup) / float(n_steps - args.warmup))
        return args.lr_min_ratio + 0.5 * (1.0 - args.lr_min_ratio) * (
            math.cos(math.pi * progress**args.cosine_theta / args.cycle_length) + 1.0
        )

    if args.scheduler == "constant":
        return lr_constant
    if args.scheduler == "linear":
        return lr_linear
    if args.scheduler == "cosine":
        return lr_cosine
    raise NotImplementedError(f"unknown scheduler: {args.scheduler}")


def build_model(args: TrainArgs, tokenizer: PreTrainedTokenizerBase) -> GPT2LMHeadModel:
    vocab_size = args.model.vocab_size or len(tokenizer)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.model.n_positions,
        n_ctx=args.model.n_positions,
        n_embd=args.model.n_embd,
        n_layer=args.model.n_layer,
        n_head=args.model.n_head,
        n_inner=args.model.n_inner,
        activation_function=args.model.activation_function,
        resid_pdrop=args.model.resid_pdrop,
        embd_pdrop=args.model.embd_pdrop,
        attn_pdrop=args.model.attn_pdrop,
        layer_norm_epsilon=args.model.layer_norm_epsilon,
        initializer_range=args.model.initializer_range,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        scale_attn_by_inverse_layer_idx=args.model.scale_attn_by_inverse_layer_idx,
        reorder_and_upcast_attn=args.model.reorder_and_upcast_attn,
    )
    return GPT2LMHeadModel(config)


def load_text_dataset(
    args: DataArgs,
    split: str,
    seed: int,
    shuffle: bool,
    *,
    rank: int = 0,
    world_size: int = 1,
):
    kwargs: dict[str, Any] = {
        "path": args.dataset_name,
        "name": args.dataset_config,
        "split": split,
        "streaming": args.streaming,
        "cache_dir": args.cache_dir,
    }
    if args.data_files is not None:
        kwargs["data_files"] = args.data_files
    dataset = load_dataset(**kwargs)
    if shuffle:
        if args.streaming:
            dataset = dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=seed)
        else:
            dataset = dataset.shuffle(seed=seed)
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)
    return dataset


def iter_text(dataset, text_column: str, max_docs: int | None) -> Iterator[str]:
    for idx, row in enumerate(dataset):
        if max_docs is not None and idx >= max_docs:
            break
        if text_column not in row:
            raise KeyError(f"text column {text_column!r} was not found in dataset row")
        text = row[text_column]
        if not isinstance(text, str):
            raise TypeError(f"text column {text_column!r} must contain strings")
        if text:
            yield text


def count_token_bytes(
    token_ids: list[int], tokenizer: PreTrainedTokenizerBase, special_ids: set[int]
) -> int:
    special_count = sum(1 for token_id in token_ids if token_id in special_ids)
    text_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    text = tokenizer.decode(
        text_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return len(text.encode("utf-8", errors="ignore")) + special_count


def iter_token_blocks(
    *,
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    data_args: DataArgs,
    max_docs: int | None,
) -> Iterator[dict[str, Any]]:
    token_buffer: list[int] = []
    eos_token_id = tokenizer.eos_token_id
    special_ids = set(tokenizer.all_special_ids)
    for text in iter_text(dataset, data_args.text_column, max_docs):
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if data_args.add_eos:
            if eos_token_id is None:
                raise ValueError("data.add_eos is true, but the tokenizer has no eos token")
            token_ids = token_ids + [eos_token_id]
        token_buffer.extend(token_ids)
        while len(token_buffer) >= data_args.seq_len:
            block = token_buffer[: data_args.seq_len]
            token_buffer = token_buffer[data_args.seq_len :]
            yield {
                "input_ids": block,
                "n_bytes": count_token_bytes(block, tokenizer, special_ids),
            }


def make_batch_iterator(
    *,
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    data_args: DataArgs,
    batch_size: int,
    max_docs: int | None,
    device: torch.device,
) -> Iterator[dict[str, Any]]:
    block_iter = iter_token_blocks(
        dataset=dataset,
        tokenizer=tokenizer,
        data_args=data_args,
        max_docs=max_docs,
    )
    while True:
        blocks = []
        n_bytes = 0
        for _ in range(batch_size):
            block = next(block_iter)
            blocks.append(block["input_ids"])
            n_bytes += block["n_bytes"]
        input_ids = torch.tensor(blocks, dtype=torch.long, device=device)
        yield {
            "input_ids": input_ids,
            "n_bytes": n_bytes,
            "n_tokens": input_ids.numel(),
        }


def build_train_batches(
    args: TrainArgs,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    rank: int,
    world_size: int,
) -> Iterator[dict[str, Any]]:
    dataset = load_text_dataset(
        args.data,
        split=args.data.train_split,
        seed=args.seed,
        shuffle=True,
        rank=rank,
        world_size=world_size,
    )
    return make_batch_iterator(
        dataset=dataset,
        tokenizer=tokenizer,
        data_args=args.data,
        batch_size=args.data.batch_size,
        max_docs=args.data.train_examples,
        device=device,
    )


def cross_entropy_sum(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="sum",
    )
    return loss, shift_labels.numel()


def attention_flops_per_token(n_layers: int, seq_len: int, dim: int) -> float:
    return 3.5 * (4 * n_layers * seq_len * dim // 2)


def estimate_flops_per_token(model: GPT2LMHeadModel, seq_len: int) -> int:
    config = model.config
    total_params = sum(p.numel() for p in model.parameters())
    token_embedding_params = config.vocab_size * config.n_embd
    non_embed_params = total_params - token_embedding_params
    per_forward = 6 * non_embed_params + attention_flops_per_token(
        config.n_layer, seq_len, config.n_embd
    )
    return int(per_forward)


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    row = dict(row)
    row["created_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        print(json.dumps(row), file=handle, flush=True)


def prune_checkpoints(checkpoint_root: Path, keep: int) -> None:
    if keep < 0:
        return
    checkpoints = sorted(path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir())
    to_prune = checkpoints if keep == 0 else checkpoints[:-keep]
    for checkpoint in to_prune:
        for child in sorted(checkpoint.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                child.rmdir()
        checkpoint.rmdir()


def save_checkpoint(
    *,
    args: TrainArgs,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: AdamW,
    scheduler: lr_scheduler.LambdaLR,
    checkpoint_root: Path,
    global_step: int,
    tokens_seen: int,
    cumulative_flops: float,
    rank: int,
    world_size: int,
) -> Path | None:
    if rank != 0:
        if world_size > 1:
            dist.barrier()
        return None
    to_save = unwrap_model(model)
    checkpoint_dir = checkpoint_root / f"checkpoint-{global_step:010d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    to_save.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "cumulative_flops": cumulative_flops,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        checkpoint_dir / "trainer_state.pt",
    )
    state = {
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "cumulative_flops": cumulative_flops,
        "config": args.model_dump(mode="json"),
    }
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    prune_checkpoints(checkpoint_root, keep=args.checkpoint.dump.keep)
    if world_size > 1:
        dist.barrier()
    return checkpoint_dir


def load_checkpoint(
    *,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: lr_scheduler.LambdaLR,
    device: torch.device,
) -> tuple[int, int, float]:
    state_dict = GPT2LMHeadModel.from_pretrained(checkpoint_dir).state_dict()
    unwrap_model(model).load_state_dict(state_dict)
    state = torch.load(checkpoint_dir / "trainer_state.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if device.type == "cuda" and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state["global_step"]), int(state["tokens_seen"]), float(state["cumulative_flops"])


@torch.no_grad()
def run_validation(
    *,
    args: TrainArgs,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    global_step: int,
    cumulative_flops: float,
    metrics_path: Path,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    if args.eval is None or not args.eval.run_ppl:
        return {}
    model.eval()
    validation = args.eval.validation
    split = validation.split or args.data.validation_split
    batch_size = validation.batch_size or args.data.validation_batch_size
    max_docs = validation.max_n_docs or args.data.validation_examples
    dataset = load_text_dataset(
        args.data,
        split=split,
        seed=args.seed,
        shuffle=False,
        rank=rank,
        world_size=world_size,
    )
    batches = make_batch_iterator(
        dataset=dataset,
        tokenizer=tokenizer,
        data_args=args.data,
        batch_size=batch_size,
        max_docs=max_docs,
        device=device,
    )
    loss_sum = 0.0
    n_loss_tokens = 0
    n_bytes = 0
    n_batches = 0
    for batch in batches:
        if validation.max_n_batches is not None and n_batches >= validation.max_n_batches:
            break
        with autocast_context(device, args.runtime.dtype):
            logits = model(input_ids=batch["input_ids"]).logits
            loss, loss_tokens = cross_entropy_sum(logits, batch["input_ids"])
        loss_sum += float(loss.item())
        n_loss_tokens += loss_tokens
        n_bytes += int(batch["n_bytes"])
        n_batches += 1
    if world_size > 1:
        t = torch.tensor(
            [loss_sum, float(n_loss_tokens), float(n_bytes), float(n_batches)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_sum = float(t[0].item())
        n_loss_tokens = int(t[1].item())
        n_bytes = int(t[2].item())
        n_batches = int(t[3].item())
    if n_batches == 0:
        raise RuntimeError("validation produced no batches")
    if n_bytes <= 0:
        raise RuntimeError("validation byte count must be positive")
    loss_mean = loss_sum / n_loss_tokens
    row = {
        "global_step": global_step,
        "validation/loss_sum": loss_sum,
        "validation/loss_mean": loss_mean,
        "validation/n_tokens": n_loss_tokens,
        "validation/n_bytes": n_bytes,
        "validation/bpb": loss_sum / math.log(2) / n_bytes,
        "validation/ppl": math.exp(loss_mean),
        "validation/n_batches": n_batches,
        "flops/cumulative": cumulative_flops,
    }
    if rank == 0:
        write_jsonl(metrics_path, row)
    model.train()
    return row


def train(args: TrainArgs) -> None:
    rank, world_size, local_rank = get_dist_env()
    configure_logging(args.runtime.log_level)
    set_seed(args.seed)

    dump_dir = Path(args.dump_dir)
    checkpoint_root = Path(args.checkpoint.path or dump_dir / "checkpoints")

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("ddp requires cuda in this trainer")
        backend = args.runtime.distributed_backend or "nccl"
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend=backend)
        if args.runtime.device not in ("auto", "cuda", f"cuda:{local_rank}"):
            logger.warning(
                "ddp: ignoring runtime.device=%s; using cuda:%s",
                args.runtime.device,
                local_rank,
            )
    else:
        device = resolve_device(args.runtime.device)

    if rank == 0:
        dump_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        (dump_dir / "config.resolved.json").write_text(
            json.dumps(args.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.runtime.matmul_allow_tf32

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer.name_or_path,
            cache_dir=args.tokenizer.cache_dir,
            revision=args.tokenizer.revision,
            use_fast=args.tokenizer.use_fast,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = build_model(args, tokenizer)
        model.to(device)
        if args.runtime.compile:
            model = torch.compile(model)

        n_steps = args.max_steps or args.steps
        fused = args.optim.fused if args.optim.fused is not None else (device.type == "cuda")
        optimizer = AdamW(
            model.parameters(),
            lr=args.optim.lr,
            betas=(args.optim.beta1, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
            eps=args.optim.epsilon,
            fused=fused,
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, build_lr_lambda(args.optim, n_steps))

        global_step = 0
        tokens_seen = 0
        cumulative_flops = 0.0
        if args.checkpoint.resume_from is not None:
            global_step, tokens_seen, cumulative_flops = load_checkpoint(
                checkpoint_dir=Path(args.checkpoint.resume_from),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model.train()

        flops_per_token = estimate_flops_per_token(unwrap_model(model), args.data.seq_len)
        n_params = sum(p.numel() for p in unwrap_model(model).parameters())
        if rank == 0:
            logger.info("ddp world_size=%s rank=%s", world_size, rank)
            logger.info("model parameters: %s", f"{n_params:,}")
            logger.info("estimated forward flops per token (heuristic): %s", f"{flops_per_token:,}")

        metrics_path = dump_dir / "metrics.jsonl"
        validation_metrics_path = dump_dir / "metrics.validation.jsonl"
        train_batches = build_train_batches(args, tokenizer, device, rank, world_size)
        optimizer.zero_grad(set_to_none=True)

        use_amp_scaler = device.type == "cuda" and args.runtime.dtype == "fp16"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp_scaler)

        interval_loss_sum = 0.0
        interval_loss_tokens = 0
        interval_bytes = 0
        interval_tokens = 0
        interval_flops = 0.0
        interval_start = timer()
        micro_step = 0

        while global_step < args.steps and global_step < n_steps:
            batch = next(train_batches)
            with autocast_context(device, args.runtime.dtype):
                logits = model(input_ids=batch["input_ids"]).logits
                loss_sum, loss_tokens = cross_entropy_sum(logits, batch["input_ids"])
                loss = loss_sum / loss_tokens / args.grad_acc_steps

            if use_amp_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_step += 1
            interval_loss_sum += float(loss_sum.detach().item())
            interval_loss_tokens += loss_tokens
            interval_bytes += int(batch["n_bytes"])
            interval_tokens += int(batch["n_tokens"])
            tokens_seen += int(batch["n_tokens"])
            batch_flops = flops_per_token * int(batch["n_tokens"])
            cumulative_flops += batch_flops
            interval_flops += batch_flops

            if micro_step % args.grad_acc_steps != 0:
                continue

            if use_amp_scaler:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.optim.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.optim.clip)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging.freq == 0 or global_step == 1:
                if interval_bytes <= 0:
                    raise RuntimeError("training byte count must be positive")
                elapsed = max(timer() - interval_start, 1e-9)
                if world_size > 1:
                    stats = torch.tensor(
                        [
                            interval_loss_sum,
                            float(interval_loss_tokens),
                            float(interval_bytes),
                            float(interval_tokens),
                            interval_flops,
                            float(tokens_seen),
                            cumulative_flops,
                        ],
                        device=device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    g_ls = float(stats[0].item())
                    g_lt = int(stats[1].item())
                    g_ib = int(stats[2].item())
                    g_it = int(stats[3].item())
                    g_ifl = float(stats[4].item())
                    g_ts = int(stats[5].item())
                    g_cf = float(stats[6].item())
                else:
                    g_ls = interval_loss_sum
                    g_lt = interval_loss_tokens
                    g_ib = interval_bytes
                    g_it = interval_tokens
                    g_ifl = interval_flops
                    g_ts = tokens_seen
                    g_cf = cumulative_flops

                if g_ib <= 0:
                    raise RuntimeError("training byte count must be positive after reduce")
                row = {
                    "global_step": global_step,
                    "train/loss": g_ls / g_lt,
                    "train/loss_sum": g_ls,
                    "train/n_tokens": g_lt,
                    "train/n_bytes": g_ib,
                    "train/bpb": g_ls / math.log(2) / g_ib,
                    "optim/lr": float(optimizer.param_groups[0]["lr"]),
                    "optim/grad_norm": float(grad_norm.item()),
                    "tokens/seen": g_ts,
                    "flops/cumulative": g_cf,
                    "speed/tokens_per_sec": g_it / elapsed,
                    "speed/flops_per_sec": g_ifl / elapsed,
                }
                if rank == 0:
                    write_jsonl(metrics_path, row)
                    logger.info(
                        "step %s loss %.4f bpb %.4f lr %.3e flops %.3e",
                        global_step,
                        row["train/loss"],
                        row["train/bpb"],
                        row["optim/lr"],
                        row["flops/cumulative"],
                    )
                interval_loss_sum = 0.0
                interval_loss_tokens = 0
                interval_bytes = 0
                interval_tokens = 0
                interval_flops = 0.0
                interval_start = timer()

            if args.checkpoint.dump.every > 0 and global_step % args.checkpoint.dump.every == 0:
                ts_t = torch.tensor([float(tokens_seen)], device=device, dtype=torch.float64)
                cf_t = torch.tensor([float(cumulative_flops)], device=device, dtype=torch.float64)
                if world_size > 1:
                    dist.all_reduce(ts_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(cf_t, op=dist.ReduceOp.SUM)
                save_checkpoint(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint_root=checkpoint_root,
                    global_step=global_step,
                    tokens_seen=int(ts_t.item()),
                    cumulative_flops=float(cf_t.item()),
                    rank=rank,
                    world_size=world_size,
                )

            eval_every = None
            if args.eval is not None:
                eval_every = args.eval.every or args.checkpoint.eval.every
            if eval_every and eval_every > 0 and global_step % eval_every == 0:
                cf_t = torch.tensor([float(cumulative_flops)], device=device, dtype=torch.float64)
                if world_size > 1:
                    dist.all_reduce(cf_t, op=dist.ReduceOp.SUM)
                cumulative_flops_global = float(cf_t.item())
                val_row = run_validation(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    global_step=global_step,
                    cumulative_flops=cumulative_flops_global,
                    metrics_path=validation_metrics_path,
                    rank=rank,
                    world_size=world_size,
                )
                if rank == 0 and val_row:
                    logger.info(
                        "validation step %s bpb %.4f ppl %.4f",
                        global_step,
                        val_row["validation/bpb"],
                        val_row["validation/ppl"],
                    )

        if args.checkpoint.save_final:
            ts_t = torch.tensor([float(tokens_seen)], device=device, dtype=torch.float64)
            cf_t = torch.tensor([float(cumulative_flops)], device=device, dtype=torch.float64)
            if world_size > 1:
                dist.all_reduce(ts_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(cf_t, op=dist.ReduceOp.SUM)
            save_checkpoint(
                args=args,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_root=checkpoint_root,
                global_step=global_step,
                tokens_seen=int(ts_t.item()),
                cumulative_flops=float(cf_t.item()),
                rank=rank,
                world_size=world_size,
            )
    finally:
        maybe_destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    train(parse_config(argv))


if __name__ == "__main__":
    main()