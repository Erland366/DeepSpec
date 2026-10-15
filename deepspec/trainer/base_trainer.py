from contextlib import nullcontext
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from deepspec.data import build_cache_dataset, validate_train_cache
from deepspec.data.cuda_prefetcher import CUDAPrefetcher
from deepspec.utils import (
    BF16Optimizer,
    StatelessResumableDistributedSampler,
    ensure_dir,
    init_dist,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
)
from deepspec.trainer.ckpt_manager import (
    discover_latest_checkpoint,
    load_init_draft_model,
    load_resume_draft_model,
    load_training_state,
    save_checkpoint,
)
import deepspec.utils.training_logger as training_logger
from deepspec.utils.config import CustomJSONEncoder
from deepspec.utils.hfai_suspend import SuspendController


_PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_SHARDING_STRATEGIES = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    "hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    "_hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
}

_HYBRID_STRATEGIES = (
    ShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
)


def _build_fsdp_kwargs(
    *, sharding_strategy_name: str, precision_dtype, world_size: int
) -> dict:
    sharding_strategy = _SHARDING_STRATEGIES[sharding_strategy_name]
    fsdp_kwargs = dict(
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=precision_dtype,
            buffer_dtype=precision_dtype,
        ),
        sharding_strategy=sharding_strategy,
    )
    if sharding_strategy in _HYBRID_STRATEGIES:
        devices_per_node = torch.cuda.device_count()
        fsdp_kwargs["device_mesh"] = init_device_mesh(
            "cuda",
            (world_size // devices_per_node, devices_per_node),
            mesh_dim_names=("replicate", "shard"),
        )
    return fsdp_kwargs


def _compute_gradient_accumulation_steps(
    *, world_size: int, local_batch_size: int, global_batch_size: int
) -> int:
    denom = world_size * local_batch_size
    assert global_batch_size % denom == 0, (
        "global_batch_size must be divisible by world_size * local_batch_size: "
        f"global_batch_size={global_batch_size}, world_size={world_size}, "
        f"local_batch_size={local_batch_size}"
    )
    return global_batch_size // denom


def _compute_samples_per_epoch(*, dataset_size: int, global_batch_size: int) -> int:
    samples_per_epoch = (dataset_size // global_batch_size) * global_batch_size
    assert samples_per_epoch > 0, (
        "train dataset is too small to form one full global batch: "
        f"dataset_size={dataset_size}, global_batch_size={global_batch_size}"
    )
    return samples_per_epoch


def _compute_training_schedule(
    *,
    world_size: int,
    dataset_size: int,
    local_batch_size: int,
    global_batch_size: int,
    num_train_epochs: int,
    max_train_steps=None,
) -> tuple[int, int, int, int, int, int, int]:
    gradient_accumulation_steps = _compute_gradient_accumulation_steps(
        world_size=world_size,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
    )
    samples_per_epoch = _compute_samples_per_epoch(
        dataset_size=dataset_size,
        global_batch_size=global_batch_size,
    )
    per_rank_samples_per_epoch = samples_per_epoch // world_size
    micro_batches_per_epoch = per_rank_samples_per_epoch // local_batch_size
    steps_per_epoch = micro_batches_per_epoch // gradient_accumulation_steps
    if max_train_steps is None:
        resolved_max_train_steps = int(num_train_epochs) * steps_per_epoch
        resolved_num_train_epochs = int(num_train_epochs)
    else:
        resolved_max_train_steps = int(max_train_steps)
        resolved_num_train_epochs = math.ceil(
            resolved_max_train_steps / steps_per_epoch
        )
    return (
        gradient_accumulation_steps,
        samples_per_epoch,
        per_rank_samples_per_epoch,
        micro_batches_per_epoch,
        steps_per_epoch,
        resolved_max_train_steps,
        resolved_num_train_epochs,
    )


def _launch_eval(
    *,
    target_model_name_or_path: str,
    checkpoint_dir: str,
    step: int,
    tensorboard_dir: str,
    exp_name: str,
) -> None:
    from deepspec.utils.constant import auto_eval_command
    if auto_eval_command is not None:
        command = auto_eval_command(target_model_name_or_path,checkpoint_dir,step,tensorboard_dir,exp_name)
        print_on_global_main(f"Submitting auto eval for {checkpoint_dir}")
        print_on_global_main(command)
        os.system(command)
    else:
        print("You can use this function to launch your auto eval script!")

def _resolve_round(*, round_cfg, steps_per_epoch: int, max_train_steps: int, dataset_size: int):
    """train.round = {start_step, stop_step, sampler_epoch, num_samples}: train only global steps
    [start_step, stop_step) on this launch's cache (a chunk group), with the LR schedule of the whole run
    (train.max_train_steps = total over all rounds). The round makes whole passes over its cache, the first with the
    cache's epoch-`sampler_epoch` shuffle (seed + epoch), the next with sampler_epoch + 1, and so on; num_samples
    must equal the cache size the round was planned for."""
    if round_cfg is None:
        return None
    resolved = {key: int(round_cfg[key]) for key in ("start_step", "stop_step", "sampler_epoch", "num_samples")}
    assert 0 <= resolved["start_step"] < resolved["stop_step"] <= int(max_train_steps), (
        f"train.round must satisfy 0 <= start_step < stop_step <= max_train_steps={max_train_steps}: {resolved}"
    )
    assert resolved["num_samples"] == int(dataset_size), (
        f"train.round was planned for {resolved['num_samples']} samples but the cache has {dataset_size}: {resolved}"
    )
    assert (resolved["stop_step"] - resolved["start_step"]) % int(steps_per_epoch) == 0, (
        f"train.round spans {resolved['stop_step'] - resolved['start_step']} steps, not whole passes of "
        f"{steps_per_epoch} steps over its cache: {resolved}"
    )
    assert resolved["sampler_epoch"] >= 0
    return resolved


def _resolve_rounds(*, rounds_cfg, max_train_steps):
    """train.rounds = [{cache_dirs, start_step, stop_step, sampler_epoch, num_samples}, ...]: one launch trains the
    rounds in order, each on its own cache (a chunk group that is shipped in while earlier rounds train and deleted
    after its round-NNNN.done marker). Rounds must tile [0, max_train_steps) exactly."""
    if rounds_cfg is None:
        return None
    assert max_train_steps is not None, "train.rounds needs train.max_train_steps (the whole-run total)"
    rounds = []
    for cfg in rounds_cfg:
        cache_dirs = cfg["cache_dirs"]
        cache_dirs = [str(d) for d in cache_dirs] if isinstance(cache_dirs, (list, tuple)) else [str(cache_dirs)]
        rounds.append(dict({key: int(cfg[key]) for key in ("start_step", "stop_step", "sampler_epoch", "num_samples")},
                           cache_dirs=cache_dirs))
    assert rounds, "train.rounds is empty"
    expected_start = 0
    for index, rnd in enumerate(rounds):
        assert rnd["start_step"] == expected_start and rnd["stop_step"] > rnd["start_step"], (
            f"train.rounds[{index}] must start at {expected_start} and be non-empty: {rnd}"
        )
        expected_start = rnd["stop_step"]
    assert expected_start == int(max_train_steps), (
        f"train.rounds end at step {expected_start}, max_train_steps is {max_train_steps}"
    )
    return rounds


def _wait_for_cache_dirs(cache_dirs, *, timeout_s: float, poll_s: float = 30.0):
    """Block until every cache dir has its manifest (chunks are published by an atomic rename); fail after timeout."""
    start = time.time()
    last_report = 0.0
    while True:
        missing = [d for d in cache_dirs if not os.path.isfile(os.path.join(d, "manifest.json"))]
        if not missing:
            return time.time() - start
        waited = time.time() - start
        assert waited < timeout_s, f"cache dirs still missing after {waited:.0f} s: {missing}"
        if waited - last_report >= 300 or last_report == 0.0:
            print_on_global_main(f"Waiting for {len(missing)} cache dir(s), e.g. {missing[0]} ({waited:.0f} s)")
            last_report = max(waited, 1e-9)
        time.sleep(poll_s)


class BaseTrainer:
    data_collator_cls = None

    def __init__(self, local_rank, args):
        self.args = args
        self.device, self.global_rank, self.world_size = init_dist(local_rank)
        self.precision_dtype = _PRECISION_DTYPES[self.args.train.precision]
        self.checkpoint_dir_root = self.args.logging.checkpoint_dir
        self.resume_checkpoint_dir = discover_latest_checkpoint(
            self.checkpoint_dir_root
        )
        self.suspend_controller = SuspendController(device=self.device)
        self.next_micro_step = 0

        if is_global_main_process(): ensure_dir(self.checkpoint_dir_root)
        training_logger.init(
            logging_steps=int(self.args.logging.logging_steps),
            tensorboard_dir=self.args.logging.tensorboard_dir,
            wandb_project=self.args.logging.get("wandb_project"),
            wandb_entity=self.args.logging.get("wandb_entity"),
            wandb_name=self.args.logging.get("wandb_name"),
            wandb_id=self.args.logging.get("wandb_id"),
            config=json.loads(json.dumps(self.args, cls=CustomJSONEncoder)),
        )

        self.draft_model, self.tokenizer = self.build_models()
        if self.resume_checkpoint_dir is not None:
            self.draft_model = load_resume_draft_model(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                draft_model=self.draft_model,
                device=self.device,
                precision_dtype=self.precision_dtype,
                global_rank=self.global_rank,
            )
        init_from = self.args.train.get("init_from")
        if init_from and self.resume_checkpoint_dir is None:
            # Warm start: weights only; optimizer, LR schedule, step and sampler start fresh for this run.
            self.draft_model = load_init_draft_model(
                init_checkpoint_dir=str(init_from),
                draft_model=self.draft_model,
                device=self.device,
                precision_dtype=self.precision_dtype,
            )
            print_on_local_main(f"Initialised draft weights from {init_from} (weights only; fresh optimizer and step).")
        elif init_from:
            print_on_local_main(
                f"train.init_from={init_from} ignored: resuming this run from its own checkpoint "
                f"{self.resume_checkpoint_dir}."
            )
        self.model = self.draft_model
        if self.args.train.torch_compile:
            coalesce = self.args.train.get("inductor_coalesce_tiling_analysis")
            if coalesce is not None:
                # torch 2.11 (ROCm image) inductor asserts in tiling_utils.get_pw_red_splits on this model with
                # coalescing analysis on; the config sets this False explicitly instead of a hidden env default.
                import torch._inductor.config as inductor_config

                inductor_config.triton.coalesce_tiling_analysis = bool(coalesce)
                print_on_local_main(f"inductor coalesce_tiling_analysis = {bool(coalesce)}")
            print_on_local_main("Compiling training model with torch.compile...")
            self.model = torch.compile(self.model, dynamic=True)
        self.model = self._wrap_with_fsdp(self.model)

        self.rounds = _resolve_rounds(
            rounds_cfg=self.args.train.get("rounds"),
            max_train_steps=self.args.train.max_train_steps,
        )
        self._last_saved_step = None
        if self.rounds is None:
            # target_cache_path: one cache dir, or a list of chunk dirs read as one cache in list order.
            self.train_dataset = build_cache_dataset(self.args.data.target_cache_path)
            validate_train_cache(
                train_dataset=self.train_dataset,
                draft_model=self.draft_model,
                target_model_name_or_path=self.args.model.target_model_name_or_path,
            )
            self._set_schedule(dataset_size=len(self.train_dataset))
            self.round = _resolve_round(
                round_cfg=self.args.train.get("round"),
                steps_per_epoch=self.steps_per_epoch,
                max_train_steps=self.max_train_steps,
                dataset_size=len(self.train_dataset),
            )
        else:
            assert self.args.train.get("round") is None, "set train.round or train.rounds, not both"
            # Caches arrive while training runs; schedule fields follow the active round (planned sizes until then).
            self.train_dataset = None
            self.round = None
            self._set_schedule(dataset_size=self.rounds[0]["num_samples"])

        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=float(self.args.train.lr),
            total_steps=self.max_train_steps,
            warmup_ratio=float(self.args.train.warmup_ratio),
            weight_decay=float(self.args.train.weight_decay),
        )
        if self.resume_checkpoint_dir is not None:
            resume_state = load_training_state(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                optimizer=self.optimizer,
                global_rank=self.global_rank,
                world_size=self.world_size,
                local_batch_size=int(self.args.train.local_batch_size),
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                micro_batches_per_epoch=self.micro_batches_per_epoch,
            )
            self.next_micro_step = resume_state.next_micro_step
            if self.args.train.get("reschedule"):
                self._reschedule_cosine()
        elif init_from:
            print_on_local_main(f"Training step 0 of a new run from the weights of {init_from}.")
        else:
            print_on_local_main("Training from scratch.")
        if self.round is not None:
            assert self.global_step >= self.round["start_step"], (
                f"Round starts at step {self.round['start_step']} but the latest checkpoint in "
                f"{self.checkpoint_dir_root} is at step {self.global_step}; run the earlier rounds first."
            )
        self.info_board()

    def _reschedule_cosine(self):
        """train.reschedule: after a resume, end the cosine at the NEW train.max_train_steps instead of the restored
        total. The cosine is stepped recursively, so the LR continues from its current value without a jump and
        reaches eta_min exactly at the new total. Warmup must be over."""
        sched = self.optimizer.scheduler
        warmup = int(sched.warmup_epochs)
        new_t_max = int(self.max_train_steps) - warmup
        assert self.global_step >= warmup, f"train.reschedule during warmup (step {self.global_step} < {warmup})"
        assert new_t_max > self.global_step - warmup, (
            f"train.reschedule: new total {self.max_train_steps} is not after the current step {self.global_step}"
        )
        old_t_max = int(sched.after_scheduler.T_max)
        sched.after_scheduler.T_max = new_t_max
        print_on_local_main(
            f"train.reschedule: cosine total {old_t_max + warmup} -> {self.max_train_steps} steps at step "
            f"{self.global_step} (lr now {self.optimizer.get_learning_rate():.3e})"
        )

    def _set_schedule(self, *, dataset_size: int):
        (
            self.gradient_accumulation_steps,
            self.samples_per_epoch,
            self.per_rank_samples_per_epoch,
            self.micro_batches_per_epoch,
            self.steps_per_epoch,
            self.max_train_steps,
            self.args.train.num_train_epochs,
        ) = _compute_training_schedule(
            world_size=self.world_size,
            dataset_size=int(dataset_size),
            local_batch_size=int(self.args.train.local_batch_size),
            global_batch_size=int(self.args.train.global_batch_size),
            num_train_epochs=int(self.args.train.num_train_epochs),
            max_train_steps=self.args.train.max_train_steps,
        )

    def _activate_round(self, index: int, rnd):
        waited = _wait_for_cache_dirs(
            rnd["cache_dirs"], timeout_s=float(self.args.data.get("round_wait_timeout_s", 6 * 3600))
        )
        dist.barrier()
        dataset = build_cache_dataset(rnd["cache_dirs"] if len(rnd["cache_dirs"]) > 1 else rnd["cache_dirs"][0])
        validate_train_cache(
            train_dataset=dataset,
            draft_model=self.draft_model,
            target_model_name_or_path=self.args.model.target_model_name_or_path,
        )
        self.train_dataset = dataset
        self._set_schedule(dataset_size=len(dataset))
        self.round = _resolve_round(
            round_cfg=rnd,
            steps_per_epoch=self.steps_per_epoch,
            max_train_steps=self.max_train_steps,
            dataset_size=len(dataset),
        )
        assert self.global_step >= self.round["start_step"], (
            f"round {index} starts at step {self.round['start_step']} but training is at step {self.global_step}"
        )
        print_on_global_main(
            f"Round {index}/{len(self.rounds)}: steps [{self.round['start_step']}, {self.round['stop_step']}) "
            f"from step {self.global_step}, {len(dataset)} samples in {len(rnd['cache_dirs'])} dir(s), "
            f"sampler_epoch {self.round['sampler_epoch']}, waited {waited:.0f} s for data"
        )

    def _finish_round(self, index: int, rnd):
        self.train_dataset.close()
        self.train_dataset = None
        if is_global_main_process():
            marker_dir = os.path.join(self.checkpoint_dir_root, "rounds")
            ensure_dir(marker_dir)
            marker = os.path.join(marker_dir, f"round-{index:04d}.done")
            with open(marker + ".tmp", "w", encoding="utf-8") as handle:
                json.dump({"round": index, "global_step": self.global_step, "cache_dirs": rnd["cache_dirs"]}, handle)
            os.replace(marker + ".tmp", marker)
        dist.barrier()

    @property
    def global_step(self):
        return self.next_micro_step // self.gradient_accumulation_steps

    def info_board(self):
        print_on_local_main("***** Running training *****")
        if self.train_dataset is not None:
            print_on_local_main(f"  Train dataset size = {len(self.train_dataset)}")
        else:
            print_on_local_main(f"  Rounds = {len(self.rounds)} (caches arrive during training)")
        print_on_local_main(f"  Num train epochs = {self.args.train.num_train_epochs}")
        print_on_local_main(f"  Samples per epoch = {self.samples_per_epoch}")
        print_on_local_main(f"  Local batch size = {self.args.train.local_batch_size}")
        print_on_local_main(f"  Global batch size = {self.args.train.global_batch_size}")
        print_on_local_main(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        print_on_local_main(f"  Steps per epoch = {self.steps_per_epoch}")
        print_on_local_main(f"  Max train steps = {self.max_train_steps}")
        if self.round is not None:
            print_on_local_main(f"  Round = {self.round}")

    def build_models(self):
        model_args = self.args.model

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.target_model_name_or_path,
        )
        target_config = AutoConfig.from_pretrained(
            model_args.target_model_name_or_path,
        )

        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)

        # Training only uses the target checkpoint to initialize frozen draft
        # embeddings and lm_head weights.
        target_model = AutoModelForCausalLM.from_pretrained(
            model_args.target_model_name_or_path,
            dtype=self.precision_dtype,
        ).to(device="cpu").eval()
        target_embed_tokens = target_model.get_input_embeddings()
        target_lm_head = target_model.get_output_embeddings()
        assert (target_lm_head is not None) and (target_embed_tokens is not None)
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )
        del target_model
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        raise NotImplementedError

    def _wrap_with_fsdp(self, model):
        fsdp_kwargs = _build_fsdp_kwargs(
            sharding_strategy_name=self.args.train.sharding_strategy,
            precision_dtype=self.precision_dtype,
            world_size=self.world_size,
        )
        return FSDP(model, **fsdp_kwargs)

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        sampler = StatelessResumableDistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            total_size=self.samples_per_epoch,
            start_global_offset_samples=start_offset_samples,
            num_samples=num_samples,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.args.train.local_batch_size),
            sampler=sampler,
            collate_fn=self.data_collator_cls(),
            num_workers=int(self.args.data.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def run_batch(self, batch):
        raise NotImplementedError

    def _checkpoint_kwargs(self):
        return dict(
            model=self.model,
            draft_model=self.draft_model,
            optimizer=self.optimizer,
            checkpoint_dir_root=self.checkpoint_dir_root,
            train_config=self.args,
            next_micro_step=self.next_micro_step,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            global_rank=self.global_rank,
            world_size=self.world_size,
            local_batch_size=int(self.args.train.local_batch_size),
        )

    def save_and_eval_checkpoint(self):
        checkpoint_dir = save_checkpoint(**self._checkpoint_kwargs())
        self._last_saved_step = self.global_step
        if is_global_main_process():
            _launch_eval(
                target_model_name_or_path=self.args.model.target_model_name_or_path,
                checkpoint_dir=checkpoint_dir,
                step=self.global_step,
                tensorboard_dir=self.args.logging.tensorboard_dir,
                exp_name=self.args.exp_name,
            )
        dist.barrier()
        return checkpoint_dir

    def _save_and_suspend(self):
        print_on_global_main("Saving checkpoint before suspending...")
        save_checkpoint(**self._checkpoint_kwargs())
        dist.barrier()
        if is_global_main_process():
            print_on_global_main("Going to suspend...")
            self.suspend_controller.go_suspend()
        dist.barrier()

    def train(self):
        if self.rounds is None:
            self._train_round()
            return
        for index, rnd in enumerate(self.rounds):
            if self.global_step >= rnd["stop_step"]:
                continue
            self._activate_round(index, rnd)
            if not self._train_round():
                self.train_dataset.close()  # suspended mid-round: no done marker; a relaunch resumes this round
                return
            self._finish_round(index, rnd)

    def _train_round(self):
        """Train up to the stop step; False if suspended before reaching it."""
        self.model.train()
        stop_step = self.max_train_steps if self.round is None else self.round["stop_step"]
        if self.global_step >= stop_step:
            print_on_global_main(f"Nothing to do: global step {self.global_step} >= stop step {stop_step}.")
            return True

        local_batch_size = int(self.args.train.local_batch_size)
        total_micro_steps = stop_step * self.gradient_accumulation_steps
        remaining_micro_steps = total_micro_steps - self.next_micro_step
        remaining_samples = remaining_micro_steps * local_batch_size
        if self.round is None:
            start_offset_samples = self.next_micro_step * local_batch_size
        else:
            # Position inside this round's cache: the round's sampler epoch (its shuffle seed) plus progress since
            # the round started, so a resumed round continues where it stopped.
            round_start_micro_step = self.round["start_step"] * self.gradient_accumulation_steps
            start_offset_samples = (
                self.round["sampler_epoch"] * self.per_rank_samples_per_epoch
                + (self.next_micro_step - round_start_micro_step) * local_batch_size
            )

        dataloader = self._build_train_dataloader(
            start_offset_samples=start_offset_samples,
            num_samples=remaining_samples,
        )
        prefetcher = CUDAPrefetcher(dataloader, self.device)
        training_logger.start_session(global_step=self.global_step)

        with self.suspend_controller.monitoring():
            for batch in prefetcher:
                should_sync = (
                    (self.next_micro_step + 1) % self.gradient_accumulation_steps == 0
                )
                sync_context = nullcontext() if should_sync else self.model.no_sync()
                with sync_context:
                    training_logger.add_tokens(batch["attention_mask"].sum())
                    loss = self.run_batch(batch) / self.gradient_accumulation_steps
                    loss.backward()
                self.next_micro_step += 1

                if not should_sync:
                    continue

                grad_norm = FSDP.clip_grad_norm_(
                    self.model,
                    float(self.args.train.max_grad_norm),
                )
                self.optimizer.step()
                training_logger.on_optimizer_step(
                    global_step=self.global_step,
                    next_micro_step=self.next_micro_step,
                    micro_batches_per_epoch=self.micro_batches_per_epoch,
                    max_train_steps=self.max_train_steps,
                    learning_rate=self.optimizer.get_learning_rate(),
                    grad_norm=grad_norm.item(),
                )

                if self.global_step % int(self.args.logging.checkpointing_steps) == 0:
                    self.save_and_eval_checkpoint()

                if self.suspend_controller.requested():
                    self._save_and_suspend()
                    return False

        del prefetcher, dataloader
        if self._last_saved_step != self.global_step:
            self.save_and_eval_checkpoint()
        return True

    def clean_up(self):
        training_logger.close()
        dist.barrier()
        dist.destroy_process_group()
