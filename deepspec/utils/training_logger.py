import time
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from deepspec.utils import ensure_dir, is_global_main_process, print_on_global_main
from deepspec.utils.metrics import add_metric, flush, reset


_writer: Optional[SummaryWriter] = None
_wandb_run = None
_logging_steps: int = 1
_session_start_wall: Optional[float] = None
_session_start_step: int = 0
_window_start_wall: Optional[float] = None
_window_start_step: int = 0
_window_tokens = None


def init(
    *,
    logging_steps: int,
    tensorboard_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> None:
    global _writer, _logging_steps, _wandb_run
    _logging_steps = int(logging_steps)
    if tensorboard_dir is not None and is_global_main_process():
        ensure_dir(tensorboard_dir)
        _writer = SummaryWriter(tensorboard_dir)
    if wandb_project is not None and is_global_main_process():
        # Requested W&B logging must not silently degrade: a missing package or login fails here, before training.
        import wandb

        if not wandb.login(anonymous="never", verify=True):
            raise RuntimeError("W&B logging requested (logging.wandb_project) but this node is not logged in")
        # wandb_id: one W&B run shared by several launches (chunk rounds), resumed in place; steps stay global.
        resume_kwargs = dict(id=str(wandb_id), resume="allow") if wandb_id else {}
        _wandb_run = wandb.init(
            project=wandb_project, entity=wandb_entity, name=wandb_name, config=config, **resume_kwargs
        )


def start_session(*, global_step: int) -> None:
    global _session_start_wall, _session_start_step
    reset()
    _session_start_wall = time.time()
    _session_start_step = int(global_step)
    _start_window(global_step=global_step)


def _start_window(*, global_step: int) -> None:
    global _window_start_wall, _window_start_step, _window_tokens
    _window_start_wall = time.time()
    _window_start_step = int(global_step)
    _window_tokens = None


def add_tokens(num_tokens) -> None:
    """Count this rank's processed sequence tokens (kept on device; read once per logging window)."""
    global _window_tokens
    _window_tokens = num_tokens if _window_tokens is None else _window_tokens + num_tokens


def on_optimizer_step(
    *,
    global_step: int,
    next_micro_step: int,
    micro_batches_per_epoch: int,
    max_train_steps: int,
    learning_rate: float,
    grad_norm: float,
):
    add_metric("lr", learning_rate, reduction="last", tag="train")
    add_metric("grad_norm", grad_norm, reduction="last", tag="train")

    if global_step % _logging_steps != 0:
        return None

    elapsed = time.time() - _window_start_wall
    window_steps = max(int(global_step) - _window_start_step, 1)
    tokens = 0.0 if _window_tokens is None else float(_window_tokens)
    add_metric("tokens_per_s_per_gpu", tokens / elapsed, reduction="dp_mean", tag="perf")
    add_metric("step_time_s", elapsed / window_steps, reduction="dp_max", tag="perf")
    add_metric("max_mem_allocated_gib", torch.cuda.max_memory_allocated() / 2**30, reduction="dp_max", tag="perf")
    add_metric("max_mem_reserved_gib", torch.cuda.max_memory_reserved() / 2**30, reduction="dp_max", tag="perf")
    summary = flush()
    _start_window(global_step=global_step)
    if is_global_main_process():
        _write_scalars(summary, global_step=global_step)
        _print_summary(
            summary=summary,
            global_step=global_step,
            next_micro_step=next_micro_step,
            micro_batches_per_epoch=micro_batches_per_epoch,
            max_train_steps=max_train_steps,
        )
    return summary


def close() -> None:
    global _writer
    global _wandb_run
    if _writer is not None:
        _writer.close()
        _writer = None
    if _wandb_run is not None:
        _wandb_run.finish()
        _wandb_run = None


def _write_scalars(summary, *, global_step: int) -> None:
    _write_wandb(summary, global_step=global_step)
    if _writer is None:
        return
    for key, value in summary.items():
        _writer.add_scalar(key, value, global_step)


def _write_wandb(summary, *, global_step: int) -> None:
    if _wandb_run is None:
        return
    _wandb_run.log(dict(summary), step=global_step)


def _print_summary(
    *,
    summary,
    global_step: int,
    next_micro_step: int,
    micro_batches_per_epoch: int,
    max_train_steps: int,
) -> None:
    session_start_wall = _session_start_wall
    if session_start_wall is None:
        session_start_wall = time.time()
    current_epoch = next_micro_step // micro_batches_per_epoch + 1
    session_elapsed = time.time() - session_start_wall
    completed_session_steps = global_step - _session_start_step
    remaining_steps = max(max_train_steps - global_step, 0)
    remaining_min = (
        session_elapsed * remaining_steps / max(completed_session_steps, 1)
    ) / 60
    loss_text = ""
    if "train/loss" in summary:
        loss_text = f" loss={summary['train/loss']:.4f}"
    print_on_global_main(
        f"epoch={current_epoch} "
        f"step={global_step}/{max_train_steps}"
        f"{loss_text} "
        f"| elapsed={session_elapsed / 60:.1f}min"
        f" | remaining={remaining_min:.1f}min"
    )
