# K2 pilot stage-1 plumbing: DFlash block 8 with mask-slot labels (vLLM method dflash reads slots 1..7) on Qwen3-4B, tiny PerfectBlend cache, node305 paths.
import os

from deepspec.trainer import Qwen3DSparkTrainer

PILOT_DIR = "/data/erland/results/k2-mi210/dflash-pilot-20261014"
SMOKE_DIR = os.path.join(PILOT_DIR, "train-smoke")
QWEN_3_4B_LOCAL = "/data/erland/results/k2-mi210/dflash-plumbing-20261014/models/Qwen--Qwen3-4B"

project_name = "k2-dflash-pilot"
exp_name = "dflash_block8_qwen3_4b_smoke_masklayout"
seed = 42

model = dict(
    target_model_name_or_path=QWEN_3_4B_LOCAL,
    block_size=8,
    num_draft_layers=5,
    target_layer_ids=[1, 9, 17, 25, 33],
    mask_token_id=151669,
    num_anchors=512,
    markov_rank=0,
    label_layout="mask",
    confidence_head_alpha=0.0,
    loss_decay_gamma=4.0,
    ce_loss_alpha=1.0,
    l1_loss_alpha=0.0,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=8,
    num_train_epochs=8,
    max_train_steps=200,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    # Whole-model torch.compile stalls 8-GPU FSDP on ROCm (one rank recompiles while the others wait in a
    # collective; 2026-10-14). Eager model + transformers-compiled flex_attention: no stall, ~0.7 s/step on 8 GPUs.
    torch_compile=False,
    # ROCm fallback: torch 2.11 inductor asserts (tiling_utils.get_pw_red_splits) with coalescing analysis on.
    inductor_coalesce_tiling_analysis=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=100,
    wandb_project="k2-dflash-pilot",
    wandb_name=None,
)

data = dict(
    target_cache_path=os.path.join(SMOKE_DIR, "target-cache"),
    chat_template="qwen",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    # Run name (W&B name and output subdir) from K2_RUN_NAME, forwarded by ./k2 py; defaults to exp_name.
    run_name = os.environ.get("K2_RUN_NAME", str(cfg["exp_name"]))
    logging_cfg["checkpoint_dir"] = os.path.join(SMOKE_DIR, "ckpt", run_name)
    logging_cfg["tensorboard_dir"] = os.path.join(SMOKE_DIR, "tb", run_name)
    if logging_cfg.get("wandb_name") is None:
        logging_cfg["wandb_name"] = run_name
    cfg["logging"] = logging_cfg
    return cfg
