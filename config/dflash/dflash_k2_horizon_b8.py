# DFlash drafter for K2-Horizon-375B-A23B-FP8 (frozen pilot card 2026-10-14): 5 Qwen3-style layers, block 8,
# mask-slot labels (vLLM dflash method), taps [1, 15, 29, 43, 57], mask token 250100 = reserved_special_token_50.
# Draft layer shape copies K2's dense attention/MLP (48 heads x 128 = hidden 6144, 8 KV heads, dense MLP 16384 as in
# K2's mlp_only_layers 0-2): the same per-layer shape vLLM's DFlash Qwen3 layers run, about 0.39B params per layer.
# Paths are node305's; the cache comes from vLLM hidden states (pipeline/convert_chunk.py), so no last-layer states.
import os

from deepspec.trainer import K2DFlashTrainer

PILOT_DIR = "/data/erland/results/k2-mi210/dflash-pilot-20261014"
K2_FP8 = "/data/erland/models/IFM--K2-Horizon-375B-A23B-FP8/f3d9e54cb096b5567a94feb26d87caa9f700daca"

project_name = "k2-dflash-pilot"
exp_name = "dflash_block8_k2_horizon"
seed = 42

model = dict(
    target_model_name_or_path=K2_FP8,
    draft_shape=dict(num_attention_heads=48, num_key_value_heads=8, head_dim=128, intermediate_size=16384),
    block_size=8,
    num_draft_layers=5,
    target_layer_ids=[1, 15, 29, 43, 57],
    mask_token_id=250100,
    mask_token="reserved_special_token_50",
    label_layout="mask",
    num_anchors=512,
    markov_rank=0,
    confidence_head_alpha=0.0,
    loss_decay_gamma=4.0,
    ce_loss_alpha=1.0,
    l1_loss_alpha=0.0,
)

train = dict(
    trainer_cls=K2DFlashTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=64,
    num_train_epochs=6,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    # Whole-model compile stalls 8-GPU FSDP on ROCm; flex_attention is still compiled by transformers.
    torch_compile=False,
    inductor_coalesce_tiling_analysis=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=500,
    wandb_project="k2-dflash-pilot",
    wandb_name=None,
)

data = dict(
    target_cache_path=None,  # set with --opts data.target_cache_path=... (one dir or a list of chunk dirs)
    chat_template="k2",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    run_name = os.environ.get("K2_RUN_NAME", str(cfg["exp_name"]))
    out_dir = os.environ.get("K2_OUT_DIR", os.path.join(PILOT_DIR, "k2-train"))
    logging_cfg["checkpoint_dir"] = os.path.join(out_dir, "ckpt", run_name)
    logging_cfg["tensorboard_dir"] = os.path.join(out_dir, "tb", run_name)
    if logging_cfg.get("wandb_name") is None:
        logging_cfg["wandb_name"] = run_name
    cfg["logging"] = logging_cfg
    return cfg
