"""DFlash drafter for K2-Horizon-375B: Qwen3-style draft layers, frozen K2 embeddings/lm_head read from safetensors.

The 375B target is never loaded. The draft is a Qwen3 decoder stack (the architecture vLLM's DFlashDraftModel runs)
shaped by model.draft_shape; the target only supplies num_hidden_layers (for tap validation) and the two frozen
[vocab, hidden] tensors, read directly from the checkpoint shards named in model.safetensors.index.json.
"""
import json
import os
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3Config

from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer
from deepspec.utils import print_on_local_main

EMBED_KEY = "model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"


def read_checkpoint_tensor(model_dir: str, key: str) -> torch.Tensor:
    """One tensor from a sharded safetensors checkpoint, as bf16; FP8 weights must carry a scale and are rejected."""
    with open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    if key not in weight_map:
        raise KeyError(f"{key} not in {model_dir}/model.safetensors.index.json")
    for scale_key in (key + "_scale", key.replace(".weight", ".weight_scale_inv")):
        if scale_key in weight_map:
            raise ValueError(f"{key} is quantized ({scale_key} present); add an explicit dequantization step first")
    with safe_open(os.path.join(model_dir, weight_map[key]), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key)
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"{key} has dtype {tensor.dtype}; only bf16/fp16/fp32 are supported")
    return tensor.to(torch.bfloat16)


def build_k2_shaped_qwen3_config(target_dir: str, draft_shape) -> Qwen3Config:
    """A Qwen3Config with the target's layer count and vocab, and the draft's layer shape (draft_shape)."""
    with open(os.path.join(target_dir, "config.json"), encoding="utf-8") as handle:
        target = json.load(handle)
    rope = target.get("rope_parameters") or {"rope_type": "default", "rope_theta": target["rope_theta"]}
    return Qwen3Config(
        vocab_size=int(target["vocab_size"]),
        hidden_size=int(target["hidden_size"]),
        num_hidden_layers=int(target["num_hidden_layers"]),
        num_attention_heads=int(draft_shape["num_attention_heads"]),
        num_key_value_heads=int(draft_shape["num_key_value_heads"]),
        head_dim=int(draft_shape["head_dim"]),
        intermediate_size=int(draft_shape["intermediate_size"]),
        hidden_act=str(target.get("hidden_act", "silu")),
        rms_norm_eps=float(target["rms_norm_eps"]),
        max_position_embeddings=int(draft_shape.get("max_position_embeddings", target["max_position_embeddings"])),
        rope_parameters=dict(rope),
        attention_bias=False,
        tie_word_embeddings=False,
        bos_token_id=target.get("bos_token_id"),
        eos_token_id=target.get("eos_token_id"),
    )


class K2DFlashTrainer(Qwen3DSparkTrainer):
    def build_models(self):
        model_args = self.args.model
        target_dir = str(model_args.target_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(target_dir)
        mask_token = tokenizer.convert_ids_to_tokens(int(model_args.mask_token_id))
        expected = model_args.get("mask_token")
        if expected is not None and mask_token != expected:
            raise ValueError(f"mask_token_id {model_args.mask_token_id} is {mask_token!r}, expected {expected!r}")
        target_config = build_k2_shaped_qwen3_config(target_dir, model_args.draft_shape)
        draft_model = self._build_draft_model(target_config=target_config, model_args=model_args)
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)
        embed = read_checkpoint_tensor(target_dir, EMBED_KEY)
        lm_head = read_checkpoint_tensor(target_dir, LM_HEAD_KEY)
        draft_model.initialize_embeddings_and_head(
            embed_tokens=SimpleNamespace(weight=embed),
            lm_head=SimpleNamespace(weight=lm_head),
            freeze=True,
        )
        del embed, lm_head
        trainable = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in draft_model.parameters() if not p.requires_grad)
        print_on_local_main(f"K2 DFlash draft: {trainable / 1e9:.3f}B trainable, {frozen / 1e9:.3f}B frozen; "
                            f"mask token {model_args.mask_token_id} = {mask_token!r}")
        return draft_model, tokenizer
