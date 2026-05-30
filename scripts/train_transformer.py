import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from config.config import default_config as config
from src.models.transformer import Transformer
from data_loader.data_loader import get_batch_iterator
from typing import Dict

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Train a modern decoder-only Transformer.")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
parser.add_argument("--compile", action="store_true", default=False, help="Compile model with torch.compile for speed.")
parser.add_argument("--8bit-adam", action="store_true", default=False, help="Use 8-bit AdamW optimizer (bitsandbytes) to fit larger models.")
args = parser.parse_args()

# Map 8bit_adam to simpler variable name
args.use_8bit_adam = getattr(args, "8bit_adam", False)

# --- Tensor Cores: enable TF32 for faster matmuls ---
torch.set_float32_matmul_precision("high")

# --- Device & dtype setup ---
device = config["device"]
dtype_str = config["t_dtype"]
use_amp = dtype_str in ("fp16", "bf16")
amp_dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
scaler = GradScaler("cuda", enabled=(dtype_str == "fp16"))

# --- Model initialization ---
model = Transformer(
    vocab_size=config["vocab_size"],
    n_embed=config["n_embed"],
    n_layers=config["n_layers"],
    n_head=config["n_head"],
    n_kv_head=config["n_kv_head"],
    intermediate_size=config["intermediate_size"],
    context_length=config["context_length"],
    tie_weights=config["tie_weights"],
    rope_theta=config["rope_theta"],
    use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
).to(device)

# torch.compile for 30-80% speedup (PyTorch 2.x)
if args.compile and hasattr(torch, "compile"):
    print("Compiling model with torch.compile (this may take a minute)...")
    model = torch.compile(model, mode="max-autotune")
    print("Model compiled.")

# Print parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Estimate memory for weights + optimizer states
weight_mem = total_params * 2  # FP16 weights
if args.use_8bit_adam:
    # 8-bit Adam: ~1.5x params (quantized states)
    optimizer_mem = total_params * 1.5
    print("Using 8-bit AdamW optimizer (~1.5x param overhead)")
else:
    # Standard FP32 Adam: 2x copy + momentum + variance = 4x
    optimizer_mem = total_params * 4 * 2
print(f"Estimated optimizer memory: ~{optimizer_mem / 1e9:.2f} GB")
print(f"Estimated total minimum memory: ~{(weight_mem + optimizer_mem) / 1e9:.2f} GB")

# --- Optimizer ---
# Do not apply weight decay to bias, norm, and embedding parameters
no_decay = set()
for name, param in model.named_parameters():
    if "bias" in name or "norm" in name or "embed" in name:
        no_decay.add(name)

param_groups = [
    {
        "params": [p for n, p in model.named_parameters() if n not in no_decay],
        "weight_decay": config["t_weight_decay"],
    },
    {
        "params": [p for n, p in model.named_parameters() if n in no_decay],
        "weight_decay": 0.0,
    },
]

if args.use_8bit_adam:
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(param_groups, lr=config["t_lr"], betas=(0.9, 0.95), eps=1e-8)
    except ImportError:
        print("WARNING: bitsandbytes not installed, falling back to standard AdamW")
        optimizer = torch.optim.AdamW(param_groups, lr=config["t_lr"], betas=(0.9, 0.95), eps=1e-8, fused=True)
else:
    # fused=True uses the CUDA kernel for ~10% speedup
    optimizer = torch.optim.AdamW(param_groups, lr=config["t_lr"], betas=(0.9, 0.95), eps=1e-8, fused=True)

# --- Learning rate schedule ---
max_steps = config["t_train_steps"]
warmup_steps = config["t_warmup_steps"]
min_lr = config["t_min_lr"]
max_lr = config["t_lr"]


def get_lr(step: int) -> float:
    """Linear warmup followed by cosine decay to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

# --- State tracking ---
losses = []
start_step = 0
AVG_WINDOW = 64
grad_accum = config["t_grad_accum"]

# --- Resume from checkpoint ---
if args.resume and os.path.exists(args.resume):
    print(f"Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    losses = checkpoint.get("losses", [])
    start_step = checkpoint.get("step", 0)
    if "scaler_state_dict" in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    print(f"Resumed at step {start_step}")

# --- Evaluation helper ---
@torch.no_grad()
def estimate_loss(steps: int) -> Dict[str, float]:
    """Evaluate average loss on train and dev splits."""
    out = {}
    model.eval()
    for split in ["train", "dev"]:
        data_path = config["train_path"] if split == "train" else config["dev_path"]
        batch_iterator_eval = get_batch_iterator(
            data_path,
            config["t_batch_size"],
            config["t_context_length"],
            device=device,
        )
        losses_eval = torch.zeros(steps)
        for k in range(steps):
            try:
                xb, yb = next(batch_iterator_eval)
                with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    _, loss, _ = model(xb, targets=yb)
                losses_eval[k] = loss.item()
            except StopIteration:
                print(f"Warning: Iterator for {split} ended early.")
                break
        out[split] = losses_eval[: k + 1].mean()
    model.train()
    return out


# --- Training loop ---
batch_iterator = get_batch_iterator(
    config["train_path"],
    config["t_batch_size"],
    config["t_context_length"],
    device=device,
)

os.makedirs(os.path.dirname(config["t_out_path"]) or "models", exist_ok=True)

pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps)
for step in pbar:
    try:
        xb, yb = next(batch_iterator)

        with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            _, loss, _ = model(xb, targets=yb)
            loss = loss / grad_accum  # Scale for gradient accumulation

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        losses.append(loss.item() * grad_accum)
        pbar.set_description(f"Train loss: {np.mean(losses[-AVG_WINDOW:]):.4f} | lr: {get_lr(step):.2e}")

        # Weight update only every grad_accum steps
        if (step + 1) % grad_accum == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["t_max_grad_norm"])

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Periodic evaluation
        if step % config["t_eval_steps"] == 0 and step > 0:
            eval_losses = estimate_loss(config["t_eval_iters"])
            train_loss = eval_losses["train"]
            dev_loss = eval_losses["dev"]
            print(f"\nStep: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}, lr: {get_lr(step):.2e}\n")

        # Periodic checkpointing
        if step % config["t_save_every"] == 0 and step > 0:
            ckpt_path = config["t_out_path"].replace(".pt", f"_step{step}.pt")
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "losses": losses,
                "config": config,
            }
            if scaler is not None:
                ckpt["scaler_state_dict"] = scaler.state_dict()
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    except StopIteration:
        print("Training data iterator finished early.")
        break

# --- Final save ---
eval_losses = estimate_loss(200)
train_loss = eval_losses["train"]
dev_loss = eval_losses["dev"]

# Unique path
modified_model_out_path = config["t_out_path"]
save_tries = 0
while os.path.exists(modified_model_out_path):
    save_tries += 1
    base = Path(config["t_out_path"]).stem
    modified_model_out_path = f"models/{base}_{save_tries}.pt"

final_ckpt = {
    "step": start_step + len(losses),
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "losses": losses,
    "train_loss": train_loss,
    "dev_loss": dev_loss,
    "config": config,
}
if scaler is not None:
    final_ckpt["scaler_state_dict"] = scaler.state_dict()

torch.save(final_ckpt, modified_model_out_path)
print(f"Saved final model to {modified_model_out_path}")
print(f"Finished training. Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
