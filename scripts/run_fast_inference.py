#!/usr/bin/env python3
"""
High-performance inference runner using flat weights + compiled engine.

Concept: bypass PyTorch pickle overhead and use a compiled autoregressive
runtime for maximum token throughput.

Workflow:
  1. Export a checkpoint with export_flat_weights.py
  2. Run this script pointing --flat_dir at the exported directory.
"""
import argparse
import json
import os
import time

import torch
import tiktoken

from config.config import default_config as config
from src.models.transformer import Transformer
from src.models.flat_loader import load_flat_weights
from src.models.fast_inference import FastInferenceEngine


def main():
    parser = argparse.ArgumentParser(
        description="Fast inference with flat weights and compiled engine."
    )
    parser.add_argument(
        "--flat_dir",
        type=str,
        required=True,
        help="Directory containing manifest.json + .bin files",
    )
    parser.add_argument(
        "--input_text", type=str, default="The future of AI is", help="Prompt"
    )
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument(
        "--device", type=str, default=config.get("device", "cuda")
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile (for debugging)",
    )
    args = parser.parse_args()

    # Load architecture config from flat export if available
    config_path = os.path.join(args.flat_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            ckpt_config = json.load(f)
    else:
        ckpt_config = config

    model = Transformer(
        vocab_size=ckpt_config.get("vocab_size", config["vocab_size"]),
        n_embed=ckpt_config.get("n_embed", config["n_embed"]),
        n_layers=ckpt_config.get("n_layers", config["n_layers"]),
        n_head=ckpt_config.get("n_head", config["n_head"]),
        n_kv_head=ckpt_config.get("n_kv_head", config["n_kv_head"]),
        intermediate_size=ckpt_config.get(
            "intermediate_size", config["intermediate_size"]
        ),
        context_length=ckpt_config.get("context_length", config["context_length"]),
        tie_weights=ckpt_config.get("tie_weights", config["tie_weights"]),
        rope_theta=ckpt_config.get("rope_theta", config["rope_theta"]),
    ).to(args.device)

    print("Loading flat weights...")
    load_flat_weights(model, args.flat_dir, device=args.device)
    print("Weights loaded.")

    engine = FastInferenceEngine(
        model,
        max_seq_len=ckpt_config.get("context_length", config["context_length"]),
        compile=not args.no_compile,
        device=args.device,
    )

    enc = tiktoken.get_encoding("r50k_base")
    start_ids = enc.encode_ordinary(args.input_text)
    if not start_ids:
        start_ids = [enc.encode_single_token(" ")]
    context = torch.tensor(start_ids, dtype=torch.long, device=args.device).unsqueeze(
        0
    )

    print(f"\nPrompt: {args.input_text}")
    print("Generating...\n")

    t0 = time.perf_counter()
    generated = engine.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    output_text = enc.decode(generated[0].tolist())
    print(output_text)
    print(f"\n--- Stats ---")
    print(f"Tokens generated: {args.max_new_tokens}")
    print(f"Time:             {elapsed * 1000:.1f} ms")
    print(f"Throughput:       {args.max_new_tokens / elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()
