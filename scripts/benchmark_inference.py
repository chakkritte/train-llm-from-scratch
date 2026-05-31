#!/usr/bin/env python3
"""
Layer-wise benchmarking and correctness verification.

Inspired by the "unit tests + benchmarks per layer" workflow from custom
inference stacks. We benchmark each component independently and verify that
optimised paths (flat weights / compiled model) match the baseline numerically.
"""
import argparse
import time
from typing import Optional

import numpy as np
import torch
import tiktoken

from config.config import default_config as config
from src.models.transformer import Transformer
from src.models.flat_loader import load_flat_weights
from src.models.fast_inference import FastInferenceEngine


def _benchmark_layer(name: str, fn, warmup: int = 3, iters: int = 10) -> float:
    """Benchmark a callable and return median latency in ms."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    median = float(np.median(times))
    print(f"  {name}: {median:.3f} ms")
    return median


def _verify_flat_correctness(
    baseline_model: Transformer,
    flat_dir: str,
    device: str,
) -> None:
    """Ensure flat-loaded model produces identical logits to baseline."""
    test_model = Transformer(
        vocab_size=config["vocab_size"],
        n_embed=config["n_embed"],
        n_layers=config["n_layers"],
        n_head=config["n_head"],
        n_kv_head=config["n_kv_head"],
        intermediate_size=config["intermediate_size"],
        context_length=config["context_length"],
        tie_weights=config["tie_weights"],
        rope_theta=config["rope_theta"],
    ).to(device)
    load_flat_weights(test_model, flat_dir, device=device)
    test_model.eval()

    x = torch.randint(0, config["vocab_size"], (2, 16), device=device)
    with torch.no_grad():
        baseline_logits, _, _ = baseline_model(x)
        test_logits, _, _ = test_model(x)
    max_diff = (baseline_logits - test_logits).abs().max().item()
    assert max_diff < 1e-4, f"Numerical mismatch too large: {max_diff}"
    print(f"  Flat-weight correctness: max diff = {max_diff:.2e} ✅")


def _verify_greedy_match(
    baseline_model: Transformer,
    engine: FastInferenceEngine,
    context: torch.Tensor,
) -> None:
    """Ensure compiled engine matches baseline on greedy generation."""
    with torch.no_grad():
        baseline_gen = baseline_model.generate(
            context, max_new_tokens=10, temperature=0.0
        )
    engine_gen = engine.generate(context, max_new_tokens=10, temperature=0.0)
    match = torch.equal(baseline_gen, engine_gen)
    print(f"  Greedy generation match: {'✅' if match else '❌'}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference components and verify correctness."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--flat_dir",
        type=str,
        default=None,
        help="Flat weight directory for correctness comparison",
    )
    parser.add_argument("--device", type=str, default=config.get("device", "cuda"))
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", config)

    model = Transformer(
        vocab_size=ckpt_config.get("vocab_size", config["vocab_size"]),
        n_embed=ckpt_config.get("n_embed", config["n_embed"]),
        n_layers=ckpt_config.get("n_layers", config["n_layers"]),
        n_head=ckpt_config.get("n_head", config["n_head"]),
        n_kv_head=ckpt_config.get("n_kv_head", config["n_kv_head"]),
        intermediate_size=ckpt_config.get("intermediate_size", config["intermediate_size"]),
        context_length=ckpt_config.get("context_length", config["context_length"]),
        tie_weights=ckpt_config.get("tie_weights", config["tie_weights"]),
        rope_theta=ckpt_config.get("rope_theta", config["rope_theta"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("=" * 50)
    print("Inference Benchmark Suite")
    print("=" * 50)
    print(f"Checkpoint: {args.model_path}")
    print(f"Device:     {device}")
    print(f"Params:     {sum(p.numel() for p in model.parameters()):,}")

    # --- Layer micro-benchmarks ---
    print("\n--- Layer Micro-Benchmarks ---")
    B, T = 1, 128
    dummy_ids = torch.randint(0, model.vocab_size, (B, T), device=device)
    dummy_embed = model.token_embed(dummy_ids)
    dummy_normed = model.norm(dummy_embed)
    freqs = model.rope(T)

    _benchmark_layer("Embedding", lambda: model.token_embed(dummy_ids))
    _benchmark_layer("RMSNorm (final)", lambda: model.norm(dummy_embed))
    _benchmark_layer("LM Head", lambda: model.lm_head(dummy_normed))

    if model.blocks:
        block = model.blocks[0]
        _benchmark_layer("TransformerBlock", lambda: block(dummy_embed, freqs))

    # --- Prefill ---
    print("\n--- Prefill Benchmark ---")
    _benchmark_layer("Prefill (B=1, T=128)", lambda: model(dummy_ids))

    # --- Generation throughput ---
    print("\n--- Generation Throughput ---")
    enc = tiktoken.get_encoding("r50k_base")
    start_ids = enc.encode_ordinary(args.prompt)
    if not start_ids:
        start_ids = [enc.encode_single_token(" ")]
    context = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Baseline generate
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            context,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_k=40,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    baseline_time = time.perf_counter() - t0
    baseline_tps = args.max_new_tokens / baseline_time
    print(f"  Baseline generate:  {baseline_time * 1000:.1f} ms ({baseline_tps:.1f} tok/s)")

    # Fast engine
    engine = FastInferenceEngine(
        model,
        max_seq_len=model.context_length,
        compile=True,
        device=device,
    )
    t0 = time.perf_counter()
    _ = engine.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=0.8,
        top_k=40,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fast_time = time.perf_counter() - t0
    fast_tps = args.max_new_tokens / fast_time
    print(f"  FastEngine generate: {fast_time * 1000:.1f} ms ({fast_tps:.1f} tok/s)")
    print(f"  Speedup: {baseline_time / fast_time:.2f}x")

    # --- Correctness verification ---
    if args.flat_dir:
        print("\n--- Correctness Verification ---")
        _verify_flat_correctness(model, args.flat_dir, device)
        _verify_greedy_match(model, engine, context)

    print("\nDone.")


if __name__ == "__main__":
    main()
