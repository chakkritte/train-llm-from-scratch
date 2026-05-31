#!/usr/bin/env python3
"""
Export a PyTorch checkpoint to flat binary weight files.

Concept (adapted from custom C++/CUDA inference stacks):
Instead of heavy pickle/zip serialization, every tensor is dumped as a raw
binary array accompanied by a JSON metadata manifest. This enables:
  - Zero-overhead loading via memory-mapping
  - Cross-language consumption (C++, Rust, etc.) without PyTorch
  - Smaller on-disk footprint (optional FP16 quantisation)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def export_flat_weights(checkpoint_path: str, out_dir: str, fp16: bool = False) -> None:
    """Load a .pt checkpoint and write each tensor to a raw binary file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    os.makedirs(out_dir, exist_ok=True)

    manifest = {}
    for key, tensor in state_dict.items():
        t = tensor.detach().cpu()
        if fp16 and t.dtype == torch.float32:
            t = t.half()
        arr = t.numpy()
        # Use the exact state-dict key as filename (dots are valid on Linux/macOS)
        fname = f"{key}.bin"
        fpath = os.path.join(out_dir, fname)
        arr.tofile(fpath)
        manifest[key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "file": fname,
            "bytes": int(arr.nbytes),
        }

    # Save manifest + config
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if "config" in ckpt:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(ckpt["config"], f, indent=2)

    total_mb = sum(m["bytes"] for m in manifest.values()) / 1e6
    print(f"Exported {len(manifest)} tensors to {out_dir} ({total_mb:.1f} MB)")
    if fp16:
        print("Weights stored as FP16 (half precision).")


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint to flat binary weights.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--fp16", action="store_true", help="Quantise weights to FP16 on disk")
    args = parser.parse_args()
    export_flat_weights(args.checkpoint, args.out_dir, args.fp16)


if __name__ == "__main__":
    main()
