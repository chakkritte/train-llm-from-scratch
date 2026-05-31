"""
Zero-overhead flat weight loader.

Maps raw binary tensor files directly into torch Tensors via numpy memmap,
avoiding pickle/zip deserialization overhead entirely.

This is the PyTorch-side equivalent of the "flat binary float32 arrays"
loading strategy used in pure C++ inference engines.
"""
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def load_flat_weights(
    model: nn.Module,
    flat_dir: str,
    device: Optional[str] = None,
) -> nn.Module:
    """Load flat binary weights into an existing model instance.

    Args:
        model: The model architecture (already instantiated).
        flat_dir: Directory containing manifest.json and .bin files.
        device: Target device (e.g. 'cuda' or 'cpu'). Defaults to model's device.

    Returns:
        The model with loaded weights (in-place).
    """
    manifest_path = os.path.join(flat_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if device is None:
        device = next(model.parameters()).device

    state_dict = {}
    for key, info in manifest.items():
        fpath = os.path.join(flat_dir, info["file"])
        dtype = np.dtype(info["dtype"])
        shape = tuple(info["shape"])
        # Memory-map the raw binary file (zero-copy until accessed)
        arr = np.memmap(fpath, dtype=dtype, mode="r", shape=shape)
        tensor = torch.from_numpy(arr).to(device)
        state_dict[key] = tensor

    model.load_state_dict(state_dict, strict=True)
    return model
