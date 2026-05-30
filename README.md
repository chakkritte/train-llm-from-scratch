# Train LLM From Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

A from-scratch PyTorch implementation of a modern decoder-only Transformer, based on the original work by **[Fareed Khan](https://github.com/FareedKhan-dev/train-llm-from-scratch)**.

This repository has been modernized with architecture improvements inspired by **Gemma / Llama 3** style models, including:

- **Grouped Query Attention (GQA)** — reduces KV-cache memory during inference
- **Rotary Position Embedding (RoPE)** — replaces learned positional embeddings
- **SwiGLU Feed-Forward Network** — gated activation for better expressiveness
- **RMSNorm** — modern normalization without mean subtraction
- **KV-Cache** — efficient autoregressive text generation
- **Gradient Checkpointing & Mixed Precision (FP16/BF16)** — train billion-parameter models on a single GPU

---

## Architecture

| Component | Implementation |
|-----------|----------------|
| Attention | Grouped Query Attention (GQA) with RoPE + FlashAttention (`scaled_dot_product_attention`) |
| FFN | SwiGLU (`gate_proj × up_proj → down_proj`) |
| Normalization | RMSNorm (pre-norm) |
| Embeddings | Token embeddings with optional weight tying to LM head |
| Generation | KV-cache autoregressive sampling with temperature, top-k, and top-p |

---

## Project Structure

```
train-llm-from-scratch/
├── config/
│   └── config.py                 # Model & training hyperparameters
├── src/models/
│   ├── transformer.py            # Main Transformer model
│   ├── attention.py              # GQA attention with RoPE
│   ├── mlp.py                    # SwiGLU feed-forward
│   ├── transformer_block.py      # Single transformer block
│   ├── rmsnorm.py                # RMSNorm layer
│   └── rope.py                   # Rotary Position Embedding
├── data_loader/
│   └── data_loader.py            # HDF5 batch iterator
├── scripts/
│   ├── data_download.py          # Download PILE dataset
│   ├── data_preprocess.py        # Tokenize & save to HDF5
│   ├── train_transformer.py      # Training loop with AMP & resume
│   └── generate_text.py          # Text generation
├── data/                         # Dataset storage
└── models/                       # Saved checkpoints
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter import issues, set the Python path:

```bash
export PYTHONPATH="$PYTHONPATH:."
```

### 2. Download Data

```bash
python scripts/data_download.py --train_max 1
```

### 3. Preprocess Data

```bash
python scripts/data_preprocess.py --max_data 1000
```

### 4. Train

Edit `config/config.py` to adjust model size and training settings, then run:

```bash
python scripts/train_transformer.py
```

Resume from a checkpoint:

```bash
python scripts/train_transformer.py --resume models/modern_transformer_step5000.pt
```

### 5. Generate Text

```bash
python scripts/generate_text.py \
  --model_path models/modern_transformer.pt \
  --input_text "The future of AI is" \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_k 40
```

---

## Configuration

Key parameters in `config/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VOCAB_SIZE` | 50304 | Vocabulary size |
| `CONTEXT_LENGTH` | 2048 | Max sequence length |
| `N_EMBED` | 2560 | Embedding dimension |
| `N_LAYERS` | 24 | Number of transformer blocks |
| `N_HEAD` | 20 | Number of query heads |
| `N_KV_HEAD` | 4 | Number of key/value heads (GQA) |
| `INTERMEDIATE_SIZE` | 7040 | SwiGLU hidden dimension |
| `T_BATCH_SIZE` | 16 | Micro-batch size |
| `T_GRAD_ACCUM` | 4 | Gradient accumulation steps |
| `T_LR` | 3e-4 | Peak learning rate |
| `T_DTYPE` | fp16 | Training dtype: fp16, bf16, or fp32 |

---

## Training Features

- **Mixed Precision Training** — FP16/BF16 with `GradScaler` for memory efficiency
- **Gradient Accumulation** — effective larger batch sizes without OOM
- **Gradient Checkpointing** — trade compute for memory on large models
- **Cosine LR Decay with Warmup** — stable training dynamics
- **Weight Decay Filtering** — no decay on bias, norm, and embedding parameters
- **Automatic Checkpointing** — saves every `N` steps + final model

---

## Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- GPU with sufficient VRAM (see config for memory estimates printed at startup)

---

## License

MIT License

---

## Acknowledgments

- Original implementation and tutorial by **[Fareed Khan](https://github.com/FareedKhan-dev/train-llm-from-scratch)**
- Architecture inspired by [Gemma](https://ai.google.dev/gemma) and [Llama 3](https://ai.meta.com/blog/meta-llama-3/)
- Dataset: [The Pile (Uncopyrighted)](https://huggingface.co/datasets/monology/pile-uncopyrighted) via HuggingFace
