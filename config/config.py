# --- Modern LLM Configuration (~2B parameters, Gemma/Llama 3 style) ---

# Model architecture
VOCAB_SIZE = 50304          # Number of unique tokens in the vocabulary
CONTEXT_LENGTH = 2048       # Maximum sequence length for the model
N_EMBED = 2560              # Dimension of the embedding space
N_LAYERS = 24               # Number of transformer blocks
N_HEAD = 20                 # Number of query attention heads
N_KV_HEAD = 4               # Number of key/value heads (GQA; must divide N_HEAD)
INTERMEDIATE_SIZE = 7040    # SwiGLU FFN hidden dimension (~2.75 * N_EMBED)
TIE_WEIGHTS = True          # Tie token embeddings with LM head
ROPE_THETA = 10000.0        # Base for RoPE frequency computation
USE_GRADIENT_CHECKPOINTING = True  # Trade compute for memory (essential for 2B+ on single GPU)

# Paths to training and development datasets
TRAIN_PATH = "data/train/pile_train.h5"
DEV_PATH = "data/val/pile_dev.h5"

# Training hyperparameters
T_BATCH_SIZE = 16           # Micro-batch size per step
T_CONTEXT_LENGTH = 512      # Context length for training sequences
T_GRAD_ACCUM = 4            # Gradient accumulation steps (effective batch = 64)
T_TRAIN_STEPS = 100000      # Total number of training steps
T_WARMUP_STEPS = 2000       # Linear warmup steps
T_EVAL_STEPS = 1000         # Evaluate every N steps
T_EVAL_ITERS = 250          # Number of batches per evaluation
T_LR = 3e-4                 # Peak learning rate after warmup
T_MIN_LR = 3e-5             # Minimum learning rate at end of cosine decay
T_MAX_GRAD_NORM = 1.0       # Gradient clipping threshold
T_WEIGHT_DECAY = 0.1        # Weight decay for non-bias / non-norm parameters
T_DTYPE = "fp16"            # Training dtype: "fp16", "bf16", or "fp32"

# Checkpointing
T_SAVE_EVERY = 5000         # Save checkpoint every N steps
T_OUT_PATH = "models/modern_transformer.pt"

# Device configuration
DEVICE = "cuda"

# Store all configurations in a dictionary for easy access and modification
default_config = {
    "vocab_size": VOCAB_SIZE,
    "context_length": CONTEXT_LENGTH,
    "n_embed": N_EMBED,
    "n_layers": N_LAYERS,
    "n_head": N_HEAD,
    "n_kv_head": N_KV_HEAD,
    "intermediate_size": INTERMEDIATE_SIZE,
    "tie_weights": TIE_WEIGHTS,
    "rope_theta": ROPE_THETA,
    "use_gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
    "train_path": TRAIN_PATH,
    "dev_path": DEV_PATH,
    "t_batch_size": T_BATCH_SIZE,
    "t_context_length": T_CONTEXT_LENGTH,
    "t_grad_accum": T_GRAD_ACCUM,
    "t_train_steps": T_TRAIN_STEPS,
    "t_warmup_steps": T_WARMUP_STEPS,
    "t_eval_steps": T_EVAL_STEPS,
    "t_eval_iters": T_EVAL_ITERS,
    "t_lr": T_LR,
    "t_min_lr": T_MIN_LR,
    "t_max_grad_norm": T_MAX_GRAD_NORM,
    "t_weight_decay": T_WEIGHT_DECAY,
    "t_dtype": T_DTYPE,
    "t_save_every": T_SAVE_EVERY,
    "t_out_path": T_OUT_PATH,
    "device": DEVICE,
}
