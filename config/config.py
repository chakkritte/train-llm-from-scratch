# --- Small Model Configuration (~18M parameters) ---
# Designed to train well on ~1.7M tokens in 1-2 hours

# Model architecture
VOCAB_SIZE = 50304
CONTEXT_LENGTH = 512
N_EMBED = 256
N_LAYERS = 8
N_HEAD = 8
N_KV_HEAD = 2
INTERMEDIATE_SIZE = 704
TIE_WEIGHTS = True
ROPE_THETA = 10000.0
USE_GRADIENT_CHECKPOINTING = False
USE_MOE = False
MOE_NUM_EXPERTS = 8
MOE_TOP_K = 2

# Paths to datasets
TRAIN_PATH = "data/train/pile_train.h5"
DEV_PATH = "data/val/pile_dev.h5"

# Training hyperparameters
T_BATCH_SIZE = 32
T_CONTEXT_LENGTH = 256
T_GRAD_ACCUM = 4          # Effective batch = 128
T_TRAIN_STEPS = 10000     # ~50 epochs on 1.7M tokens
T_WARMUP_STEPS = 500
T_EVAL_STEPS = 500
T_EVAL_ITERS = 50
T_LR = 5e-4               # Slightly higher LR for small model
T_MIN_LR = 5e-5
T_MAX_GRAD_NORM = 1.0
T_WEIGHT_DECAY = 0.1
T_DTYPE = "bf16"

# Checkpointing
T_SAVE_EVERY = 2000
T_OUT_PATH = "models/small_transformer.pt"

# Device
DEVICE = "cuda"

# Config dict
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
    "use_moe": USE_MOE,
    "moe_num_experts": MOE_NUM_EXPERTS,
    "moe_top_k": MOE_TOP_K,
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
