import torch
import tiktoken
import argparse
from pathlib import Path

from config.config import default_config as config
from src.models.transformer import Transformer


def generate_text(
    model_path: str,
    input_text: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cuda",
) -> str:
    """
    Generate text using a trained modern Transformer with KV-cache and sampling.

    Args:
        model_path (str): Path to the saved model checkpoint.
        input_text (str): Prompt text to seed generation.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature. 0 = greedy decoding.
        top_k (int, optional): Top-k sampling filter.
        top_p (float, optional): Nucleus (top-p) sampling filter.
        device (str): Device to run on.

    Returns:
        str: Generated text including the prompt.
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    ckpt_config = checkpoint.get("config", config)

    # Initialize model from checkpoint config (or fall back to current config)
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
        use_gradient_checkpointing=ckpt_config.get("use_gradient_checkpointing", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)

    # Load tokenizer
    enc = tiktoken.get_encoding("r50k_base")

    start_ids = enc.encode_ordinary(input_text)
    if not start_ids:
        start_ids = [enc.encode_single_token(" ")]
    context = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generation
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    output_text = enc.decode(generated[0].tolist())
    return output_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using a trained modern Transformer.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint.")
    parser.add_argument("--input_text", type=str, default="Hello", help="Prompt text.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p (nucleus) sampling.")
    parser.add_argument("--device", type=str, default=config.get("device", "cuda"), help="Device to run on.")

    args = parser.parse_args()

    output = generate_text(
        args.model_path,
        args.input_text,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.device,
    )
    print(f"Generated text:\n{output}")


if __name__ == "__main__":
    main()
