import argparse
import json
from pathlib import Path

import torch

from dsv41_train import DeepSeekV41Config, DeepSeekV41ForCausalLM


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny DeepSeek V4.1 causal LM")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/final")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_batch(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate short, learnable token sequences without a dataset dependency."""

    starts = torch.randint(0, 10, (batch_size, 1), device=device)
    offsets = torch.arange(seq_len - 1, device=device).unsqueeze(0)
    body = 3 + (starts + offsets) % 10
    bos = torch.ones((batch_size, 1), dtype=torch.long, device=device)
    return torch.cat((bos, body), dim=1)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.log_every < 1:
        raise ValueError("steps, batch-size, and log-every must be positive")
    if not 2 <= args.seq_len <= 128:
        raise ValueError("seq-len must be between 2 and 128")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    config = DeepSeekV41Config.tiny()
    model = DeepSeekV41ForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(json.dumps({"device": str(device), "parameters": parameter_count}))
    for step in range(1, args.steps + 1):
        input_ids = make_batch(args.batch_size, args.seq_len, device)
        attention_mask = torch.ones_like(input_ids)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = output.loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss.detach().float().item(),
                        "grad_norm": torch.as_tensor(grad_norm).float().item(),
                    }
                )
            )

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": args.steps,
    }
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
