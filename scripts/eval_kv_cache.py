import argparse
import json
import time

import torch
import yaml
from safetensors.torch import load_file

from fogen.model import GPT, ModelConfig


def load_model(checkpoint, config, device):
    cfg = yaml.safe_load(open(config))
    model = GPT(ModelConfig(**cfg["model"])).to(device)
    state = {key: value.float() for key, value in load_file(checkpoint).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(key.startswith("rope_") for key in missing)
    return model.eval(), cfg


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def prefill_latency(model, inputs, mode, repeats):
    with torch.no_grad(), torch.autocast(
        device_type=inputs.device.type,
        dtype=torch.bfloat16,
        enabled=inputs.device.type == "cuda",
    ):
        for _ in range(3):
            model.forward_cached(inputs, mode=mode)
        synchronize(inputs.device)
        started = time.perf_counter()
        for _ in range(repeats):
            model.forward_cached(inputs, mode=mode)
        synchronize(inputs.device)
    return (time.perf_counter() - started) / repeats


def decode_latency(model, inputs, mode, new_tokens, repeats):
    durations = []
    with torch.no_grad(), torch.autocast(
        device_type=inputs.device.type,
        dtype=torch.bfloat16,
        enabled=inputs.device.type == "cuda",
    ):
        for _ in range(repeats):
            logits, cache = model.forward_cached(inputs, mode=mode)
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            synchronize(inputs.device)
            started = time.perf_counter()
            for _ in range(new_tokens):
                logits, cache = model.forward_cached(token, cache=cache, mode=mode)
                token = logits[:, -1].argmax(dim=-1, keepdim=True)
            synchronize(inputs.device)
            durations.append(time.perf_counter() - started)
    return sum(durations) / (len(durations) * new_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--prompt_lengths", type=int, nargs="+", default=[128, 512, 896])
    parser.add_argument("--new_tokens", type=int, default=32)
    parser.add_argument("--prefill_repeats", type=int, default=10)
    parser.add_argument("--decode_repeats", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, args.config, device)
    modes = ["sequential", "parallel", "parallel_fused"]
    torch.manual_seed(2026)
    rows = []
    for batch_size in args.batch_sizes:
        for prompt_length in args.prompt_lengths:
            if prompt_length + args.new_tokens > cfg["model"]["ctx_len"]:
                continue
            inputs = torch.randint(
                0, cfg["model"]["vocab_size"],
                (batch_size, prompt_length), device=device)
            for mode in modes:
                rows.append({
                    "batch_size": batch_size,
                    "prompt_length": prompt_length,
                    "mode": mode,
                    "ttft_seconds": prefill_latency(
                        model, inputs, mode, args.prefill_repeats),
                    "tpot_seconds": decode_latency(
                        model, inputs, mode, args.new_tokens, args.decode_repeats),
                })
    result = {
        "checkpoint": args.ckpt,
        "device": str(device),
        "new_tokens": args.new_tokens,
        "rows": rows,
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
