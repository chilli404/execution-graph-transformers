"""Serving latency comparison across execution modes (sequential, parallel,
parallel_fused), following vllm bench latency's own methodology, but with
prompt tokens sampled in-vocab (vllm bench latency hardcodes
np.random.randint(10000, ...), which is out of range for this checkpoint's
8192-token vocab) and decode-only TPOT computed from each request's own
first_token_ts/last_token_ts (RequestStateStats), not a lump-sum that
includes prefill.

Run from the vllm_plugin directory with the vLLM virtualenv:
    uv run python scripts/bench_execution_modes.py --hf-dir <path>
"""

import argparse
import json
import time

import numpy as np


def bench_mode(hf_dir: str, mode: str, batch_size: int, input_len: int,
                output_len: int, num_iters: int, num_iters_warmup: int,
                enforce_eager: bool, attention_backend: str | None) -> dict:
    import fogen.hf_model  # noqa: F401 registers "fogen" with transformers Auto*
    from vllm import LLM, SamplingParams, TokensPrompt

    llm_kwargs = dict(
        model=hf_dir,
        trust_remote_code=True,
        dtype="float32",
        enforce_eager=enforce_eager,
        hf_overrides={"execution_mode": mode},
        disable_log_stats=False,
    )
    if attention_backend:
        llm_kwargs["attention_backend"] = attention_backend
    llm = LLM(**llm_kwargs)
    vocab_size = llm.llm_engine.model_config.hf_config.vocab_size

    sampling = SamplingParams(
        temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=output_len,
        detokenize=False,
    )
    prompt_token_ids = np.random.randint(
        vocab_size, size=(batch_size, input_len)).tolist()
    prompts = [TokensPrompt(prompt_token_ids=p) for p in prompt_token_ids]

    def run_once():
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        wall = time.perf_counter() - start
        decode_tpot_ms = []
        for o in outputs:
            m = o.metrics
            n = m.num_generation_tokens
            if m is not None and n > 1:
                decode_tpot_ms.append(
                    (m.last_token_ts - m.first_token_ts) / (n - 1) * 1000.0)
        return wall, decode_tpot_ms

    for _ in range(num_iters_warmup):
        run_once()

    wall_times = []
    all_decode_tpots = []
    for _ in range(num_iters):
        wall, decode_tpot_ms = run_once()
        wall_times.append(wall)
        all_decode_tpots.extend(decode_tpot_ms)

    wall_times = np.array(wall_times)
    all_decode_tpots = np.array(all_decode_tpots)
    total_output_tokens = batch_size * output_len

    del llm
    return {
        "mode": mode,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "avg_wall_latency_s": float(wall_times.mean()),
        "throughput_toks_per_s": float(total_output_tokens / wall_times.mean()),
        "avg_decode_tpot_ms": float(all_decode_tpots.mean()),
        "std_decode_tpot_ms": float(all_decode_tpots.std()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--num-iters-warmup", type=int, default=2)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = []
    for mode in ["sequential", "parallel", "parallel_fused"]:
        r = bench_mode(
            args.hf_dir, mode, args.batch_size, args.input_len,
            args.output_len, args.num_iters, args.num_iters_warmup,
            args.enforce_eager, args.attention_backend,
        )
        print(f"{mode:16s} avg_decode_tpot={r['avg_decode_tpot_ms']:.4f}ms "
              f"(+/-{r['std_decode_tpot_ms']:.4f}) "
              f"throughput={r['throughput_toks_per_s']:.1f} tok/s")
        results.append(r)

    seq_tpot = results[0]["avg_decode_tpot_ms"]
    for r in results[1:]:
        pct = (seq_tpot - r["avg_decode_tpot_ms"]) / seq_tpot * 100.0
        print(f"{r['mode']} vs sequential decode-TPOT improvement: {pct:.2f}%")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
