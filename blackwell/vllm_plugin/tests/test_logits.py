"""Correctness gates: compare vLLM plugin output against HF reference.

Run from the vllm_plugin directory with the vLLM virtualenv:
    uv run --python .venv-vllm/bin/python pytest tests/ -v

Requires a HF-exported checkpoint (see scripts/export_hf_checkpoint.py).
"""

import argparse
import json
import sys

import torch


def compare_logits(hf_dir: str, mode: str = "sequential", max_tokens: int = 32):
    """Compare first-token logits between HF model and vLLM."""
    from transformers import AutoModelForCausalLM, AutoConfig

    # Load HF reference
    config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_dir, trust_remote_code=True, torch_dtype=torch.float32
    ).cuda().eval()

    prompt_ids = torch.tensor([[1, 100, 200, 300, 400]], device="cuda")

    with torch.no_grad():
        hf_out = hf_model(prompt_ids, execution_mode=mode)
        hf_logits = hf_out.logits[0, -1].float().cpu()

    del hf_model
    torch.cuda.empty_cache()

    # Load via vLLM
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=hf_dir,
        trust_remote_code=True,
        dtype="float32",
        enforce_eager=True,
    )

    sampling = SamplingParams(max_tokens=1, temperature=0, logprobs=config.vocab_size)
    outputs = llm.generate(prompt_token_ids=[[1, 100, 200, 300, 400]],
                           sampling_params=sampling)

    vllm_logprobs = outputs[0].outputs[0].logprobs[0]
    vllm_logits = torch.zeros(config.vocab_size)
    for token_id, logprob_obj in vllm_logprobs.items():
        vllm_logits[token_id] = logprob_obj.logprob

    # Compare
    max_diff = torch.max(torch.abs(hf_logits - vllm_logits)).item()
    top_hf = torch.argmax(hf_logits).item()
    top_vllm = torch.argmax(vllm_logits).item()

    print(f"Mode: {mode}")
    print(f"  Max logit diff: {max_diff:.6f}")
    print(f"  Top token HF: {top_hf}, vLLM: {top_vllm}")
    print(f"  Match: {top_hf == top_vllm}")

    return {
        "mode": mode,
        "max_logit_diff": max_diff,
        "top_token_match": top_hf == top_vllm,
    }


def test_greedy_generation(hf_dir: str, max_tokens: int = 32):
    """Compare greedy generation between HF and vLLM."""
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_dir, trust_remote_code=True, torch_dtype=torch.float32
    ).cuda().eval()

    prompt = torch.tensor([[1, 100, 200, 300, 400]], device="cuda")
    hf_tokens = hf_model.generate(
        prompt, max_new_tokens=max_tokens, do_sample=False
    )[0].tolist()

    del hf_model
    torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=hf_dir,
        trust_remote_code=True,
        dtype="float32",
        enforce_eager=True,
    )

    sampling = SamplingParams(max_tokens=max_tokens, temperature=0)
    outputs = llm.generate(
        prompt_token_ids=[[1, 100, 200, 300, 400]],
        sampling_params=sampling,
    )
    vllm_tokens = [1, 100, 200, 300, 400] + list(outputs[0].outputs[0].token_ids)

    match_len = 0
    for h, v in zip(hf_tokens, vllm_tokens):
        if h == v:
            match_len += 1
        else:
            break

    print(f"Greedy generation:")
    print(f"  HF tokens:   {hf_tokens[:20]}...")
    print(f"  vLLM tokens: {vllm_tokens[:20]}...")
    print(f"  Match length: {match_len}/{len(hf_tokens)}")

    return {
        "match_length": match_len,
        "total_length": len(hf_tokens),
        "exact_match": hf_tokens == vllm_tokens,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True,
                        help="Path to HF-exported checkpoint")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = []
    for mode in ["sequential", "parallel"]:
        results.append(compare_logits(args.hf_dir, mode))
    results.append(test_greedy_generation(args.hf_dir))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    all_pass = all(r.get("top_token_match", r.get("exact_match", False))
                   for r in results)
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)
