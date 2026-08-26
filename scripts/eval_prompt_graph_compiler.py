import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fogen.evals.bpb import val_stream
from fogen.execution_graph import compile_graph, fit_composition_scale
from fogen.model import GPT, ModelConfig


def load_model(checkpoint, config, device):
    cfg = yaml.safe_load(open(config))
    model = GPT(ModelConfig(**cfg["model"])).to(device)
    state = {key: value.float() for key, value in load_file(checkpoint).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(key.startswith("rope_") for key in missing)
    return model.eval(), cfg


def mean_token_kl(reference_logits, candidate_logits):
    reference_logprob = F.log_softmax(reference_logits.float(), dim=-1)
    candidate_logprob = F.log_softmax(candidate_logits.float(), dim=-1)
    reference_probability = reference_logprob.exp()
    return float(torch.sum(
        reference_probability * (reference_logprob - candidate_logprob), dim=-1
    ).mean())


def sample_prompts(stream, count, length, rng, device):
    offsets = rng.integers(0, len(stream) - length - 1, size=count)
    return [
        torch.tensor(
            np.asarray(stream[offset:offset + length], dtype=np.int64)[None],
            device=device,
        )
        for offset in offsets
    ]


def random_mask(n_layers, count, rng):
    selected = set(rng.choice(n_layers, size=count, replace=False).tolist())
    return [int(layer in selected) for layer in range(n_layers)]


def logits_for_mask(model, tokens, bits):
    modes = ["parallel" if bit else "sequential" for bit in bits]
    with torch.no_grad(), torch.autocast(
        device_type=tokens.device.type,
        dtype=torch.bfloat16,
        enabled=tokens.device.type == "cuda",
    ):
        return model(tokens, mode=modes).float()


def conformal_scale(predicted, actual, coverage):
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    ratios = actual[predicted > 0] / predicted[predicted > 0]
    rank = min(len(ratios) - 1, int(np.ceil((len(ratios) + 1) * coverage)) - 1)
    return float(np.sort(ratios)[rank])


def calibrate(model, prompts, masks_per_prompt, rng, coverage):
    predicted, actual = [], []
    actual_values = []
    n_layers = len(model.blocks)
    for tokens in prompts:
        with torch.no_grad():
            defects = model.layer_defects(tokens).float().cpu().numpy() ** 2
        reference = logits_for_mask(model, tokens, [0] * n_layers)
        for _ in range(masks_per_prompt):
            count = int(rng.integers(1, n_layers + 1))
            bits = random_mask(n_layers, count, rng)
            kl = mean_token_kl(reference, logits_for_mask(model, tokens, bits))
            predicted.append(float(np.dot(bits, defects)))
            actual.append(kl)
            actual_values.append(kl)
    least_squares = fit_composition_scale(predicted, actual)
    residuals = np.asarray(actual) - least_squares * np.asarray(predicted)
    rank = min(
        len(residuals) - 1,
        int(np.ceil((len(residuals) + 1) * coverage)) - 1)
    return {
        "least_squares": least_squares,
        "conformal_multiplier": conformal_scale(
            predicted, actual, coverage),
        "conformal_slack": float(max(0.0, np.sort(residuals)[rank])),
    }, np.asarray(actual_values)


def policy_slacks(model, prompts, alpha, budgets, coverage):
    n_layers = len(model.blocks)
    residuals = {str(budget): [] for budget in budgets}
    for tokens in prompts:
        with torch.no_grad():
            costs = model.layer_defects(tokens).float().cpu().numpy() ** 2
        reference = logits_for_mask(model, tokens, [0] * n_layers)
        for budget in budgets:
            compiled = compile_graph(costs, np.ones(n_layers), budget, alpha)
            actual = mean_token_kl(
                reference, logits_for_mask(model, tokens, compiled["bits"]))
            residuals[str(budget)].append(
                actual - compiled["predicted_effect"])
    slacks = {}
    for budget, values in residuals.items():
        rank = min(
            len(values) - 1,
            int(np.ceil((len(values) + 1) * coverage)) - 1)
        slacks[budget] = float(max(0.0, np.sort(values)[rank]))
    return slacks


def evaluate(model, prompts, alpha, slacks, budgets, random_repeats, rng):
    n_layers = len(model.blocks)
    trials = {str(budget): [] for budget in budgets}
    for tokens in prompts:
        with torch.no_grad():
            costs = model.layer_defects(tokens).float().cpu().numpy() ** 2
        reference = logits_for_mask(model, tokens, [0] * n_layers)
        for budget in budgets:
            slack = slacks[str(budget)]
            effective_budget = max(0.0, budget - slack)
            compiled = compile_graph(
                costs, np.ones(n_layers), effective_budget, alpha)
            bits = compiled["bits"]
            count = sum(bits)
            compiled_kl = mean_token_kl(
                reference, logits_for_mask(model, tokens, bits))
            random_kls = []
            for _ in range(random_repeats):
                random_bits = random_mask(n_layers, count, rng)
                random_kls.append(mean_token_kl(
                    reference, logits_for_mask(model, tokens, random_bits)))
            trials[str(budget)].append({
                "parallel_layers": count,
                "predicted_kl": compiled["predicted_effect"] + slack,
                "actual_kl": compiled_kl,
                "budget_satisfied": compiled_kl <= budget,
                "random_kl": float(np.mean(random_kls)),
            })
    summaries = {}
    for budget, rows in trials.items():
        summaries[budget] = {
            "prompts": len(rows),
            "coverage": sum(row["budget_satisfied"] for row in rows) / len(rows),
            "mean_parallel_layers": float(np.mean([
                row["parallel_layers"] for row in rows])),
            "mean_compiled_kl": float(np.mean([row["actual_kl"] for row in rows])),
            "mean_random_kl": float(np.mean([row["random_kl"] for row in rows])),
            "compiled_win_rate": sum(
                row["actual_kl"] <= row["random_kl"] for row in rows
            ) / len(rows),
        }
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_ckpt", required=True)
    parser.add_argument("--calibration_config", required=True)
    parser.add_argument("--calibration_shards", required=True)
    parser.add_argument("--test_ckpt", required=True)
    parser.add_argument("--test_config", required=True)
    parser.add_argument("--test_shards", required=True)
    parser.add_argument("--calibration_prompts", type=int, default=32)
    parser.add_argument("--test_prompts", type=int, default=64)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--masks_per_prompt", type=int, default=4)
    parser.add_argument("--random_repeats", type=int, default=4)
    parser.add_argument("--coverage_target", type=float, default=0.9)
    parser.add_argument(
        "--budget_quantiles", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibration_rng = np.random.default_rng(2026)
    test_rng = np.random.default_rng(2027)
    calibration_model, _ = load_model(
        args.calibration_ckpt, args.calibration_config, device)
    calibration_stream = val_stream(args.calibration_shards)
    calibration_prompts = sample_prompts(
        calibration_stream, args.calibration_prompts,
        args.sequence_length, calibration_rng, device)
    split = len(calibration_prompts) // 2
    fit_prompts = calibration_prompts[:split]
    conformal_prompts = calibration_prompts[split:]
    scales, calibration_kls = calibrate(
        calibration_model, fit_prompts,
        args.masks_per_prompt, calibration_rng,
        args.coverage_target)
    alpha = scales["least_squares"]
    budgets = [
        float(np.quantile(calibration_kls, quantile))
        for quantile in args.budget_quantiles
    ]
    slacks = policy_slacks(
        calibration_model, conformal_prompts, alpha,
        budgets, args.coverage_target)
    del calibration_model, calibration_prompts
    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_model, _ = load_model(args.test_ckpt, args.test_config, device)
    test_stream = val_stream(args.test_shards)
    test_prompts = sample_prompts(
        test_stream, args.test_prompts,
        args.sequence_length, test_rng, device)
    result = {
        "composition_scales": scales,
        "policy_slacks": slacks,
        "coverage_target": args.coverage_target,
        "budgets": budgets,
        "summaries": evaluate(
            test_model, test_prompts, alpha, slacks, budgets,
            args.random_repeats, test_rng),
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
