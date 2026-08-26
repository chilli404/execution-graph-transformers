"""Config-driven training with in-loop probe evals and dense checkpointing.

Usage:
  python -m fogen.training.train --config configs/v1_repro.yaml --seed 42 \
      [--out runs/v1_repro_s42] [--no-wandb]

Reproduces the v1 setup: Muon (matrices) + AdamW (embeddings), wd decaying
to 0, cosine warmdown over the final fraction of steps, bf16 autocast,
probe battery scored every probe_every steps (every step for the first 50).
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from fogen.data import ShardedLoader, load_tokenizer
from fogen.evals.scoring import aggregate, fogen_scorer, load_battery
from fogen.model import GPT, ModelConfig
from fogen.training.margin_guard import forced_choice_margin, project_gradient_
from fogen.training.muon import Muon


def active_loader(step, loader_a, loader_b=None, switch_step=None):
    """Windowed exposure (Step-6T amendment 2026-06-11): batches for
    step < switch_step draw from loader_a, step >= switch_step from
    loader_b. Without data.phase_b in the config, loader_a serves every
    step — the pre-amendment behavior, bit-for-bit."""
    if loader_b is not None and step >= switch_step:
        return loader_b
    return loader_a


def lr_scale(step: int, total: int, warmdown_frac: float) -> float:
    start = int(total * (1 - warmdown_frac))
    if step < start:
        return 1.0
    t = (step - start) / max(1, total - start)
    return 0.5 * (1 + math.cos(math.pi * t))


def random_execution_mask(n_layers, parallel_probability, generator):
    parallel = torch.rand(n_layers, generator=generator) < parallel_probability
    return ["parallel" if value else "sequential" for value in parallel.tolist()]


def consistency_weight(step, total, config):
    start_weight = config.get("consistency_weight", 0.1)
    end_weight = config.get("consistency_weight_end", start_weight)
    decay_start = int(total * config.get("consistency_decay_start", 1.0))
    decay_end = int(total * config.get("consistency_decay_end", 1.0))
    if step <= decay_start:
        return start_weight
    if step >= decay_end or decay_end <= decay_start:
        return end_weight
    progress = (step - decay_start) / (decay_end - decay_start)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return end_weight + (start_weight - end_weight) * cosine


def polymorphic_loss(model, inputs, targets, parallel_weight, consistency_weight,
                     teacher_detach=False):
    sequential_logits = model(inputs, mode="sequential")
    parallel_logits = model(inputs, mode="parallel")
    sequential_loss = F.cross_entropy(
        sequential_logits.view(-1, sequential_logits.size(-1)), targets.reshape(-1))
    parallel_loss = F.cross_entropy(
        parallel_logits.view(-1, parallel_logits.size(-1)), targets.reshape(-1))
    sequential_centered = sequential_logits - sequential_logits.mean(dim=-1, keepdim=True)
    parallel_centered = parallel_logits - parallel_logits.mean(dim=-1, keepdim=True)
    consistency_target = sequential_centered.detach() if teacher_detach else sequential_centered
    consistency = F.mse_loss(parallel_centered, consistency_target)
    total = (
        (1 - parallel_weight) * sequential_loss
        + parallel_weight * parallel_loss
        + consistency_weight * consistency
    )
    return total, {
        "sequential_loss": sequential_loss,
        "parallel_loss": parallel_loss,
        "consistency": consistency,
    }


def save_checkpoint(model, out_dir: Path, step: int):
    from safetensors.torch import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.bfloat16() for k, v in model.state_dict().items()
             if not k.startswith("rope_")}
    save_file(state, str(out_dir / f"step{step:06d}.safetensors"))


def checkpoint_steps(cfg: dict, total: int) -> set[int]:
    steps = {0, total}
    every = cfg.get("ckpt_every", 100)
    steps.update(range(0, total + 1, every))
    for lo, hi, dense in cfg.get("dense_windows", []):
        steps.update(range(lo, min(hi, total) + 1, dense))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--init-ckpt")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out or f"runs/{cfg['run_name']}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.dump({**cfg, "seed": args.seed}))

    mcfg = ModelConfig(**cfg["model"])
    model = GPT(mcfg).to(device)
    if args.init_ckpt:
        from safetensors.torch import load_file
        state = {key: value.float() for key, value in load_file(args.init_ckpt).items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected
        assert all(key.startswith("rope_") for key in missing)
    print(f"params: {model.num_params()/1e6:.1f}M  device: {device}")

    tokenizer = load_tokenizer(cfg["data"]["tokenizer_dir"])
    loader = ShardedLoader(cfg["data"]["shard_dir"], cfg["batch_seqs"],
                           mcfg.ctx_len, seed=args.seed, device=device,
                           one_doc_per_seq=cfg["data"].get("one_doc_per_seq", False),
                           mask_padding=cfg["data"].get("mask_padding", False),
                           max_tokens=cfg["data"].get("max_tokens"))
    pb = cfg["data"].get("phase_b")
    loader_b, switch_step = None, None
    if pb:
        # seed+1: independent offset stream for phase B, avoids replaying
        # phase A's offsets on a different shard set
        loader_b = ShardedLoader(pb["shard_dir"], cfg["batch_seqs"],
                                 mcfg.ctx_len, seed=args.seed + 1,
                                 device=device,
                                 one_doc_per_seq=cfg["data"].get("one_doc_per_seq", False),
                                 mask_padding=cfg["data"].get("mask_padding", False),
                                 max_tokens=pb.get("max_tokens"))
        switch_step = int(pb["switch_step"])
    battery = load_battery(cfg["probes"]["battery_path"])
    scorer = fogen_scorer(model, tokenizer, device=device,
                          batch_size=cfg["probes"].get("batch_size", 256))
    guard_cfg = cfg.get("margin_guard", {})
    guard_items = []
    if guard_cfg.get("enabled", False):
        guard_items = [item for item in battery
                       if item["probe"] == guard_cfg["probe"]
                       and item["split"] == guard_cfg.get("split", "train")]
        guard_items = guard_items[:guard_cfg.get("max_items", len(guard_items))]
        if not guard_items:
            raise ValueError("margin_guard selected no probe items")

    t = cfg["train"]
    execution_cfg = cfg.get("execution_training", {})
    execution_generator = torch.Generator().manual_seed(args.seed + 10_000)
    muon = Muon(model.matrix_params(), lr=t["matrix_lr"],
                weight_decay=t["weight_decay"])
    # head_lr (optional): separate unembedding LR, as in v1's released
    # train.py (unembedding_lr 0.004 vs embedding_lr 0.2). Absent -> head
    # stays in the embed group, preserving all pre-2026-06-10 runs.
    if t.get("head_lr") is not None:
        adamw_groups = [dict(params=model.embed_params(exclude_head=True),
                             lr=t["embed_lr"]),
                        dict(params=model.head_params(), lr=t["head_lr"])]
    else:
        adamw_groups = [dict(params=model.embed_params(), lr=t["embed_lr"])]
    if model.scalar_params():
        # ve mixing scalars; v1 paper gives no scalar LR, use matrix LR
        adamw_groups.append(dict(params=model.scalar_params(), lr=t["matrix_lr"]))
    adamw = torch.optim.AdamW(adamw_groups, betas=(0.9, 0.95), weight_decay=0.0)
    for g in adamw.param_groups:
        g["base_lr"] = g["lr"]
    total = t["steps"]
    ckpt_at = checkpoint_steps(cfg.get("checkpointing", {}), total)

    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.get("wandb_project", "fogen-phase"),
                                   name=out.name, config={**cfg, "seed": args.seed})
        except Exception as e:
            print(f"wandb disabled: {e}")

    probe_log = (out / "probe_log.jsonl").open("a")
    train_log = (out / "train_log.jsonl").open("a")

    def run_probes(step: int):
        model.eval()
        rows = scorer.score_items(battery)
        aggs = aggregate(rows)
        for a in aggs:
            probe_log.write(json.dumps({"step": step, **a}) + "\n")
        probe_log.flush()
        if wandb_run:
            wandb_run.log({f"probe/{a['probe']}/{a['split']}/acc": a["argmax_acc"]
                           for a in aggs} | {"step": step}, step=step)
        model.train()

    model.train()
    t0 = time.time()
    for step in range(total + 1):
        s = lr_scale(step, total, t.get("warmdown_frac", 0.3))
        wd = t["weight_decay"] * (1 - step / total)  # decay wd to 0
        for g in muon.param_groups:
            g["lr"], g["weight_decay"] = t["matrix_lr"] * s, wd
        for g in adamw.param_groups:
            g["lr"] = g["base_lr"] * s

        if step in ckpt_at:
            save_checkpoint(model, out / "ckpts", step)
        pe = cfg["probes"]["every"]
        if step <= cfg["probes"].get("dense_until", 50) or step % pe == 0:
            run_probes(step)
        if step == total:
            break

        x, y = active_loader(step, loader, loader_b, switch_step).next_batch()
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=(device != "cpu")):
            if (execution_cfg.get("enabled", False)
                    and execution_cfg.get("strategy") == "random_mask"):
                execution_mask = random_execution_mask(
                    mcfg.n_layer,
                    execution_cfg.get("parallel_probability", 0.5),
                    execution_generator)
                loss = model.loss(x, y, mode=execution_mask)
                execution_metrics = {
                    "parallel_fraction": torch.tensor(
                        execution_mask.count("parallel") / mcfg.n_layer,
                        device=loss.device)
                }
            elif execution_cfg.get("enabled", False):
                current_consistency_weight = consistency_weight(
                    step, total, execution_cfg)
                loss, execution_metrics = polymorphic_loss(
                    model, x, y,
                    execution_cfg.get("parallel_weight", 0.5),
                    current_consistency_weight,
                    execution_cfg.get("teacher_detach", False))
                execution_metrics["consistency_weight"] = torch.tensor(
                    current_consistency_weight, device=loss.device)
            else:
                loss = model.loss(x, y)
                execution_metrics = None
        loss.backward()
        guard_record = None
        if guard_items and step % guard_cfg.get("every", 1) == 0:
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=(device != "cpu")):
                margin = forced_choice_margin(
                    model, lambda text: tokenizer.encode(text).ids,
                    guard_items, device)
            parameters = [parameter for parameter in model.parameters()
                          if parameter.requires_grad]
            margin_gradients = torch.autograd.grad(
                margin, parameters, allow_unused=True)
            if margin.item() < guard_cfg.get("trigger_margin", 0.0):
                guard_record = project_gradient_(
                    parameters, margin_gradients,
                    max_dot=guard_cfg.get("max_dot", 0.0))
            else:
                guard_record = {"projected": False}
            guard_record["margin"] = margin.item()
        muon.step(); adamw.step()
        model.zero_grad(set_to_none=True)

        if step % 20 == 0:
            rec = {"step": step, "loss": loss.item(),
                   "tok_s": cfg["batch_seqs"] * mcfg.ctx_len * max(step, 1)
                            / (time.time() - t0)}
            if guard_record is not None:
                rec.update({f"guard_{key}": value
                            for key, value in guard_record.items()})
            if execution_metrics is not None:
                rec.update({f"execution_{key}": value.item()
                            for key, value in execution_metrics.items()})
            train_log.write(json.dumps(rec) + "\n"); train_log.flush()
            if wandb_run:
                wandb_run.log({"train/loss": rec["loss"]}, step=step)
            print(rec)

    if wandb_run:
        wandb_run.finish()
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
