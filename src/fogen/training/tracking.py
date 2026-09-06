"""MLflow tracking wrapper — keeps logging logic out of the training loop."""

import json
from pathlib import Path


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = str(v)
        else:
            out[key] = v
    return out


class MLflowTracker:
    """Thin wrapper around mlflow that no-ops when disabled."""

    def __init__(self, cfg, out_dir, run_name):
        self._active = False
        self._out = Path(out_dir)
        try:
            import mlflow
            self._mlflow = mlflow

            mlflow.set_experiment(cfg.get("wandb_project", "fogen-phase"))
            self._run = mlflow.start_run(
                run_name=run_name, log_system_metrics=True)
            mlflow.log_params(_flatten({**cfg}))
            self._log_dataset(cfg)
            self._active = True
        except Exception as e:
            print(f"mlflow disabled: {e}")

    def _log_dataset(self, cfg):
        shard_dir = Path(cfg["data"]["shard_dir"])
        manifest_path = shard_dir / "manifest.json"
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
        import numpy as np
        dataset = self._mlflow.data.from_numpy(
            features=np.empty(0),
            source=str(shard_dir),
            name=shard_dir.parent.name,
        )
        self._mlflow.log_input(dataset, context="training")
        self._mlflow.set_tags({
            "dataset.total_tokens": manifest.get("total_tokens"),
            "dataset.n_shards": len(manifest.get("shards", [])),
            "dataset.max_tokens": cfg["data"].get("max_tokens", "all"),
            "dataset.tokenizer": cfg["data"]["tokenizer_dir"],
            "dataset.shard_dir": str(shard_dir),
        })

    def log_step(self, step, rec, execution_metrics=None, guard_record=None,
                 muon_lr=None, adamw_lr=None):
        if not self._active:
            return
        metrics = {
            "train/loss": rec["loss"],
            "train/tok_s": rec["tok_s"],
            "train/step": step,
        }
        if muon_lr is not None:
            metrics["lr/muon"] = muon_lr
        if adamw_lr is not None:
            metrics["lr/adamw"] = adamw_lr
        if execution_metrics is not None:
            metrics.update({f"execution/{k}": v.item()
                            for k, v in execution_metrics.items()})
        if guard_record is not None:
            for k, v in guard_record.items():
                metrics[f"guard/{k}"] = float(v)
        self._mlflow.log_metrics(metrics, step=step)

    def log_probes(self, step, aggs):
        if not self._active:
            return
        probe_metrics = {}
        for a in aggs:
            prefix = f"probe/{a['probe']}/{a['split']}"
            probe_metrics[f"{prefix}/acc"] = a["argmax_acc"]
            probe_metrics[f"{prefix}/logprob_diff"] = a["logprob_diff"]
        self._mlflow.log_metrics(probe_metrics, step=step)

    def log_checkpoint(self, step):
        if not self._active:
            return
        self._mlflow.log_metrics({"train/step": step}, step=step)
        self._mlflow.log_artifact(str(self._out / "config_used.yaml"))
        self._mlflow.log_artifact(str(self._out / "train_log.jsonl"))
        self._mlflow.log_artifact(str(self._out / "probe_log.jsonl"))
        ckpt_path = self._out / "ckpts" / f"step{step:06d}.safetensors"
        if ckpt_path.exists():
            self._mlflow.log_artifact(str(ckpt_path), artifact_path="ckpts")

    def finish(self):
        if not self._active:
            return
        self._mlflow.log_artifact(str(self._out / "train_log.jsonl"))
        self._mlflow.log_artifact(str(self._out / "probe_log.jsonl"))
        self._mlflow.log_artifact(str(self._out / "config_used.yaml"))
        self._mlflow.end_run()


class NoopTracker:
    """Drop-in replacement that does nothing."""

    def log_step(self, *a, **kw): pass
    def log_probes(self, *a, **kw): pass
    def log_checkpoint(self, *a, **kw): pass
    def finish(self): pass
