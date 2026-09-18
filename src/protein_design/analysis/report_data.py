"""Data contract and collection helpers for the final report figures.

The report notebook intentionally imports only the read/validation/statistics
functions in this module.  GPU work is performed by
``scripts/analysis/collect_report_plot_data.py`` through the collection
functions below.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, spearmanr

from protein_design.constants import C05_CDRH3, WT_M22_BINDING_ENRICHMENT

from . import registry
from .novelty import annotate_sequence_membership, build_reference_index

SCHEMA_VERSION = 1
REQUIRED_PREFERENCE_METRICS = (
    "test_loss",
    "test_reward_accuracy",
    "test_reward_margin",
)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else registry.REPO_ROOT / path


def load_report_config(path: str | Path) -> dict[str, Any]:
    config_path = repo_path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_path"] = str(config_path)
    return config


def data_dir(config: dict[str, Any]) -> Path:
    return repo_path(config["output"]["data_dir"])


def artifact_path(config: dict[str, Any], name: str) -> Path:
    return data_dir(config) / name


def parts_dir(config: dict[str, Any]) -> Path:
    """Directory for independently produced, provenance-bearing work units."""
    return data_dir(config) / "parts"


def part_path(config: dict[str, Any], name: str) -> Path:
    return parts_dir(config) / name


def execution_profile(config: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return dict(config["execution"]["profiles"][name])
    except KeyError as exc:
        known = sorted(config.get("execution", {}).get("profiles", {}))
        raise ValueError(f"Unknown report execution profile {name!r}; available: {known}") from exc


def artifact_names(config: dict[str, Any], sections: Iterable[str] | None = None) -> list[str]:
    sections = set(sections or ("functional", "preference", "generation"))
    names: list[str] = []
    if "functional" in sections:
        names.append("functional_metrics.json")
    if "dms_distribution" in sections:
        names.append("dms_binding_enrichment_distributions.json")
    if "preference" in sections:
        names.append("preference_test_metrics.json")
    if "generation" in sections:
        names.append("generation_reference_ed2_m22_test.json")
        names.extend(f"generation_library_{key}.json" for key in config["models"]["generation_models"])
    return names


def _functional_metrics_complete(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    return all(
        "wild_type_cdr_pseudo_perplexity" in metrics.get(model, {})
        and all("auroc_above_wt" in metrics[model].get("datasets", {}).get(dataset, {})
                for dataset in config["datasets"]["functional"])
        for model in config["models"]["order"]
    )


def _functional_part_complete(path: Path, config: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        metrics = load_artifact(path)["metrics"]
        return ("wild_type_cdr_pseudo_perplexity" in metrics
                and all("auroc_above_wt" in metrics.get("datasets", {}).get(dataset, {})
                        for dataset in config["datasets"]["functional"]))
    except (KeyError, OSError, ValueError):
        return False


def missing_artifacts(config: dict[str, Any], sections: Iterable[str] | None = None) -> list[Path]:
    missing = [artifact_path(config, name) for name in artifact_names(config, sections) if not artifact_path(config, name).exists()]
    if "functional" in set(sections or ("functional", "preference", "generation")):
        path = artifact_path(config, "functional_metrics.json")
        if path.exists():
            try:
                complete = _functional_metrics_complete(load_artifact(path).get("models", {}), config)
            except (OSError, ValueError):
                complete = False
            if not complete:
                missing.append(path)
    return list(dict.fromkeys(missing))


def collection_commands(config: dict[str, Any], sections: Iterable[str] | None = None) -> list[tuple[str, list[str]]]:
    """Return exact, manually submitted commands for incomplete report work units."""
    selected = set(sections or ("functional", "preference", "generation"))
    config_arg = "--config conf/analysis/report_plots.yaml"
    commands: list[tuple[str, list[str]]] = []
    functional_path = artifact_path(config, "functional_metrics.json")
    try:
        functional_complete = functional_path.exists() and _functional_metrics_complete(
            load_artifact(functional_path).get("models", {}), config)
    except (OSError, ValueError):
        functional_complete = False
    if "functional" in selected and not functional_complete:
        workers = [f"sbatch bash_scripts/report_plot_data_4090.sbatch {config_arg} --work-unit functional-model --model {model}"
                   for model in config["models"]["order"]
                   if not _functional_part_complete(part_path(config, f"functional_{model}.json"), config)]
        reducer = [] if workers else [f"sbatch bash_scripts/report_plot_data_cpu.sbatch {config_arg} --work-unit functional-reduce"]
        if workers:
            commands.append(("Functional model jobs (independent)", workers))
        if reducer:
            commands.append(("Functional reducer", reducer))
    if "dms_distribution" in selected and not artifact_path(config, "dms_binding_enrichment_distributions.json").exists():
        commands.append(("DMS enrichment distributions", [
            f"sbatch bash_scripts/report_plot_data_cpu.sbatch {config_arg} --work-unit dms-distribution",
        ]))
    if "preference" in selected and not artifact_path(config, "preference_test_metrics.json").exists():
        workers = [f"sbatch bash_scripts/report_plot_data_4090.sbatch {config_arg} --work-unit preference-model --model {model}"
                   for model in config["models"]["preference_models"] if not part_path(config, f"preference_{model}.json").exists()]
        reducer = [] if workers else [f"sbatch bash_scripts/report_plot_data_cpu.sbatch {config_arg} --work-unit preference-reduce"]
        if workers:
            commands.append(("Preference model jobs (independent)", workers))
        if reducer:
            commands.append(("Preference reducer", reducer))
    generation_missing = any(not artifact_path(config, name).exists() for name in artifact_names(config, ["generation"]))
    if "generation" in selected and generation_missing:
        if not part_path(config, "generation_baselines.json").exists():
            commands.append(("Generation baseline", [f"sbatch bash_scripts/report_plot_data_cpu.sbatch {config_arg} --work-unit generation-baselines"]))
        samples = [
            f"sbatch bash_scripts/report_plot_data_a100_80gb.sbatch {config_arg} --work-unit generation-sample --model {model} --sampler {sampler}"
            for model in config["models"]["generation_models"] for sampler in ("gibbs", "stochastic_beam")
            if not part_path(config, f"generation_sample_{model}_{sampler}.json").exists()
        ]
        if samples:
            commands.append(("Generation sampling jobs (independent)", samples))
        native = [
            f"sbatch bash_scripts/report_plot_data_a100_80gb.sbatch {config_arg} --work-unit generation-native-score --model {model}"
            for model in config["models"]["generation_models"]
            if not part_path(config, f"generation_native_scores_{model}.json").exists()
        ]
        if native:
            commands.append(("Native generation scorers (after that model's samples)", native))
        if not part_path(config, "generation_common_scores.json").exists():
            commands.append(("Common ESM2 scorer (after all samples)", [
                f"sbatch bash_scripts/report_plot_data_a100_80gb.sbatch {config_arg} --work-unit generation-common-score",
            ]))
        all_parts = (part_path(config, "generation_baselines.json").exists()
                     and part_path(config, "generation_common_scores.json").exists()
                     and all(part_path(config, f"generation_native_scores_{model}.json").exists()
                             for model in config["models"]["generation_models"]))
        if all_parts:
            commands.append(("Generation reducer", [f"sbatch bash_scripts/report_plot_data_cpu.sbatch {config_arg} --work-unit generation-reduce"]))
    return commands


def fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": registry.git_sha(),
        **payload,
    }
    encoded = json.dumps(_json_value(full_payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Reducers may run immediately after several independent workers.  A
    # replace is atomic on the shared filesystem and prevents half-written JSON
    # from being accepted by a preflight or a notebook kernel.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    temporary.replace(path)


def load_artifact(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported report artifact schema in {path}: {payload.get('schema_version')!r}")
    return payload


def complete_preference_summary(summary: dict[str, Any]) -> bool:
    return all(summary.get(name) is not None and math.isfinite(float(summary[name])) for name in REQUIRED_PREFERENCE_METRICS)


def _resolved_config(checkpoint: Path, config: dict[str, Any], model_key: str, *, batch_size: int | None = None):
    """Load run configuration, or construct the configured canonical fallback."""
    from omegaconf import OmegaConf

    run_config = checkpoint.parent / "resolved_config.yaml"
    if run_config.exists():
        cfg = OmegaConf.load(run_config)
        source = str(run_config)
    else:
        root = registry.REPO_ROOT
        pref = config["preference_evaluation"]
        task_name = pref["per_model"][model_key]["task"]
        task_path = root / pref["task_configs"][task_name]
        cfg = OmegaConf.merge(
            OmegaConf.load(root / "conf/config.yaml"),
            OmegaConf.load(root / pref["data_config"]),
            OmegaConf.load(root / "conf/model/esm2_650m.yaml"),
            OmegaConf.load(task_path),
        )
        source = f"canonical fallback: {config['_path']}"
    pref = config["preference_evaluation"]
    cfg.training.device = str(pref["device"])
    cfg.training.batch_size = int(batch_size or pref["batch_size"])
    cfg.training.num_workers = int(pref["num_workers"])
    cfg.training.pin_memory = False
    cfg.training.persistent_workers = False
    reference_checkpoint = str(pref["per_model"][model_key]["reference_checkpoint"])
    if reference_checkpoint.startswith("facebook/"):
        # Base ESM2 references are constructed by the model preset, not loaded
        # as a local state-dict checkpoint.
        cfg.model.init.source = "huggingface"
        cfg.model.init.checkpoint = None
    else:
        cfg.model.init.source = "checkpoint"
        cfg.model.init.checkpoint = reference_checkpoint
    return cfg, source


def recompute_preference_metrics(config: dict[str, Any], model_key: str, *, batch_size: int | None = None) -> dict[str, Any]:
    """Evaluate a checkpoint on the exact DPO test-pair construction.

    This calls the DPO module's existing pair construction, dataloader and
    epoch routine.  The only new responsibility is selecting the stored policy
    checkpoint and its frozen reference checkpoint.
    """
    from protein_design.dpo import train as dpo_train
    from protein_design.dpo.dataset import build_split_pair_dataframes_from_cfg

    spec = registry.resolve_model(model_key)
    checkpoint = Path(str(spec["checkpoint"]))
    cfg, config_source = _resolved_config(checkpoint, config, model_key, batch_size=batch_size)
    _, _, test_df = build_split_pair_dataframes_from_cfg(cfg)
    task = str(config["preference_evaluation"]["per_model"][model_key]["task"])
    if task == "lora":
        # LoRA checkpoints contain adapters only.  Rebuilding the policy with
        # the recorded init checkpoint before restoring adapters is essential:
        # directly loading an adapter into vanilla ESM2 would lose the Evo base.
        from protein_design.lora_dpo import train as lora_train
        loader = lora_train._build_dataloader(test_df, batch_size=int(cfg.training.batch_size), shuffle=False,
                                              seed=int(cfg.seed), num_workers=int(cfg.training.num_workers),
                                              pin_memory=False, persistent_workers=False, prefetch_factor=None)
        policy = lora_train._build_scorers(cfg, __import__("logging").getLogger(__name__))
        lora_train._load_checkpoint(checkpoint, policy=policy, optimizer=None, scheduler=None)
        reference = lora_train._ReferenceView(policy)
        metrics, _ = lora_train._run_epoch(
            policy=policy, reference=reference, dataloader=loader, loss=str(cfg.training.loss),
            beta=float(cfg.training.beta), temperature=float(cfg.training.temperature), optimizer=None,
            scheduler=None, scheduler_step_on_batch=False, grad_clip_norm=0.0,
            logger=__import__("logging").getLogger(__name__), log_every_n_steps=0, track_metrics=True,
        )
    else:
        loader = dpo_train._build_dataloader(test_df, batch_size=int(cfg.training.batch_size), shuffle=False,
                                             seed=int(cfg.seed), num_workers=int(cfg.training.num_workers),
                                             pin_memory=False, persistent_workers=False, prefetch_factor=None)
        policy, reference = dpo_train._build_scorers(cfg, __import__("logging").getLogger(__name__))
        dpo_train._load_checkpoint(checkpoint, policy=policy, optimizer=None, scheduler=None)
        metrics, _ = dpo_train._run_epoch(
            policy=policy, reference=reference, dataloader=loader, loss=str(cfg.training.loss),
            beta=float(cfg.training.beta), temperature=float(cfg.training.temperature), optimizer=None,
            scheduler=None, scheduler_step_on_batch=False, grad_clip_norm=0.0,
            logger=__import__("logging").getLogger(__name__), log_every_n_steps=0, track_metrics=True,
        )
    del policy, reference
    return {
        "test_loss": float(metrics["loss"]),
        "test_reward_accuracy": float(metrics["reward_accuracy"]),
        "test_reward_margin": float(metrics["reward_margin"]),
        "test_pairs": int(metrics["num_pairs"]),
        "source": "recomputed",
        "evaluation_config_source": config_source,
        "evaluation_config_fingerprint": fingerprint({"cfg": str(cfg), "checkpoint": str(checkpoint)}),
    }


def collect_preference_model(
    config: dict[str, Any], model_key: str, *, profile: str, force: bool = False,
) -> Path:
    """Collect one complete preference tuple; never mix summary and recomputed fields."""
    if model_key not in config["models"]["preference_models"]:
        raise ValueError(f"{model_key!r} is not a configured preference model")
    out = part_path(config, f"preference_{model_key}.json")
    if out.exists() and not force:
        return out
    checkpoint = Path(str(registry.resolve_model(model_key)["checkpoint"]))
    summary_path = checkpoint.parent / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profile_values = execution_profile(config, profile)
    effective_batch_size: int | None = None
    if complete_preference_summary(summary):
        row = {name: float(summary[name]) for name in REQUIRED_PREFERENCE_METRICS}
        row.update({"test_pairs": summary.get("test_pairs"), "source": "summary", "summary_path": str(summary_path)})
    else:
        import torch

        effective_batch_size = int(profile_values["preference_batch_size"])
        while True:
            try:
                row = recompute_preference_metrics(config, model_key, batch_size=effective_batch_size)
                break
            except torch.cuda.OutOfMemoryError:
                if effective_batch_size <= 1:
                    raise
                torch.cuda.empty_cache()
                effective_batch_size = max(1, effective_batch_size // int(config["execution"]["oom_backoff_factor"]))
                print(f"[oom] retrying preference evaluation with batch_size={effective_batch_size}")
    row["checkpoint"] = str(checkpoint)
    row["checkpoint_fingerprint"] = fingerprint({"checkpoint": str(checkpoint), "model": model_key})
    write_artifact(out, {"model": model_key, "metrics": row, "execution_profile": profile,
                         "execution": {"preference_batch_size": effective_batch_size or profile_values["preference_batch_size"]},
                         "config_fingerprint": fingerprint(config)})
    return out


def reduce_preference_metrics(config: dict[str, Any]) -> Path:
    out = artifact_path(config, "preference_test_metrics.json")
    rows: dict[str, Any] = {}
    missing: list[Path] = []
    for model_key in config["models"]["preference_models"]:
        path = part_path(config, f"preference_{model_key}.json")
        if not path.exists():
            missing.append(path)
        else:
            rows[model_key] = load_artifact(path)["metrics"]
    if missing:
        raise FileNotFoundError("Preference reducer requires: " + ", ".join(map(str, missing)))
    write_artifact(out, {"models": rows, "config_fingerprint": fingerprint(config)})
    return out


def collect_preference_metrics(config: dict[str, Any], *, force: bool = False) -> Path:
    """Compatibility wrapper for a single-machine collection invocation."""
    for model_key in config["models"]["preference_models"]:
        collect_preference_model(config, model_key, profile="rtx_4090", force=force)
    return reduce_preference_metrics(config)


def _run(command: list[str]) -> None:
    print("[run]", " ".join(command))
    started = time.perf_counter()
    subprocess.run(command, cwd=registry.REPO_ROOT, check=True)
    print(f"[run] completed in {time.perf_counter() - started:.1f}s")


def _auroc_above_threshold(scores: np.ndarray, enrichment: np.ndarray, threshold: float) -> tuple[float, int, int]:
    """Tie-aware AUROC for classifying variants above a fixed enrichment threshold."""
    valid = np.isfinite(scores) & np.isfinite(enrichment)
    clean_scores, clean_enrichment = scores[valid], enrichment[valid]
    labels = clean_enrichment > threshold
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    if not n_positive or not n_negative:
        return float("nan"), n_positive, n_negative
    ranks = rankdata(clean_scores)
    auroc = (ranks[labels].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    return float(auroc), n_positive, n_negative


def _wild_type_pseudo_perplexity(model_key: str, *, batch_size: int, min_batch_size: int) -> float:
    """Score the fixed C05 wild-type CDR-H3 with the same masked-PLL protocol."""
    result = subprocess.run(
        [sys.executable, "scripts/analysis/compute_pll.py", "--model", model_key,
         "--sequence", C05_CDRH3, "--device", "cuda", "--batch-size", str(batch_size),
         "--min-batch-size", str(min_batch_size)],
        cwd=registry.REPO_ROOT, check=True, capture_output=True, text=True,
    )
    return float(json.loads(result.stdout.strip().splitlines()[-1])["pseudo_perplexity"])


def collect_functional_model(
    config: dict[str, Any], model_key: str, *, profile: str, force: bool = False,
) -> Path:
    """Evaluate all functional datasets with one loaded model."""
    if model_key not in config["models"]["order"]:
        raise ValueError(f"{model_key!r} is not a configured report model")
    out = part_path(config, f"functional_{model_key}.json")
    if out.exists() and not force:
        cached = load_artifact(out).get("metrics", {})
        if ("wild_type_cdr_pseudo_perplexity" in cached
                and all("auroc_above_wt" in cached.get("datasets", {}).get(key, {}) for key in config["datasets"]["functional"])):
            return out
    datasets = config["datasets"]["functional"]
    batch_size = int(execution_profile(config, profile)["pll_batch_size"])
    _run([sys.executable, "scripts/analysis/compute_pll.py", "--model", model_key,
          "--dataset", ",".join(datasets), "--device", "cuda", "--batch-size", str(batch_size),
          "--min-batch-size", str(config["execution"]["min_pll_batch_size"])] +
         (["--force"] if force else []))
    per_dataset: dict[str, Any] = {}
    for dataset_key in datasets:
        ds = registry.load_datasets_cfg()["datasets"][dataset_key]
        joined = registry.load_pll(model_key, dataset_key).merge(registry.load_truth(dataset_key), on=ds["seq_col"], how="inner")
        pll = joined["pll"].to_numpy(float)
        truth = joined["enrichment"].to_numpy(float)
        valid = np.isfinite(pll) & np.isfinite(truth)
        rho = float(spearmanr(pll[valid], truth[valid]).statistic) if valid.sum() >= 2 else float("nan")
        cdr_ppl = float(math.exp(-pll[valid].sum() / sum(len(s) for s in joined.loc[valid, ds["seq_col"]])))
        auroc, n_positive, n_negative = _auroc_above_threshold(pll, truth, WT_M22_BINDING_ENRICHMENT)
        per_dataset[dataset_key] = {"spearman_pll_enrichment": rho, "auroc_above_wt": auroc, "n": int(valid.sum()),
                                    "n_positive_above_wt": n_positive, "n_negative_at_or_below_wt": n_negative,
                                    "cdr_pseudo_perplexity": cdr_ppl}
    wt_ppl = _wild_type_pseudo_perplexity(model_key, batch_size=batch_size,
                                          min_batch_size=int(config["execution"]["min_pll_batch_size"]))
    write_artifact(out, {"model": model_key, "metrics": {"checkpoint": registry.resolve_model(model_key)["checkpoint"],
                                                             "wild_type_cdr_pseudo_perplexity": wt_ppl,
                                                             "datasets": per_dataset},
                         "execution_profile": profile, "execution": {"pll_batch_size": batch_size},
                         "config_fingerprint": fingerprint(config)})
    return out


def reduce_functional_metrics(config: dict[str, Any]) -> Path:
    out = artifact_path(config, "functional_metrics.json")
    rows: dict[str, Any] = {}
    missing: list[Path] = []
    for model_key in config["models"]["order"]:
        path = part_path(config, f"functional_{model_key}.json")
        if path.exists():
            rows[model_key] = load_artifact(path)["metrics"]
        else:
            missing.append(path)
    if missing:
        raise FileNotFoundError("Functional reducer requires: " + ", ".join(map(str, missing)))
    write_artifact(out, {"models": rows, "datasets": config["datasets"]["functional"],
                         "config_fingerprint": fingerprint(config)})
    return out


def collect_functional_metrics(config: dict[str, Any], *, force: bool = False) -> Path:
    for model_key in config["models"]["order"]:
        collect_functional_model(config, model_key, profile="rtx_4090", force=force)
    return reduce_functional_metrics(config)


def build_generation_reference(config: dict[str, Any], *, force: bool = False) -> Path:
    out = artifact_path(config, "generation_reference_ed2_m22_test.json")
    if out.exists() and not force:
        return out
    key = config["datasets"]["generation_reference"]
    ds = registry.load_datasets_cfg()["datasets"][key]
    frame = pd.read_csv(ds["path"])[[ds["seq_col"], ds["enrichment_col"]]].drop_duplicates(ds["seq_col"])
    frame = frame.rename(columns={ds["seq_col"]: "sequence", ds["enrichment_col"]: "enrichment"})
    frame = frame[frame["sequence"].astype(str).str.len() == len(C05_CDRH3)].copy()
    positives = frame[frame["enrichment"] > WT_M22_BINDING_ENRICHMENT]
    write_artifact(out, {"dataset": key, "source_path": ds["path"], "wild_type": C05_CDRH3,
                         "wild_type_enrichment": WT_M22_BINDING_ENRICHMENT,
                         "sequences": frame.to_dict("records"),
                         "positive_sequences": positives["sequence"].astype(str).tolist()})
    return out


def _hamming(a: str, b: str) -> int:
    return sum(left != right for left, right in zip(a, b))


def mean_pairwise_hamming(sequences: list[str], *, seed: int = 42, max_n: int = 1000) -> float:
    if len(sequences) < 2:
        return 0.0
    if len(sequences) > max_n:
        sequences = [sequences[i] for i in np.random.default_rng(seed).choice(len(sequences), max_n, replace=False)]
    chars = np.asarray([list(sequence) for sequence in sequences])
    distances = chars.shape[1] - (chars[:, None, :] == chars[None, :, :]).sum(axis=2)
    return float(distances[np.triu_indices(len(chars), k=1)].mean())


def nearest_dms_proxy(sequences: list[str], reference: list[dict[str, Any]]) -> list[float]:
    estimates: list[float] = []
    for sequence in sequences:
        distances = np.asarray([_hamming(sequence, str(row["sequence"])) for row in reference])
        nearest = distances == distances.min()
        estimates.append(float(np.mean([float(reference[i]["enrichment"]) for i in np.flatnonzero(nearest)])))
    return estimates


def library_statistics(rows: list[dict[str, Any]], *, evaluator: str, reference: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    score_key = "native_pll" if evaluator == "native" else "common_esm2_pll"
    ordered = sorted(rows, key=lambda row: float(row[score_key]), reverse=True)
    chosen = ordered[:min(top_k, len(ordered))]
    scores = np.asarray([float(row[score_key]) for row in rows])
    wt_score = next(float(row["wt_" + score_key]) for row in rows if "wt_" + score_key in row)
    proxy = nearest_dms_proxy([str(row["sequence"]) for row in chosen], reference)
    return {
        "evaluator": evaluator,
        "top_k": len(chosen),
        "fraction_above_wt": float(np.mean(scores > wt_score)),
        "mean_top_k_pll": float(np.mean([float(row[score_key]) for row in chosen])),
        "mean_pairwise_hamming": mean_pairwise_hamming([str(row["sequence"]) for row in rows]),
        "median_mutation_count": float(np.median([float(row["n_mutations"]) for row in rows])),
        "novelty": float(np.mean([not bool(row["present_in_training_reference"]) for row in rows])),
        "top_k_nearest_dms_enrichment": float(np.mean(proxy)),
        "top_sequences": [str(row["sequence"]) for row in chosen],
    }


def _generation_work_dir() -> Path:
    root = Path(os.environ.get("SCRATCH_DIR", f"/cluster/scratch/{os.environ.get('USER', 'unknown')}/protein-design"))
    path = root / "report_plot_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _generate_baselines(config: dict[str, Any], work_dir: Path, *, force: bool) -> dict[str, Path]:
    generation = config["generation"]
    output: dict[str, Path] = {}
    commands = {
        "random": [sys.executable, "scripts/random_sampling.py", "--dataset-key", "ed2_m22",
                   "--dms-config", "conf/data/dms/default.yaml", "--enrichment-threshold", str(WT_M22_BINDING_ENRICHMENT),
                   "--trust-radius", str(generation["max_mutations"]), "--n-sequences", str(generation["n_sequences"]),
                   "--seed", str(generation["seed"])],
        "pssm": [sys.executable, "scripts/pssm_sampling.py", "--dataset-key", "ed2_m22",
                 "--dms-config", "conf/data/dms/default.yaml", "--enrichment-threshold", str(WT_M22_BINDING_ENRICHMENT),
                 "--temperature", str(generation["temperature"]), "--n-sequences", str(generation["n_sequences"]),
                 "--seed", str(generation["seed"])],
    }
    for sampler, command in commands.items():
        path = work_dir / f"{sampler}.csv"
        if force or not path.exists():
            _run([*command, "--output-path", str(path)])
        output[sampler] = path
    return output


def _generate_model_library(
    config: dict[str, Any], model_key: str, sampler: str, work_dir: Path, *, profile: str, force: bool,
) -> Path:
    generation = config["generation"]
    model = registry.resolve_model(model_key)
    # ``checkpoint: null`` denotes a vanilla model in the registry.  The
    # samplers require a concrete model reference, so use its declared base
    # model rather than stringifying the null value to the invalid HF ID
    # ``"None"``.
    checkpoint = str(model["checkpoint"] or model["base_model"])
    adapter_args = (["--adapter-base-checkpoint", str(model["adapter_base_checkpoint"])]
                    if model.get("adapter_base_checkpoint") else [])
    specs = {
        "gibbs": [sys.executable, "scripts/gibbs_sampling.py", "--model-variant", model_key,
                  "--checkpoint-path", checkpoint, *adapter_args, "--n-chains", str(generation["chains_or_beams"]),
                  "--n-steps", str(max(generation["retained_steps"])), "--snapshot-every", "1",
                  "--temperature", str(generation["temperature"]), "--seed", str(generation["seed"]),
                  "--max-mutations", str(generation["max_mutations"]), "--start-mode", "wt",
                  "--chain-batch-size", str(execution_profile(config, profile)["gibbs_chain_batch_size"])],
        "stochastic_beam": [sys.executable, "scripts/stochastic_beam_search.py", "--model-variant", model_key,
                            "--checkpoint-path", checkpoint, *adapter_args, "--beam-size", str(generation["chains_or_beams"]),
                            "--n-steps", str(max(generation["retained_steps"])), "--snapshot-every", "1",
                            "--temperature", str(generation["temperature"]), "--seed", str(generation["seed"]), "--start-mode", "wt",
                            "--template-batch-size", str(execution_profile(config, profile)["beam_template_batch_size"])],
    }
    if sampler not in specs:
        raise ValueError(f"Unsupported report sampler {sampler!r}")
    path = work_dir / f"{model_key}_{sampler}.csv"
    if force or not path.exists():
        _run([*specs[sampler], "--output-path", str(path)])
    return path


def _load_library(path: Path, retained_steps: set[int] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if retained_steps is not None:
        frame = frame[frame["gibbs_step"].isin(retained_steps)].copy()
    frame = frame[frame["cdrh3"].astype(str).str.len() == len(C05_CDRH3)].copy()
    if frame.empty:
        raise ValueError(f"No valid CDR-H3 sequences in {path}")
    return frame[["cdrh3", "n_mutations"]].rename(columns={"cdrh3": "sequence"}).reset_index(drop=True)


def _score_libraries_once(
    config: dict[str, Any], libraries: dict[str, Path], model_key: str, *, cache_namespace: str, profile: str, force: bool,
) -> dict[str, dict[str, Any]]:
    """Write a scoring manifest and reuse one evaluator-model process for all libraries."""
    model = registry.resolve_model(model_key)
    work_dir = _generation_work_dir()
    batch_size = int(execution_profile(config, profile)["pll_batch_size"])
    manifest: dict[str, dict[str, str]] = {}
    inputs: dict[str, tuple[Path, Path]] = {}
    for label, csv_path in libraries.items():
        retained = set(int(step) for step in config["generation"]["retained_steps"]) if label.endswith("gibbs") or label.endswith("stochastic_beam") or label in {"gibbs", "stochastic_beam"} else None
        frame = _load_library(csv_path, retained)
        if len(frame) != int(config["generation"]["n_sequences"]):
            raise ValueError(f"{csv_path} has {len(frame)} rows, expected {config['generation']['n_sequences']}")
        input_path = work_dir / f"score_input_{cache_namespace}_{model_key}_{csv_path.stem}.csv"
        output_path = work_dir / f"scores_{cache_namespace}_{model_key}_{csv_path.stem}.csv"
        pd.concat([frame[["sequence"]], pd.DataFrame({"sequence": [C05_CDRH3]})], ignore_index=True).to_csv(input_path, index=False)
        manifest[label] = {"input_csv": str(input_path), "output_csv": str(output_path), "seq_col": "sequence"}
        inputs[label] = (csv_path, output_path)
    manifest_path = work_dir / f"score_manifest_{cache_namespace}_{model_key}_{fingerprint(manifest)}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [sys.executable, "scripts/analysis/score_generated_with_pll.py", "--manifest", str(manifest_path),
               "--base-model", str(model["base_model"]), "--device", "cuda", "--batch-size", str(batch_size),
               "--min-batch-size", str(config["execution"]["min_pll_batch_size"])]
    if model["checkpoint"]:
        command.extend(["--checkpoint", str(model["checkpoint"])])
    if force:
        command.append("--force")
    _run(command)
    values: dict[str, dict[str, Any]] = {}
    for label, (csv_path, output_path) in inputs.items():
        scores = pd.read_csv(output_path)
        score_map = dict(zip(scores["sequence"].astype(str), scores["pll"].astype(float)))
        if C05_CDRH3 not in score_map:
            raise ValueError(f"WT score absent from {output_path}")
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        effective = batch_size
        if meta_path.exists():
            effective = int(json.loads(meta_path.read_text(encoding="utf-8")).get("effective_batch_size", batch_size))
        values[label] = {"csv_path": str(csv_path), "scores": score_map,
                         "wt_score": float(score_map[C05_CDRH3]), "effective_pll_batch_size": effective}
    return values


def collect_generation_baselines(config: dict[str, Any], *, force: bool = False) -> Path:
    """Create the shared random/PSSM inputs once, without a GPU allocation."""
    out = part_path(config, "generation_baselines.json")
    if out.exists() and not force:
        return out
    reference_path = build_generation_reference(config, force=force)
    files = _generate_baselines(config, _generation_work_dir(), force=force)
    write_artifact(out, {"reference_artifact": str(reference_path),
                         "libraries": {key: str(value) for key, value in files.items()},
                         "config_fingerprint": fingerprint(config)})
    return out


def collect_dms_enrichment_distributions(config: dict[str, Any], *, force: bool = False) -> Path:
    """Store the finite held-out enrichment values used by the DMS distribution figure."""
    out = artifact_path(config, "dms_binding_enrichment_distributions.json")
    if out.exists() and not force:
        return out
    datasets: dict[str, Any] = {}
    for dataset_key in config["datasets"]["functional"]:
        spec = registry.load_datasets_cfg()["datasets"][dataset_key]
        values = pd.read_csv(spec["path"])[spec["enrichment_col"]].to_numpy(float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"{dataset_key} contains no finite {spec['enrichment_col']} values")
        datasets[dataset_key] = {"source_path": str(spec["path"]), "enrichment_column": spec["enrichment_col"],
                                 "n": int(len(values)), "values": values.tolist()}
    write_artifact(out, {"datasets": datasets, "wild_type_enrichment": WT_M22_BINDING_ENRICHMENT,
                         "config_fingerprint": fingerprint(config)})
    return out


def collect_generation_sample(
    config: dict[str, Any], model_key: str, sampler: str, *, profile: str, force: bool = False,
) -> Path:
    """Generate one model/sampler library, allowing all eight runs to fan out."""
    if model_key not in config["models"]["generation_models"]:
        raise ValueError(f"{model_key!r} is not a configured generation model")
    if sampler not in {"gibbs", "stochastic_beam"}:
        raise ValueError("Generation sample work units must be gibbs or stochastic_beam")
    out = part_path(config, f"generation_sample_{model_key}_{sampler}.json")
    if out.exists() and not force:
        return out
    csv_path = _generate_model_library(
        config, model_key, sampler, _generation_work_dir(), profile=profile, force=force,
    )
    write_artifact(out, {"model": model_key, "sampler": sampler, "csv_path": str(csv_path),
                         "execution_profile": profile,
                         "execution": {"chain_batch_size": execution_profile(config, profile).get("gibbs_chain_batch_size"),
                                       "template_batch_size": execution_profile(config, profile).get("beam_template_batch_size")},
                         "config_fingerprint": fingerprint(config)})
    return out


def _required_part(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prerequisite work unit: {path}")
    return load_artifact(path)


def _generation_sources(config: dict[str, Any], model_key: str) -> dict[str, Path]:
    base = _required_part(part_path(config, "generation_baselines.json"))
    sources = {key: Path(value) for key, value in base["libraries"].items()}
    for sampler in ("gibbs", "stochastic_beam"):
        sample = _required_part(part_path(config, f"generation_sample_{model_key}_{sampler}.json"))
        sources[sampler] = Path(sample["csv_path"])
    return sources


def collect_generation_native_scores(
    config: dict[str, Any], model_key: str, *, profile: str, force: bool = False,
) -> Path:
    """Score all four libraries with one native evaluator model load per job."""
    out = part_path(config, f"generation_native_scores_{model_key}.json")
    if out.exists() and not force:
        return out
    values = _score_libraries_once(config, _generation_sources(config, model_key), model_key,
                                   cache_namespace="native", profile=profile, force=force)
    write_artifact(out, {"model": model_key, "scores": values, "execution_profile": profile,
                         "execution": {"pll_batch_size": execution_profile(config, profile)["pll_batch_size"]},
                         "config_fingerprint": fingerprint(config)})
    return out


def _common_library_key(model_key: str, sampler: str) -> str:
    return sampler if sampler in {"random", "pssm"} else f"{model_key}_{sampler}"


def collect_generation_common_scores(config: dict[str, Any], *, profile: str, force: bool = False) -> Path:
    """Score each unique library exactly once with vanilla ESM2."""
    out = part_path(config, "generation_common_scores.json")
    if out.exists() and not force:
        return out
    sources: dict[str, Path] = {}
    for model_key in config["models"]["generation_models"]:
        for sampler, path in _generation_sources(config, model_key).items():
            sources.setdefault(_common_library_key(model_key, sampler), path)
    common_model = str(config["generation"]["common_evaluator"])
    values = _score_libraries_once(config, sources, common_model,
                                   cache_namespace="common", profile=profile, force=force)
    write_artifact(out, {"model": common_model, "scores": values, "execution_profile": profile,
                         "execution": {"pll_batch_size": execution_profile(config, profile)["pll_batch_size"]},
                         "config_fingerprint": fingerprint(config)})
    return out


def reduce_generation_libraries(config: dict[str, Any]) -> list[Path]:
    reference = load_artifact(_required_part(part_path(config, "generation_baselines.json"))["reference_artifact"])["sequences"]
    common = _required_part(part_path(config, "generation_common_scores.json"))["scores"]
    generation = config["generation"]
    novelty_index = build_reference_index(registry.REPO_ROOT, splits=set(generation["novelty_reference_splits"]))
    written: list[Path] = []
    for model_key in config["models"]["generation_models"]:
        native = _required_part(part_path(config, f"generation_native_scores_{model_key}.json"))["scores"]
        sampler_rows: dict[str, Any] = {}
        for sampler, source in native.items():
            frame = _load_library(Path(source["csv_path"]),
                                  set(int(step) for step in generation["retained_steps"]) if sampler in {"gibbs", "stochastic_beam"} else None)
            annotated = annotate_sequence_membership(frame, seq_col="sequence", reference_index=novelty_index)
            common_source = common[_common_library_key(model_key, sampler)]
            rows: list[dict[str, Any]] = []
            for record in annotated.to_dict("records"):
                sequence = str(record["sequence"])
                rows.append({"sequence": sequence, "n_mutations": int(record["n_mutations"]),
                             "native_pll": float(source["scores"][sequence]),
                             "common_esm2_pll": float(common_source["scores"][sequence]),
                             "wt_native_pll": float(source["wt_score"]),
                             "wt_common_esm2_pll": float(common_source["wt_score"]),
                             "present_in_training_reference": bool(record["present_in_existing_dataset"])})
            sampler_rows[sampler] = {"rows": rows,
                                     "native": library_statistics(rows, evaluator="native", reference=reference, top_k=int(generation["top_k"])),
                                     "common_esm2": library_statistics(rows, evaluator="common", reference=reference, top_k=int(generation["top_k"]))}
        out = artifact_path(config, f"generation_library_{model_key}.json")
        write_artifact(out, {"model": model_key, "checkpoint": registry.resolve_model(model_key)["checkpoint"],
                             "generation": generation, "samplers": sampler_rows,
                             "config_fingerprint": fingerprint(config)})
        written.append(out)
    return written


def collect_generation_libraries(config: dict[str, Any], *, force: bool = False) -> list[Path]:
    """Compatibility wrapper for a serial collection invocation."""
    collect_generation_baselines(config, force=force)
    for model_key in config["models"]["generation_models"]:
        for sampler in ("gibbs", "stochastic_beam"):
            collect_generation_sample(config, model_key, sampler, profile="a100_80gb", force=force)
        collect_generation_native_scores(config, model_key, profile="a100_80gb", force=force)
    collect_generation_common_scores(config, profile="a100_80gb", force=force)
    return reduce_generation_libraries(config)
