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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

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


def artifact_names(config: dict[str, Any], sections: Iterable[str] | None = None) -> list[str]:
    sections = set(sections or ("functional", "preference", "generation"))
    names: list[str] = []
    if "functional" in sections:
        names.append("functional_metrics.json")
    if "preference" in sections:
        names.append("preference_test_metrics.json")
    if "generation" in sections:
        names.append("generation_reference_ed2_m22_test.json")
        names.extend(f"generation_library_{key}.json" for key in config["models"]["generation_models"])
    return names


def missing_artifacts(config: dict[str, Any], sections: Iterable[str] | None = None) -> list[Path]:
    return [artifact_path(config, name) for name in artifact_names(config, sections) if not artifact_path(config, name).exists()]


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
    path.write_text(json.dumps(_json_value(full_payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_artifact(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported report artifact schema in {path}: {payload.get('schema_version')!r}")
    return payload


def complete_preference_summary(summary: dict[str, Any]) -> bool:
    return all(summary.get(name) is not None and math.isfinite(float(summary[name])) for name in REQUIRED_PREFERENCE_METRICS)


def _resolved_config(checkpoint: Path, config: dict[str, Any], model_key: str):
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
    cfg.training.batch_size = int(pref["batch_size"])
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


def recompute_preference_metrics(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    """Evaluate a checkpoint on the exact DPO test-pair construction.

    This calls the DPO module's existing pair construction, dataloader and
    epoch routine.  The only new responsibility is selecting the stored policy
    checkpoint and its frozen reference checkpoint.
    """
    from protein_design.dpo import train as dpo_train
    from protein_design.dpo.dataset import build_split_pair_dataframes_from_cfg

    spec = registry.resolve_model(model_key)
    checkpoint = Path(str(spec["checkpoint"]))
    cfg, config_source = _resolved_config(checkpoint, config, model_key)
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


def collect_preference_metrics(config: dict[str, Any], *, force: bool = False) -> Path:
    out = artifact_path(config, "preference_test_metrics.json")
    if out.exists() and not force:
        return out
    rows: dict[str, Any] = {}
    for model_key in config["models"]["preference_models"]:
        checkpoint = Path(str(registry.resolve_model(model_key)["checkpoint"]))
        summary_path = checkpoint.parent / "summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if complete_preference_summary(summary):
            row = {name: float(summary[name]) for name in REQUIRED_PREFERENCE_METRICS}
            row.update({"test_pairs": summary.get("test_pairs"), "source": "summary",
                        "summary_path": str(summary_path)})
        else:
            row = recompute_preference_metrics(config, model_key)
        row["checkpoint"] = str(checkpoint)
        row["checkpoint_fingerprint"] = fingerprint({"checkpoint": str(checkpoint), "model": model_key})
        rows[model_key] = row
    write_artifact(out, {"models": rows, "config_fingerprint": fingerprint(config)})
    return out


def _run(command: list[str]) -> None:
    print("[run]", " ".join(command))
    subprocess.run(command, cwd=registry.REPO_ROOT, check=True)


def collect_functional_metrics(config: dict[str, Any], *, force: bool = False) -> Path:
    out = artifact_path(config, "functional_metrics.json")
    if out.exists() and not force:
        return out
    datasets = config["datasets"]["functional"]
    model_rows: dict[str, Any] = {}
    for model_key in config["models"]["order"]:
        _run([sys.executable, "scripts/analysis/compute_pll.py", "--model", model_key,
              "--dataset", ",".join(datasets), "--device", "cuda"] + (["--force"] if force else []))
        per_dataset: dict[str, Any] = {}
        for dataset_key in datasets:
            ds = registry.load_datasets_cfg()["datasets"][dataset_key]
            joined = registry.load_pll(model_key, dataset_key).merge(registry.load_truth(dataset_key), on=ds["seq_col"], how="inner")
            pll = joined["pll"].to_numpy(float)
            truth = joined["enrichment"].to_numpy(float)
            valid = np.isfinite(pll) & np.isfinite(truth)
            rho = float(spearmanr(pll[valid], truth[valid]).statistic) if valid.sum() >= 2 else float("nan")
            cdr_ppl = float(math.exp(-pll[valid].sum() / sum(len(s) for s in joined.loc[valid, ds["seq_col"]])))
            per_dataset[dataset_key] = {"spearman_pll_enrichment": rho, "n": int(valid.sum()), "cdr_pseudo_perplexity": cdr_ppl}
        model_rows[model_key] = {"checkpoint": registry.resolve_model(model_key)["checkpoint"], "datasets": per_dataset}
    write_artifact(out, {"models": model_rows, "datasets": datasets, "config_fingerprint": fingerprint(config)})
    return out


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


def _generate_model_libraries(config: dict[str, Any], model_key: str, work_dir: Path, *, force: bool) -> dict[str, Path]:
    generation = config["generation"]
    model = registry.resolve_model(model_key)
    outputs: dict[str, Path] = {}
    specs = {
        "gibbs": [sys.executable, "scripts/gibbs_sampling.py", "--model-variant", model_key,
                  "--checkpoint-path", str(model["checkpoint"]), "--n-chains", str(generation["chains_or_beams"]),
                  "--n-steps", str(max(generation["retained_steps"])), "--snapshot-every", "1",
                  "--temperature", str(generation["temperature"]), "--seed", str(generation["seed"]),
                  "--max-mutations", str(generation["max_mutations"]), "--start-mode", "wt"],
        "stochastic_beam": [sys.executable, "scripts/stochastic_beam_search.py", "--model-variant", model_key,
                            "--checkpoint-path", str(model["checkpoint"]), "--beam-size", str(generation["chains_or_beams"]),
                            "--n-steps", str(max(generation["retained_steps"])), "--snapshot-every", "1",
                            "--temperature", str(generation["temperature"]), "--seed", str(generation["seed"]), "--start-mode", "wt"],
    }
    for sampler, command in specs.items():
        path = work_dir / f"{model_key}_{sampler}.csv"
        if force or not path.exists():
            _run([*command, "--output-path", str(path)])
        outputs[sampler] = path
    return outputs


def _load_library(path: Path, retained_steps: set[int] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if retained_steps is not None:
        frame = frame[frame["gibbs_step"].isin(retained_steps)].copy()
    frame = frame[frame["cdrh3"].astype(str).str.len() == len(C05_CDRH3)].copy()
    if frame.empty:
        raise ValueError(f"No valid CDR-H3 sequences in {path}")
    return frame[["cdrh3", "n_mutations"]].rename(columns={"cdrh3": "sequence"}).reset_index(drop=True)


def _score_library(path: Path, model_key: str, sequences: pd.DataFrame, work_dir: Path, *, force: bool) -> tuple[dict[str, float], float]:
    model = registry.resolve_model(model_key)
    input_path = work_dir / f"score_input_{model_key}_{path.stem}.csv"
    scored_path = work_dir / f"scores_{model_key}_{path.stem}.csv"
    score_input = pd.concat([sequences[["sequence"]], pd.DataFrame({"sequence": [C05_CDRH3]})], ignore_index=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    score_input.to_csv(input_path, index=False)
    if force or not scored_path.exists():
        command = [sys.executable, "scripts/analysis/score_generated_with_pll.py", "--input-csv", str(input_path),
                   "--seq-col", "sequence", "--output-csv", str(scored_path), "--base-model", str(model["base_model"]),
                   "--device", "cuda"]
        if model["checkpoint"]:
            command.extend(["--checkpoint", str(model["checkpoint"])])
        _run(command)
    scores = pd.read_csv(scored_path)
    score_map = dict(zip(scores["sequence"].astype(str), scores["pll"].astype(float)))
    if C05_CDRH3 not in score_map:
        raise ValueError(f"WT score absent from {scored_path}")
    return score_map, float(score_map[C05_CDRH3])


def collect_generation_libraries(config: dict[str, Any], *, force: bool = False) -> list[Path]:
    reference_path = build_generation_reference(config, force=force)
    reference = load_artifact(reference_path)["sequences"]
    work_dir = _generation_work_dir()
    generation = config["generation"]
    baselines = _generate_baselines(config, work_dir, force=force)
    novelty_index = build_reference_index(registry.REPO_ROOT, splits=set(generation["novelty_reference_splits"]))
    written: list[Path] = []
    common_model = str(generation["common_evaluator"])
    for model_key in generation["generation_models"]:
        out = artifact_path(config, f"generation_library_{model_key}.json")
        if out.exists() and not force:
            written.append(out)
            continue
        files = {**baselines, **_generate_model_libraries(config, model_key, work_dir, force=force)}
        sampler_rows: dict[str, Any] = {}
        for sampler, csv_path in files.items():
            retained = set(int(step) for step in generation["retained_steps"]) if sampler in {"gibbs", "stochastic_beam"} else None
            frame = _load_library(csv_path, retained)
            if len(frame) != int(generation["n_sequences"]):
                raise ValueError(f"{model_key}/{sampler} has {len(frame)} rows, expected {generation['n_sequences']}")
            native_scores, native_wt = _score_library(csv_path, model_key, frame, work_dir, force=force)
            common_scores, common_wt = _score_library(csv_path, common_model, frame, work_dir, force=force)
            annotated = annotate_sequence_membership(frame, seq_col="sequence", reference_index=novelty_index)
            rows: list[dict[str, Any]] = []
            for record in annotated.to_dict("records"):
                sequence = str(record["sequence"])
                rows.append({"sequence": sequence, "n_mutations": int(record["n_mutations"]),
                             "native_pll": float(native_scores[sequence]), "common_esm2_pll": float(common_scores[sequence]),
                             "wt_native_pll": native_wt, "wt_common_esm2_pll": common_wt,
                             "present_in_training_reference": bool(record["present_in_existing_dataset"])})
            sampler_rows[sampler] = {"rows": rows,
                                     "native": library_statistics(rows, evaluator="native", reference=reference, top_k=int(generation["top_k"])),
                                     "common_esm2": library_statistics(rows, evaluator="common", reference=reference, top_k=int(generation["top_k"]))}
        write_artifact(out, {"model": model_key, "checkpoint": registry.resolve_model(model_key)["checkpoint"],
                             "generation": generation, "samplers": sampler_rows,
                             "config_fingerprint": fingerprint(config)})
        written.append(out)
    return written
