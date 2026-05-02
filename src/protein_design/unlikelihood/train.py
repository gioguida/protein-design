"""Hydra-driven unlikelihood training entrypoint."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from protein_design.config import (
    build_model_config,
    build_run_config,
    generate_run_name,
)
from protein_design.constants import C05_CDRH3
from protein_design.dpo.data_processing import (
    build_clean_ed5_csv,
    build_processed_views,
    build_validation_perplexity_csvs,
)
from protein_design.dpo.dataset import default_data_paths
from protein_design.dpo.train import (
    _load_test_pll_eval_sets,
    _load_test_spearman_df,
    _load_validation_pll_eval_sets,
    _load_validation_spearman_df,
    evaluate_perplexity,
)
from protein_design.dpo.utils import load_hydra_runtime_modules
from protein_design.eval import corpus_perplexity, run_scoring_evaluation
from protein_design.model import ESM2Model
from protein_design.unlikelihood.data import (
    load_unwanted_lookup_json,
    make_unlikelihood_dataloaders,
)
from protein_design.unlikelihood.loss import (
    build_unwanted_token_id_lookup,
    unlikelihood_mlm_loss,
)
from protein_design.unlikelihood.preprocessing import build_unwanted_set
from protein_design.utils import ensure_dir, init_wandb, setup_train_logger

hydra, OmegaConf, HydraConfig, to_absolute_path = load_hydra_runtime_modules()

logger = logging.getLogger(__name__)


def _resolve_raw_ed2_csv_path(cfg: Any) -> Path:
    defaults = default_data_paths()
    return (
        defaults["raw_m22"]
        if getattr(cfg.data, "raw_csv", None) is None
        else Path(to_absolute_path(str(cfg.data.raw_csv)))
    )


def _resolve_processed_dir(cfg: Any) -> Path:
    defaults = default_data_paths()
    return (
        defaults["processed_dir"]
        if getattr(cfg.data, "processed_dir", None) is None
        else Path(to_absolute_path(str(cfg.data.processed_dir)))
    )


def _resolve_unwanted_json_path(cfg: Any) -> Path:
    raw_path = str(getattr(cfg.data, "unwanted_set_path", ""))
    if not raw_path:
        raise ValueError("data.unwanted_set_path must be set for unlikelihood training.")
    return Path(to_absolute_path(raw_path))


def _resolve_ed5_raw_csv_path(cfg: Any) -> Path:
    defaults = default_data_paths()
    fallback = defaults["raw_m22"].parent / "ED5_M22_enrichment.csv"
    test_cfg = getattr(cfg.data, "test", None)
    ed5_csv = None if test_cfg is None else getattr(test_cfg, "ed5_csv", None)
    return fallback if ed5_csv is None else Path(to_absolute_path(str(ed5_csv)))


def _ensure_unwanted_set_json(cfg: Any, run_log: logging.Logger) -> Path:
    unwanted_path = _resolve_unwanted_json_path(cfg)
    if unwanted_path.exists():
        return unwanted_path

    raw_csv_path = _resolve_raw_ed2_csv_path(cfg)
    processed_dir = unwanted_path.parent
    summary_csv_name = str(
        getattr(
            cfg.data,
            "unwanted_summary_csv_name",
            "unwanted_substitution_enrichment.csv",
        )
    )
    run_log.warning(
        "Unwanted-set JSON missing at %s. Building it automatically from %s.",
        unwanted_path,
        raw_csv_path,
    )
    summary_csv_path, built_json_path = build_unwanted_set(
        raw_csv_path=raw_csv_path,
        processed_dir=processed_dir,
        enrichment_col=str(
            getattr(cfg.data, "unwanted_enrichment_col", "M22_binding_enrichment_adj")
        ),
        wt_seq=str(getattr(cfg.data, "wt_seq", C05_CDRH3)),
        min_total_reads=int(getattr(cfg.data, "unwanted_min_total_reads", 10)),
        min_observations=int(getattr(cfg.data, "unwanted_min_observations", 30)),
        summary_csv_name=summary_csv_name,
        unwanted_json_name=unwanted_path.name,
    )
    run_log.info("Built unwanted summary CSV at %s", summary_csv_path)
    run_log.info("Built unwanted-set JSON at %s", built_json_path)

    if not unwanted_path.exists():
        raise FileNotFoundError(
            f"Unwanted-set JSON still missing after automatic build: {unwanted_path}"
        )
    return unwanted_path


def _ensure_preprocessed_artifacts(cfg: Any, run_log: logging.Logger) -> None:
    raw_ed2_csv = _resolve_raw_ed2_csv_path(cfg)
    processed_dir = _resolve_processed_dir(cfg)
    force = bool(getattr(cfg.data, "force_rebuild", False))

    if not raw_ed2_csv.exists():
        raise FileNotFoundError(f"ED2 raw CSV not found: {raw_ed2_csv}")

    processed_paths = build_processed_views(
        raw_csv_path=raw_ed2_csv,
        processed_dir=processed_dir,
        force=force,
        verbose=False,
    )
    run_log.info(
        "Ensured processed ED2 views: %s, %s, %s",
        processed_paths["ed2_all"],
        processed_paths["d2_clustered_mut1"],
        processed_paths["d2_clustered_mut2"],
    )

    try:
        val_outputs = build_validation_perplexity_csvs(
            raw_csv_path=raw_ed2_csv,
            processed_dir=processed_dir,
            cfg=cfg,
            seed=int(cfg.seed),
            force=force,
            verbose=False,
        )
        run_log.info(
            "Ensured validation eval CSVs: %s, %s, %s",
            val_outputs["val_pos"],
            val_outputs["val_neg"],
            val_outputs["val_spearman"],
        )
    except Exception as exc:
        run_log.warning("Could not prebuild validation eval CSVs (%s).", exc)

    ed5_raw_csv = _resolve_ed5_raw_csv_path(cfg)
    if ed5_raw_csv.exists():
        try:
            d5_path = build_clean_ed5_csv(
                raw_csv_path=ed5_raw_csv,
                processed_dir=processed_dir,
                force=force,
                verbose=False,
            )
            run_log.info("Ensured processed ED5 CSV: %s", d5_path)
        except Exception as exc:
            run_log.warning("Could not prebuild processed ED5 CSV (%s).", exc)
    else:
        run_log.warning(
            "ED5 raw CSV not found at %s. Test ED5 perplexity/Spearman may be unavailable.",
            ed5_raw_csv,
        )

    _ensure_unwanted_set_json(cfg, run_log)


def _load_unwanted_token_ids(cfg: Any, model: ESM2Model) -> Dict[int, List[int]]:
    unwanted_path = _resolve_unwanted_json_path(cfg)
    if not unwanted_path.exists():
        raise FileNotFoundError(
            f"Unwanted-set JSON not found at {unwanted_path}. "
            "Preflight artifact build should have created it."
        )

    unwanted_lookup = load_unwanted_lookup_json(unwanted_path)
    token_ids = build_unwanted_token_id_lookup(unwanted_lookup, tokenizer=model.tokenizer)
    if not token_ids:
        logger.warning("Unwanted-set JSON is empty after tokenization; unlikelihood term will be zero.")
    return token_ids


def _evaluate_unlikelihood_objective(
    model: ESM2Model,
    dataloader: DataLoader,
    device: torch.device,
    unwanted_token_ids_by_position: Dict[int, List[int]],
    alpha: float,
) -> Dict[str, float]:
    model.eval()
    total = 0.0
    mlm_total = 0.0
    ul_total = 0.0
    unwanted_prob_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=None,
            )
            terms = unlikelihood_mlm_loss(
                logits=outputs.logits,
                labels=batch["labels"],
                cdr_positions=batch["cdr_positions"],
                unwanted_token_ids_by_position=unwanted_token_ids_by_position,
                alpha=alpha,
            )
            total += float(terms["loss"].item())
            mlm_total += float(terms["mlm_loss"].item())
            ul_total += float(terms["unlikelihood_loss"].item())
            unwanted_prob_total += float(terms["unwanted_probability"].item())
            n_batches += 1

    n = max(1, n_batches)
    return {
        "loss": total / n,
        "mlm_loss": mlm_total / n,
        "unlikelihood_loss": ul_total / n,
        "unwanted_probability": unwanted_prob_total / n,
        "num_batches": float(n_batches),
    }


def _save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: ESM2Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    val_loss: float,
) -> None:
    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_loss": float(val_loss),
    }
    torch.save(state, path)


def run_unlikelihood(cfg: Any) -> Path:
    run_name = generate_run_name(cfg)
    run_cfg = build_run_config(cfg)
    run_dir = ensure_dir(str(Path(run_cfg.train_dir) / run_name))
    checkpoint_dir = ensure_dir(str(run_dir / "checkpoints"))

    level_name = str(getattr(cfg.logging, "level", "INFO"))
    log_every_n_steps = int(getattr(cfg.logging, "log_every_n_steps", 50))
    run_log = setup_train_logger(run_dir, level_name=level_name, logger_name=__name__)

    snapshot = OmegaConf.to_container(cfg, resolve=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.dump(snapshot, fh, default_flow_style=False, sort_keys=False)
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")

    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_log.info("Device: %s", device)
    run_log.info("Run directory: %s", run_dir)
    _ensure_preprocessed_artifacts(cfg, run_log)

    wandb_mod, wandb_run = init_wandb(
        cfg,
        run_dir,
        run_log,
        run_name=run_name,
        group="unlikelihood",
    )

    model_cfg = build_model_config(cfg, device=str(device))
    model = ESM2Model(model_cfg)

    if run_cfg.finetune:
        finetune_path = Path(to_absolute_path(str(run_cfg.finetune)))
        run_log.info("Loading finetune checkpoint: %s", finetune_path)
        ckpt = torch.load(finetune_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    summary = model.param_summary()
    run_log.info(
        "Parameters - total: %s, trainable: %s, frozen: %s",
        f"{summary['total']:,}",
        f"{summary['trainable']:,}",
        f"{summary['frozen']:,}",
    )
    model.to(device)

    alpha = float(getattr(cfg.training, "alpha", 1.0))
    unwanted_token_ids_by_position = _load_unwanted_token_ids(cfg, model=model)
    train_loader, val_loader, test_loader, sequences_by_split = make_unlikelihood_dataloaders(
        cfg=cfg,
        tokenizer=model.tokenizer,
        to_absolute_path=to_absolute_path,
    )
    run_log.info(
        "Good-sequence split sizes | train=%d val=%d test=%d",
        len(sequences_by_split["train"]),
        len(sequences_by_split["val"]),
        len(sequences_by_split["test"]),
    )
    if len(sequences_by_split["train"]) == 0:
        raise ValueError("No training sequences above enrichment threshold.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found.")

    optimizer = AdamW(trainable, lr=float(cfg.training.learning_rate), weight_decay=0.01)
    accum_steps = int(cfg.training.gradient_accumulation_steps)
    max_steps = cfg.training.max_steps
    epoch_based_steps = int(cfg.training.max_epochs) * len(train_loader) // max(1, accum_steps)
    num_training_steps = int(max_steps) if max_steps else int(epoch_based_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.training.warmup_steps),
        num_training_steps=max(1, num_training_steps),
    )

    use_fp16 = bool(cfg.training.fp16) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    val_eval_sets = _load_validation_pll_eval_sets(cfg, run_log)
    val_spearman_df = _load_validation_spearman_df(cfg, run_log)
    test_eval_sets = _load_test_pll_eval_sets(cfg, run_log)
    test_spearman_df = _load_test_spearman_df(cfg, run_log)
    scoring_batch_size = int(getattr(cfg.model, "pll_mask_chunk_size", 64))

    global_step = 0
    optim_step = 0
    save_every_n_steps = cfg.training.save_every_n_steps
    max_epochs = int(cfg.training.max_epochs)
    training_history: List[Dict[str, float]] = []
    scoring_history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_val_ppl_pos = float("inf")
    best_val_spearman_avg = float("-inf")
    best_ckpt_path: Optional[Path] = None
    train_start = time.time()
    last_validation_eval_step = -1

    running_total = 0.0
    running_mlm = 0.0
    running_ul = 0.0
    running_unwanted_prob = 0.0
    running_batches = 0
    hit_max_steps = False

    val_objective = _evaluate_unlikelihood_objective(
        model=model,
        dataloader=val_loader,
        device=device,
        unwanted_token_ids_by_position=unwanted_token_ids_by_position,
        alpha=alpha,
    )
    val_cdr_ppl = corpus_perplexity(
        sequences_by_split["val"],
        scorer=model,
        cdr_only=True,
    ) if sequences_by_split["val"] else float("nan")
    eval_record = {
        "step": float(global_step),
        "epoch": 0.0,
        "val_loss": float(val_objective["loss"]),
        "val_mlm_loss": float(val_objective["mlm_loss"]),
        "val_unlikelihood_loss": float(val_objective["unlikelihood_loss"]),
        "val_unwanted_probability": float(val_objective["unwanted_probability"]),
        "val_cdr_perplexity": float(val_cdr_ppl),
        "wall_time": float(time.time() - train_start),
    }

    if val_eval_sets is not None:
        val_ppl_metrics = evaluate_perplexity(
            model=model,
            eval_sets=val_eval_sets,
            device=str(device),
        )
        eval_record.update(val_ppl_metrics)

    if val_spearman_df is not None:
        try:
            val_spearman = run_scoring_evaluation(
                scorer=model,
                df=val_spearman_df,
                enrichment_col="M22_binding_enrichment_adj",
                batch_size=scoring_batch_size,
                seed=int(cfg.seed),
                scoring_mode="cdr_pll",
            )
            eval_record.update(
                {
                    "val_spearman_avg": float(val_spearman["spearman_avg"]),
                    "val_spearman_avg_pval": float(val_spearman["spearman_avg_pval"]),
                    "val_spearman_random": float(val_spearman["spearman_random"]),
                    "val_spearman_random_pval": float(val_spearman["spearman_random_pval"]),
                }
            )
        except Exception as exc:  # pragma: no cover - best effort eval
            run_log.warning("Validation Spearman evaluation failed at step 0 (%s).", exc)

    training_history.append(eval_record)
    scoring_history.append(dict(eval_record))
    run_log.info(
        "Validation (step 0) - loss: %.4f - cdr_ppl: %.4f - unwanted_prob: %.4f",
        float(eval_record["val_loss"]),
        float(eval_record["val_cdr_perplexity"]),
        float(eval_record["val_unwanted_probability"]),
    )
    if wandb_run is not None:
        wandb_mod.log(
            {f"val/{k}": v for k, v in eval_record.items() if k not in {"step", "epoch", "wall_time"}},
            step=global_step,
        )

    step0_ckpt_path = checkpoint_dir / "step_0.pt"
    _save_checkpoint(
        path=step0_ckpt_path,
        epoch=0,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        val_loss=float(eval_record["val_loss"]),
    )
    best_val_loss = min(best_val_loss, float(eval_record["val_loss"]))
    candidate_val_ppl_pos = float(eval_record.get("ppl/val_pos", float("nan")))
    candidate_val_spearman = float(eval_record.get("val_spearman_avg", float("nan")))
    should_update_best = False
    reason = ""
    if math.isfinite(candidate_val_ppl_pos):
        if (not math.isfinite(best_val_ppl_pos)) or candidate_val_ppl_pos < best_val_ppl_pos:
            should_update_best = True
            reason = "ppl/val_pos"
    elif (not math.isfinite(best_val_ppl_pos)) and math.isfinite(candidate_val_spearman):
        if (not math.isfinite(best_val_spearman_avg)) or candidate_val_spearman > best_val_spearman_avg:
            should_update_best = True
            reason = "val_spearman_avg"

    if should_update_best:
        if reason == "ppl/val_pos":
            best_val_ppl_pos = candidate_val_ppl_pos
        elif reason == "val_spearman_avg":
            best_val_spearman_avg = candidate_val_spearman
        best_ckpt_path = run_dir / "best.pt"
        _save_checkpoint(
            path=best_ckpt_path,
            epoch=0,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            val_loss=best_val_loss,
        )
        run_log.info(
            "New best checkpoint saved to %s (%s=%.6f) [step 0]",
            best_ckpt_path,
            reason,
            candidate_val_ppl_pos if reason == "ppl/val_pos" else candidate_val_spearman,
        )

    last_validation_eval_step = global_step
    model.train()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_had_batches = False
        for batch in train_loader:
            epoch_had_batches = True
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_fp16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None,
                )
                terms = unlikelihood_mlm_loss(
                    logits=outputs.logits,
                    labels=batch["labels"],
                    cdr_positions=batch["cdr_positions"],
                    unwanted_token_ids_by_position=unwanted_token_ids_by_position,
                    alpha=alpha,
                )
                loss = terms["loss"] / max(1, accum_steps)

            scaler.scale(loss).backward()

            running_total += float(terms["loss"].item())
            running_mlm += float(terms["mlm_loss"].item())
            running_ul += float(terms["unlikelihood_loss"].item())
            running_unwanted_prob += float(terms["unwanted_probability"].item())
            running_batches += 1

            if global_step % max(1, accum_steps) == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

            if global_step % max(1, log_every_n_steps) == 0:
                denom = max(1, running_batches)
                avg_total = running_total / denom
                avg_mlm = running_mlm / denom
                avg_ul = running_ul / denom
                avg_unwanted_prob = running_unwanted_prob / denom
                lr = float(scheduler.get_last_lr()[0])

                run_log.info(
                    "Epoch %d Step %d - total: %.4f - mlm: %.4f - unlikelihood: %.4f - unwanted_prob: %.4f - lr: %.2e",
                    epoch,
                    global_step,
                    avg_total,
                    avg_mlm,
                    avg_ul,
                    avg_unwanted_prob,
                    lr,
                )
                step_record = {
                    "step": float(global_step),
                    "epoch": float(epoch),
                    "train_total_loss": float(avg_total),
                    "train_mlm_loss": float(avg_mlm),
                    "train_unlikelihood_loss": float(avg_ul),
                    "train_unwanted_probability": float(avg_unwanted_prob),
                    "learning_rate": float(lr),
                    "wall_time": float(time.time() - train_start),
                }
                training_history.append(step_record)
                if wandb_run is not None:
                    wandb_mod.log(
                        {
                            "train/loss": avg_total,
                            "train/mlm_loss": avg_mlm,
                            "train/unlikelihood_loss": avg_ul,
                            "train/unwanted_probability": avg_unwanted_prob,
                            "train/lr": lr,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

                running_total = 0.0
                running_mlm = 0.0
                running_ul = 0.0
                running_unwanted_prob = 0.0
                running_batches = 0

            if save_every_n_steps and global_step % int(save_every_n_steps) == 0:
                val_objective = _evaluate_unlikelihood_objective(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    unwanted_token_ids_by_position=unwanted_token_ids_by_position,
                    alpha=alpha,
                )
                val_cdr_ppl = corpus_perplexity(
                    sequences_by_split["val"],
                    scorer=model,
                    cdr_only=True,
                ) if sequences_by_split["val"] else float("nan")
                eval_record: Dict[str, float] = {
                    "step": float(global_step),
                    "epoch": float(epoch),
                    "val_loss": float(val_objective["loss"]),
                    "val_mlm_loss": float(val_objective["mlm_loss"]),
                    "val_unlikelihood_loss": float(val_objective["unlikelihood_loss"]),
                    "val_unwanted_probability": float(val_objective["unwanted_probability"]),
                    "val_cdr_perplexity": float(val_cdr_ppl),
                    "wall_time": float(time.time() - train_start),
                }

                if val_eval_sets is not None:
                    val_ppl_metrics = evaluate_perplexity(
                        model=model,
                        eval_sets=val_eval_sets,
                        device=str(device),
                    )
                    eval_record.update(val_ppl_metrics)

                if val_spearman_df is not None:
                    try:
                        val_spearman = run_scoring_evaluation(
                            scorer=model,
                            df=val_spearman_df,
                            enrichment_col="M22_binding_enrichment_adj",
                            batch_size=scoring_batch_size,
                            seed=int(cfg.seed),
                            scoring_mode="cdr_pll",
                        )
                        eval_record.update(
                            {
                                "val_spearman_avg": float(val_spearman["spearman_avg"]),
                                "val_spearman_avg_pval": float(val_spearman["spearman_avg_pval"]),
                                "val_spearman_random": float(val_spearman["spearman_random"]),
                                "val_spearman_random_pval": float(val_spearman["spearman_random_pval"]),
                            }
                        )
                    except Exception as exc:  # pragma: no cover - best effort eval
                        run_log.warning("Validation Spearman evaluation failed (%s).", exc)

                training_history.append(eval_record)
                scoring_history.append(dict(eval_record))
                run_log.info(
                    "Validation @ step %d - loss: %.4f - cdr_ppl: %.4f - unwanted_prob: %.4f",
                    global_step,
                    float(eval_record["val_loss"]),
                    float(eval_record["val_cdr_perplexity"]),
                    float(eval_record["val_unwanted_probability"]),
                )
                if wandb_run is not None:
                    wandb_mod.log({f"val/{k}": v for k, v in eval_record.items() if k not in {"step", "epoch", "wall_time"}}, step=global_step)

                ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                _save_checkpoint(
                    path=ckpt_path,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    val_loss=float(eval_record["val_loss"]),
                )
                best_val_loss = min(best_val_loss, float(eval_record["val_loss"]))
                candidate_val_ppl_pos = float(eval_record.get("ppl/val_pos", float("nan")))
                candidate_val_spearman = float(eval_record.get("val_spearman_avg", float("nan")))
                should_update_best = False
                reason = ""
                if math.isfinite(candidate_val_ppl_pos):
                    if (not math.isfinite(best_val_ppl_pos)) or candidate_val_ppl_pos < best_val_ppl_pos:
                        should_update_best = True
                        reason = "ppl/val_pos"
                elif (not math.isfinite(best_val_ppl_pos)) and math.isfinite(candidate_val_spearman):
                    if (not math.isfinite(best_val_spearman_avg)) or candidate_val_spearman > best_val_spearman_avg:
                        should_update_best = True
                        reason = "val_spearman_avg"

                if should_update_best:
                    if reason == "ppl/val_pos":
                        best_val_ppl_pos = candidate_val_ppl_pos
                    elif reason == "val_spearman_avg":
                        best_val_spearman_avg = candidate_val_spearman
                    best_ckpt_path = run_dir / "best.pt"
                    _save_checkpoint(
                        path=best_ckpt_path,
                        epoch=epoch,
                        global_step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        val_loss=best_val_loss,
                    )
                    run_log.info(
                        "New best checkpoint saved to %s (%s=%.6f)",
                        best_ckpt_path,
                        reason,
                        candidate_val_ppl_pos if reason == "ppl/val_pos" else candidate_val_spearman,
                    )

                last_validation_eval_step = global_step
                model.train()

            if max_steps and optim_step >= int(max_steps):
                run_log.info("Reached max_steps=%d, stopping training.", int(max_steps))
                hit_max_steps = True
                break

        if epoch_had_batches and global_step != last_validation_eval_step:
            val_objective = _evaluate_unlikelihood_objective(
                model=model,
                dataloader=val_loader,
                device=device,
                unwanted_token_ids_by_position=unwanted_token_ids_by_position,
                alpha=alpha,
            )
            val_cdr_ppl = corpus_perplexity(
                sequences_by_split["val"],
                scorer=model,
                cdr_only=True,
            ) if sequences_by_split["val"] else float("nan")
            eval_record = {
                "step": float(global_step),
                "epoch": float(epoch),
                "val_loss": float(val_objective["loss"]),
                "val_mlm_loss": float(val_objective["mlm_loss"]),
                "val_unlikelihood_loss": float(val_objective["unlikelihood_loss"]),
                "val_unwanted_probability": float(val_objective["unwanted_probability"]),
                "val_cdr_perplexity": float(val_cdr_ppl),
                "wall_time": float(time.time() - train_start),
            }

            if val_eval_sets is not None:
                val_ppl_metrics = evaluate_perplexity(
                    model=model,
                    eval_sets=val_eval_sets,
                    device=str(device),
                )
                eval_record.update(val_ppl_metrics)

            if val_spearman_df is not None:
                try:
                    val_spearman = run_scoring_evaluation(
                        scorer=model,
                        df=val_spearman_df,
                        enrichment_col="M22_binding_enrichment_adj",
                        batch_size=scoring_batch_size,
                        seed=int(cfg.seed),
                        scoring_mode="cdr_pll",
                    )
                    eval_record.update(
                        {
                            "val_spearman_avg": float(val_spearman["spearman_avg"]),
                            "val_spearman_avg_pval": float(val_spearman["spearman_avg_pval"]),
                            "val_spearman_random": float(val_spearman["spearman_random"]),
                            "val_spearman_random_pval": float(val_spearman["spearman_random_pval"]),
                        }
                    )
                except Exception as exc:  # pragma: no cover - best effort eval
                    run_log.warning("Validation Spearman evaluation failed (%s).", exc)

            training_history.append(eval_record)
            scoring_history.append(dict(eval_record))
            run_log.info(
                "Validation (epoch end) @ step %d - loss: %.4f - cdr_ppl: %.4f - unwanted_prob: %.4f",
                global_step,
                float(eval_record["val_loss"]),
                float(eval_record["val_cdr_perplexity"]),
                float(eval_record["val_unwanted_probability"]),
            )
            if wandb_run is not None:
                wandb_mod.log(
                    {f"val/{k}": v for k, v in eval_record.items() if k not in {"step", "epoch", "wall_time"}},
                    step=global_step,
                )

            ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
            _save_checkpoint(
                path=ckpt_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                val_loss=float(eval_record["val_loss"]),
            )
            best_val_loss = min(best_val_loss, float(eval_record["val_loss"]))
            candidate_val_ppl_pos = float(eval_record.get("ppl/val_pos", float("nan")))
            candidate_val_spearman = float(eval_record.get("val_spearman_avg", float("nan")))
            should_update_best = False
            reason = ""
            if math.isfinite(candidate_val_ppl_pos):
                if (not math.isfinite(best_val_ppl_pos)) or candidate_val_ppl_pos < best_val_ppl_pos:
                    should_update_best = True
                    reason = "ppl/val_pos"
            elif (not math.isfinite(best_val_ppl_pos)) and math.isfinite(candidate_val_spearman):
                if (not math.isfinite(best_val_spearman_avg)) or candidate_val_spearman > best_val_spearman_avg:
                    should_update_best = True
                    reason = "val_spearman_avg"

            if should_update_best:
                if reason == "ppl/val_pos":
                    best_val_ppl_pos = candidate_val_ppl_pos
                elif reason == "val_spearman_avg":
                    best_val_spearman_avg = candidate_val_spearman
                best_ckpt_path = run_dir / "best.pt"
                _save_checkpoint(
                    path=best_ckpt_path,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    val_loss=best_val_loss,
                )
                run_log.info(
                    "New best checkpoint saved to %s (%s=%.6f)",
                    best_ckpt_path,
                    reason,
                    candidate_val_ppl_pos if reason == "ppl/val_pos" else candidate_val_spearman,
                )

            last_validation_eval_step = global_step
            model.train()

        if wandb_run is not None:
            wandb_mod.log({"train/epoch": epoch}, step=global_step)
        if hit_max_steps:
            break

    final_ckpt = checkpoint_dir / "final.pt"
    _save_checkpoint(
        path=final_ckpt,
        epoch=max_epochs,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        val_loss=best_val_loss if math.isfinite(best_val_loss) else float("inf"),
    )
    run_log.info("Saved final checkpoint to %s", final_ckpt)

    if bool(getattr(cfg.training, "evaluate_best_checkpoint", True)) and best_ckpt_path is not None and best_ckpt_path.exists():
        best_state = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(best_state["model_state_dict"])
        run_log.info("Loaded best checkpoint for final test evaluation: %s", best_ckpt_path)

    test_objective = _evaluate_unlikelihood_objective(
        model=model,
        dataloader=test_loader,
        device=device,
        unwanted_token_ids_by_position=unwanted_token_ids_by_position,
        alpha=alpha,
    )
    test_cdr_ppl = corpus_perplexity(
        sequences_by_split["test"],
        scorer=model,
        cdr_only=True,
    ) if sequences_by_split["test"] else float("nan")

    test_metrics: Dict[str, Any] = {
        "test_loss": float(test_objective["loss"]),
        "test_mlm_loss": float(test_objective["mlm_loss"]),
        "test_unlikelihood_loss": float(test_objective["unlikelihood_loss"]),
        "test_unwanted_probability": float(test_objective["unwanted_probability"]),
        "test_cdr_perplexity": float(test_cdr_ppl),
        "ppl/test_pos": None,
        "ppl/test_neg": None,
        "ppl/test_wt": None,
        "test_spearman_avg": None,
        "test_spearman_avg_pval": None,
        "test_spearman_random": None,
        "test_spearman_random_pval": None,
    }
    if test_eval_sets is not None:
        test_metrics.update(
            evaluate_perplexity(
                model=model,
                eval_sets=test_eval_sets,
                device=str(device),
            )
        )
    if test_spearman_df is not None:
        try:
            test_spearman = run_scoring_evaluation(
                scorer=model,
                df=test_spearman_df,
                enrichment_col="M22_binding_enrichment_adj",
                batch_size=scoring_batch_size,
                seed=int(cfg.seed),
                scoring_mode="cdr_pll",
            )
            test_metrics.update(
                {
                    "test_spearman_avg": float(test_spearman["spearman_avg"]),
                    "test_spearman_avg_pval": float(test_spearman["spearman_avg_pval"]),
                    "test_spearman_random": float(test_spearman["spearman_random"]),
                    "test_spearman_random_pval": float(test_spearman["spearman_random_pval"]),
                }
            )
        except Exception as exc:  # pragma: no cover - best effort eval
            run_log.warning("Test Spearman evaluation failed (%s).", exc)

    metrics = {
        "metadata": {
            "run_name": run_name,
            "total_steps": int(global_step),
            "total_time_seconds": round(time.time() - train_start, 2),
            "device": str(device),
            "param_summary": summary,
            "num_train_sequences": int(len(sequences_by_split["train"])),
            "num_val_sequences": int(len(sequences_by_split["val"])),
            "num_test_sequences": int(len(sequences_by_split["test"])),
        },
        "training_history": training_history,
        "scoring_history": scoring_history,
        "final_metrics": test_metrics,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    run_log.info("Saved metrics to %s", run_dir / "metrics.json")

    summary_payload = {
        "run_name": run_name,
        "best_val_loss": None if not math.isfinite(best_val_loss) else float(best_val_loss),
        **test_metrics,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    if wandb_run is not None:
        wandb_mod.log({f"test/{k}": v for k, v in test_metrics.items()}, step=global_step)
        wandb_run.summary.update(summary_payload)
        wandb_run.finish()

    if best_ckpt_path is not None and best_ckpt_path.exists():
        return best_ckpt_path
    return final_ckpt


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: Any) -> None:
    HydraConfig.get().runtime.output_dir
    task_cfg = getattr(cfg, "task", None)
    runner = None if task_cfg is None else str(getattr(task_cfg, "runner", "")).strip().lower()
    if runner != "unlikelihood":
        raise ValueError(
            "Unlikelihood entrypoint requires task.runner='unlikelihood'. "
            "Run with `task=unlikelihood`."
        )
    run_unlikelihood(cfg)


if __name__ == "__main__":
    import sys

    if all(not arg.startswith("task=") for arg in sys.argv[1:]):
        sys.argv.insert(1, "task=unlikelihood")
    main()
