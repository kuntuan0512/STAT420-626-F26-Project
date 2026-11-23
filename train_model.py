#!/usr/bin/env python3
"""Train/evaluate the simplified true 2D study on 100 patient groups.
Defaults: A0 masked mean versus A1 cross-attention; width16; seed11; 10 epochs;
16 sampled training slices per patient per epoch; batch4; validation every2 epochs.
Both models receive matched full/single-missing supervision. No consistency tuning
is required for the default comparison. A2 is an optional extension.
The main exploratory comparison is A1-A0 across four missing conditions and WT/TC/ET.
2D predictions are stacked for patient-volume Dice/HD95; the model sees one slice.
Use a fresh subset-specific output directory; earlier 2.5D checkpoints are incompatible.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

# Required for deterministic CUDA matrix operations; set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError("Install PyTorch in your chosen environment. See STAT626_CODE_README.md.") from exc

from eda import MODALITIES, REGIONS, build_2d_input, case_paths, load_case, plot_prediction_comparison
from design_network import (MODEL_OBJECTIVES, build_model, count_parameters,
                            drop_one_modality, model_objective, objective_kind)

CONDITIONS = ("complete", *MODALITIES)
METRIC_POLICY = {
    "dice_both_empty": 1.0,
    "dice_one_empty": 0.0,
    "hd95": "max of directed 95th-percentile 6-connected surface distances, mm",
    "hd95_both_empty_mm": 0.0,
    "hd95_one_empty": "norm((shape-1)*spacing_mm), flagged finite penalty",
    "aggregation": "average scans within patient, then equal patient weighting",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def fresh_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest(eda_dir: Path, allow_partial: bool = False) -> tuple[list[dict], dict]:
    """Verify EDA provenance and reject duplicate scans or patient leakage."""
    eda_dir = eda_dir.expanduser().resolve()
    summary = json.loads((eda_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("full_audit", False) and not allow_partial:
        raise ValueError("Full EDA QC is required. For a pipeline demo only, pass --allow-partial-audit.")
    for name, key in (("inventory.csv", "manifest_sha256"), ("patient_splits.csv", "split_sha256")):
        expected = summary.get(key)
        if not expected or sha256_file(eda_dir / name) != expected:
            raise ValueError(f"Missing or changed {name}. Rerun EDA into a fresh directory.")
    rows = read_csv(eda_dir / "patient_splits.csv")
    if not rows:
        raise ValueError("EDA contains no retained split rows")
    seen_cases, patient_splits = set(), {}
    retained_qc = {r["case_id"] for r in read_csv(eda_dir / "qc.csv") if r["qc_pass"].lower() == "true"}
    if {r["case_id"] for r in rows} != retained_qc:
        raise ValueError("Split membership does not match QC-passing scans")
    for row in rows:
        if not {"case_id", "patient_id", "case_dir", "split"} <= row.keys():
            raise ValueError("Split CSV has missing required columns")
        case, patient, split = row["case_id"], row["patient_id"], row["split"]
        if not patient or case in seen_cases or split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid or duplicate split record: {row}")
        seen_cases.add(case)
        if patient in patient_splits and patient_splits[patient] != split:
            raise ValueError(f"Patient leakage across partitions: {patient}")
        patient_splits[patient] = split
        folder = Path(row["case_dir"])
        if folder.name != case or not folder.is_absolute() or not folder.is_dir():
            raise ValueError(f"Case directory is missing or inconsistent: {folder}")
    if set(patient_splits.values()) != {"train", "validation", "test"}:
        raise ValueError("All three partitions must contain at least one patient group")
    # File-size checks catch moved/replaced/incomplete inputs without hashing every MRI byte.
    inventory = {r["case_id"]: r for r in read_csv(eda_dir / "inventory.csv")}
    for row in rows:
        source = inventory[row["case_id"]]
        for modality, filename in case_paths(Path(row["case_dir"])).items():
            if not filename.is_file():
                raise ValueError(f"Required file disappeared after EDA: {filename}")
            key = f"{modality}_bytes"
            if key in source and filename.stat().st_size != int(source[key]):
                raise ValueError(f"File size changed after EDA: {filename}")
    return sorted(rows, key=lambda r: r["case_id"]), summary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Unsupported deterministic operators raise rather than silently change the protocol.
    torch.use_deterministic_algorithms(True)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this PyTorch installation")
    return torch.device(name)


def amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def atomic_torch_save(value, path: Path) -> None:
    """Keep the last intact checkpoint when a runtime disconnects during saving."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def capture_rng_state() -> dict:
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_resume_config(previous, args, model_name, seed, weight, summary) -> None:
    expected = {"model": model_name, "seed": seed, "consistency_weight": weight,
                "base_channels": args.base_channels, "manifest_sha256": summary["manifest_sha256"],
                "split_sha256": summary["split_sha256"]}
    for key, value in expected.items():
        if previous.get(key) != value:
            raise ValueError(f"Resume configuration changed: {key}")
    for key in ("epochs", "batch_size", "slices_per_patient", "validate_every", "patience",
                "learning_rate", "weight_decay", "cache_cases", "no_augmentation", "no_amp", "device"):
        if previous["arguments"].get(key) != vars_json(args).get(key):
            raise ValueError(f"Resume training setting changed: {key}")


class PatientSliceDataset(Dataset):
    """16 sampled slices per patient, keeping a small volume cache.

    Patient blocks change order every epoch. Samples within a patient are
    shuffled, but not globally shuffled, to avoid repeated NIfTI decompression.
    Independent random streams pair slice sampling, augmentation and dropout
    between M0/M1/M2 and lambda candidates. Tumor/negative pools contain only
    slices with brain voxels, because consistency over an empty brain is undefined.
    """
    def __init__(self, rows: list[dict], seed: int, slices_per_patient: int = 16,
                 cache_size: int = 2, augment: bool = True):
        self.rows, self.seed = rows, seed
        self.slices_per_patient, self.cache_size, self.augment = slices_per_patient, cache_size, augment
        self.cache = OrderedDict()
        self.pools = defaultdict(lambda: {"positive": [], "negative": []})
        self.sampling_fallbacks = {}
        shapes = set()
        for index, row in enumerate(rows):
            case = load_case(Path(row["case_dir"]))
            seg, brain = case["segmentation"], case["brain_mask"]
            shapes.add(tuple(seg.shape[:2]))
            has_brain = brain.any(axis=(0, 1))
            tumor = (seg > 0).any(axis=(0, 1))
            for z in np.flatnonzero(has_brain):
                key = "positive" if tumor[z] else "negative"
                self.pools[row["patient_id"]][key].append((index, int(z)))
        if len(shapes) != 1:
            raise ValueError("Training scan slice dimensions differ. Define aligned preprocessing before batching.")
        if min(next(iter(shapes))) < 32:
            raise ValueError("Model inputs must be at least 32x32")
        for patient, pools in self.pools.items():
            if not pools["positive"] and not pools["negative"]:
                raise ValueError(f"No usable brain slices for {patient}")
            empty = [key for key, entries in pools.items() if not entries]
            if empty:
                self.sampling_fallbacks[patient] = empty
        self.set_epoch(1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 101]))
        patients = sorted(self.pools)
        rng.shuffle(patients)
        self.samples = []
        for patient in patients:
            pools = self.pools[patient]
            block = []
            all_entries = pools["positive"] + pools["negative"]
            scan_order = sorted({entry[0] for entry in all_entries})
            rng.shuffle(scan_order)
            for i in range(self.slices_per_patient):
                # Balance the patient's sampling budget across repeat scans first.
                # Then aim for half tumor slices, with a within-scan fallback.
                scan_index = scan_order[i % len(scan_order)]
                pool = [entry for entry in pools["positive" if i < self.slices_per_patient // 2 else "negative"]
                        if entry[0] == scan_index]
                if not pool:
                    pool = [entry for entry in all_entries if entry[0] == scan_index]
                block.append(pool[int(rng.integers(len(pool)))])
            rng.shuffle(block)
            self.samples.extend(block)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row_index, z = self.samples[index]
        row = self.rows[row_index]
        case_id = row["case_id"]
        if case_id not in self.cache:
            self.cache[case_id] = load_case(Path(row["case_dir"]))
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(case_id)
        case = self.cache[case_id]
        x = build_2d_input(case["volumes"], z)
        y = case["segmentation"][:, :, z].copy()
        brain = case["brain_mask"][:, :, z].copy()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, 303]))
        if self.augment:
            # Exact flips and rotations preserve label values without interpolation.
            for axis in (0, 1):
                if rng.random() < 0.5:
                    x, y, brain = np.flip(x, axis=axis + 1), np.flip(y, axis), np.flip(brain, axis)
            k = int(rng.integers(4)) if y.shape[0] == y.shape[1] else 2 * int(rng.integers(2))
            x, y, brain = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k), np.rot90(brain, k)
        missing_rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, 202]))
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(brain)), int(missing_rng.integers(4)))


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    size = int(prediction.sum()) + int(target.sum())
    return 1.0 if size == 0 else 2.0 * np.count_nonzero(prediction & target) / size


def hd95_score(prediction: np.ndarray, target: np.ndarray, spacing) -> tuple[float, str]:
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
    if not prediction.any() and not target.any():
        return 0.0, "both_empty"
    if not prediction.any() or not target.any():
        return float(np.linalg.norm((np.asarray(target.shape) - 1) * np.asarray(spacing))), "one_empty_fov_penalty"
    structure = generate_binary_structure(3, 1)
    pred_surface = prediction ^ binary_erosion(prediction, structure=structure, border_value=0)
    target_surface = target ^ binary_erosion(target, structure=structure, border_value=0)
    a_to_b = distance_transform_edt(~target_surface, sampling=spacing)[pred_surface]
    b_to_a = distance_transform_edt(~pred_surface, sampling=spacing)[target_surface]
    return float(max(np.percentile(a_to_b, 95), np.percentile(b_to_a, 95))), "defined"


@torch.inference_mode()
def infer_volume(model, case: dict, condition: str, device, batch_size: int,
                 amp: bool, probability_path: Path | None = None) -> tuple[np.ndarray, dict]:
    model.eval()
    shape = case["segmentation"].shape
    prediction = np.empty(shape, dtype=np.uint8)
    probabilities = None
    if probability_path is not None:
        probabilities = np.lib.format.open_memmap(probability_path, mode="w+", dtype=np.float16, shape=(4, *shape))
    missing = None if condition == "complete" else condition
    # Exclude a warm-up forward pass, file loading and file writing from timing.
    warm = torch.from_numpy(build_2d_input(case["volumes"], 0, missing)[None]).to(device)
    with amp_context(device, amp):
        model(warm)
    del warm
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = 0.0
    for start in range(0, shape[2], batch_size):
        zs = list(range(start, min(start + batch_size, shape[2])))
        sync(device)
        begin = time.perf_counter()
        inputs = torch.from_numpy(np.stack([build_2d_input(case["volumes"], z, missing) for z in zs])).to(device)
        with amp_context(device, amp):
            logits = model(inputs)
        hard = logits.argmax(1).cpu().numpy().astype(np.uint8)
        probs = logits.float().softmax(1).cpu().numpy() if probabilities is not None else None
        sync(device)
        elapsed += time.perf_counter() - begin
        for i, z in enumerate(zs):
            prediction[:, :, z] = hard[i]
            if probabilities is not None:
                probabilities[:, :, :, z] = probs[i]
        del inputs, logits, hard, probs
    if probabilities is not None:
        probabilities.flush()
        del probabilities
    return prediction, {"inference_seconds": elapsed,
                        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0}


def save_prediction(case_dir: Path, output: Path, prediction, case) -> None:
    import nibabel as nib
    header = nib.load(str(case_paths(case_dir)["seg"])).header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(prediction, case["affine"], header), output / "segmentation.nii.gz")
    for region, labels in REGIONS.items():
        mask = np.isin(prediction, labels).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask, case["affine"], header.copy()), output / f"{region}.nii.gz")


def evaluate_rows(model, rows: list[dict], device, batch_size: int, amp: bool,
                  model_name: str, seed: int, calculate_hd95: bool = False,
                  save_root: Path | None = None, save_probabilities: bool = False, plot_cases: int = 0) -> list[dict]:
    scores = []
    for index, row in enumerate(rows, 1):
        case_dir = Path(row["case_dir"])
        case = load_case(case_dir)
        for condition in CONDITIONS:
            target_dir = None
            if save_root is not None:
                target_dir = save_root / f"{model_name}_seed{seed}" / row["case_id"] / condition
                target_dir.mkdir(parents=True, exist_ok=False)
            probability_path = target_dir / "probabilities.npy" if save_probabilities and target_dir else None
            prediction, resource = infer_volume(model, case, condition, device, batch_size, amp, probability_path)
            for region, labels in REGIONS.items():
                pred_mask = np.isin(prediction, labels)
                true_mask = np.isin(case["segmentation"], labels)
                hd95, flag = hd95_score(pred_mask, true_mask, case["spacing_mm"]) if calculate_hd95 else (None, "not_computed_for_validation")
                scores.append({"model": model_name, "seed": seed, "patient_id": row["patient_id"],
                               "case_id": row["case_id"], "condition": condition, "region": region,
                               "dice": dice_score(pred_mask, true_mask), "hd95_mm": hd95,
                               "hd95_status": flag, **resource, "parameter_count": count_parameters(model)})
            if target_dir is not None:
                save_prediction(case_dir, target_dir, prediction, case)
                if index <= plot_cases:
                    plot_prediction_comparison(case, prediction, target_dir / "qualitative_comparison.png",
                                               f"{model_name}, seed {seed}, {row['case_id']}, {condition}", condition)
                write_json(target_dir / "metadata.json", {"case_id": row["case_id"], "patient_id": row["patient_id"],
                           "condition": condition, "modalities": MODALITIES, "affine": case["affine"].tolist(),
                           "spacing_mm": list(case["spacing_mm"]), "probability_axis_order": "class,H,W,D",
                           "probability_dtype": "float16" if save_probabilities else None,
                           "metric_policy": METRIC_POLICY})
        if index % 10 == 0 or index == len(rows):
            print(f"{model_name} seed {seed}: evaluated {index}/{len(rows)} scans", flush=True)
    return scores


def mean_patient_dice(rows: list[dict]) -> float:
    patients = defaultdict(list)
    for row in rows:
        patients[row["patient_id"]].append(float(row["dice"]))
    if not patients:
        raise ValueError("No evaluation scores")
    return float(np.mean([np.mean(values) for values in patients.values()]))


def train_one(args, rows: list[dict], model_name: str, seed: int, weight: float,
              directory: Path, summary: dict, device) -> dict:
    directory.mkdir(parents=True, exist_ok=args.resume)
    if (directory / "config.json").is_file():
        previous = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        validate_resume_config(previous, args, model_name, seed, weight, summary)
        if (directory / "run_summary.json").is_file():
            print(f"Skipping completed run {directory.name}", flush=True)
            return json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    set_seed(seed)
    model = build_model(model_name, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_rows = [r for r in rows if r["split"] == "train"]
    validation_rows = [r for r in rows if r["split"] == "validation"]
    dataset = PatientSliceDataset(train_rows, seed, args.slices_per_patient, args.cache_cases, not args.no_augmentation)
    # No subprocess workers makes epoch-specific sampling reliable on Windows.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    config = {"model": model_name, "seed": seed, "consistency_weight": weight,
              "architecture": type(model).__name__, "fusion_mode": getattr(model, "fusion_mode", "early_fusion"),
              "fusion_channels": getattr(model, "fusion_channels", None),
              "attention_grid_stride": getattr(model, "grid_stride", None),
              "objective_kind": objective_kind(model_name),
              "base_channels": args.base_channels, "manifest_sha256": summary["manifest_sha256"],
              "split_sha256": summary["split_sha256"], "sampling_fallbacks": dataset.sampling_fallbacks,
              "metric_policy": METRIC_POLICY, "arguments": vars_json(args)}
    write_json(directory / "config.json", config)
    best, non_improving, total_steps, training_seconds = -float("inf"), 0, 0, 0.0
    history = []
    run_begin = time.perf_counter()
    start_epoch, prior_wall_seconds = 1, 0.0
    if args.resume and (directory / "last.pt").is_file():
        last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        validate_resume_config(last["config"], args, model_name, seed, weight, summary)
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer_state"])
        scaler.load_state_dict(last["scaler_state"])
        history, best = last["history"], last["best"]
        non_improving, total_steps = last["non_improving"], last["total_steps"]
        training_seconds, prior_wall_seconds = last["training_seconds"], last["wall_seconds"]
        start_epoch = last["epoch"] + 1
        restore_rng_state(last["rng_state"])
        del last
        print(f"Resuming {directory.name} at epoch {start_epoch}", flush=True)
    epoch = start_epoch - 1
    end_epoch = epoch if non_improving >= args.patience else args.epochs
    for epoch in range(start_epoch, end_epoch + 1):
        dataset.set_epoch(epoch)
        model.train()
        sync(device)
        begin = time.perf_counter()
        totals, sample_count = defaultdict(float), 0
        print(f"{directory.name}: starting epoch {epoch}/{args.epochs}, {len(loader)} batches", flush=True)
        for batch_index, (inputs, target, brain, missing) in enumerate(loader, 1):
            inputs, target, brain = inputs.to(device), target.to(device), brain.to(device)
            missing = missing.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp):
                full_logits = model(inputs)
                missing_logits = model(drop_one_modality(inputs, missing)) if objective_kind(model_name) != "M0" else None
                loss, components = model_objective(model_name, full_logits, target, missing_logits,
                                                   brain, consistency_weight=weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, step {total_steps}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for key, value in components.items():
                totals[key] += float(value) * inputs.shape[0]
            sample_count += inputs.shape[0]
            total_steps += 1
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
                print(f"{directory.name}: epoch {epoch}, batch {batch_index}/{len(loader)}, loss {float(loss.detach()):.4f}", flush=True)
        sync(device)
        duration = time.perf_counter() - begin
        training_seconds += duration
        epoch_row = {"epoch": epoch, "steps": total_steps, "training_seconds": duration,
                     **{k: v / sample_count for k, v in totals.items()}}
        # Always validate at the last epoch so a short smoke run has a checkpoint.
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            print(f"{directory.name}: validating complete + 4 missing conditions", flush=True)
            scores = evaluate_rows(model, validation_rows, device, args.batch_size, amp, model_name, seed)
            score = mean_patient_dice(scores)
            epoch_row["validation_mean_dice"] = score
            if score > best:
                best, non_improving = score, 0
                atomic_torch_save({"model_state": model.state_dict(), "config": config, "epoch": epoch,
                            "validation_mean_dice": score}, directory / "best.pt")
                write_csv(directory / "best_validation_scores.csv", scores)
            else:
                non_improving += 1
        history.append(epoch_row)
        atomic_torch_save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                           "scaler_state": scaler.state_dict(), "config": config, "epoch": epoch,
                           "history": history, "best": best, "non_improving": non_improving,
                           "total_steps": total_steps, "training_seconds": training_seconds,
                           "wall_seconds": prior_wall_seconds + time.perf_counter() - run_begin,
                           "rng_state": capture_rng_state()}, directory / "last.pt")
        write_json(directory / "history.json", history)
        print(f"{directory.name}: epoch {epoch}/{args.epochs}, loss {epoch_row['loss']:.4f}, best val Dice {best:.4f}", flush=True)
        if non_improving >= args.patience:
            break
    report = {"model": model_name, "seed": seed, "consistency_weight": weight,
              "best_validation_mean_dice": best, "epochs_completed": epoch,
              "optimizer_steps": total_steps, "training_gpu_hours": training_seconds / 3600 if device.type == "cuda" else 0.0,
              "training_wall_seconds": training_seconds, "total_wall_seconds": prior_wall_seconds + time.perf_counter() - run_begin,
              "device": str(device), "checkpoint": str((directory / "best.pt").resolve()),
              "parameter_count": count_parameters(model)}
    write_json(directory / "run_summary.json", report)
    del loader, dataset, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def vars_json(args) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not callable(v)}


def train_command(args) -> None:
    rows, summary = load_manifest(args.eda_dir, args.allow_partial_audit)
    device = choose_device(args.device)
    output = args.output.expanduser().resolve() if args.resume else fresh_output(args.output)
    if args.resume:
        if not (output / "experiment.json").is_file():
            raise ValueError("--resume requires an existing experiment.json")
        previous = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
        for key in ("manifest_sha256", "split_sha256"):
            if previous.get(key) != summary[key]:
                raise ValueError(f"Resumed experiment changed {key}")
        for key in ("models", "seeds", "lambda_grid", "consistency_weight", "tuning_seed",
                    "epochs", "batch_size", "slices_per_patient", "validate_every", "patience",
                    "learning_rate", "weight_decay", "base_channels", "cache_cases",
                    "no_augmentation", "no_amp", "device"):
            if previous["arguments"].get(key) != vars_json(args).get(key):
                raise ValueError(f"Resumed experiment changed {key}")
    if summary.get("grouping_source") == "filename_inference":
        print("Patient groups use filename inference. Confirm identity mapping before reporting study results.", flush=True)
    write_json(output / "experiment.json", {"status": "training", "arguments": vars_json(args),
               "manifest_sha256": summary["manifest_sha256"], "split_sha256": summary["split_sha256"],
               "full_audit": summary["full_audit"], "grouping_source": summary.get("grouping_source"),
               "torch_version": str(torch.__version__), "numpy_version": np.__version__, "device": str(device),
               "metric_policy": METRIC_POLICY, "test_data_used_during_training": False})
    weights, tuning = {}, []
    consistency_models = [m for m in args.models if objective_kind(m) == "M2"]
    if consistency_models and args.consistency_weight is None:
        (output / "tuning").mkdir(exist_ok=args.resume)
    for model_name in consistency_models:
        if args.consistency_weight is not None:
            weights[model_name] = float(args.consistency_weight)
            continue
        candidates = []
        for value in args.lambda_grid:
            candidates.append(train_one(args, rows, model_name, args.tuning_seed, value,
                              output / "tuning" / f"{model_name}_lambda{value:g}", summary, device))
        # Tune each architecture separately. Ties select the smaller lambda.
        selected = max(candidates, key=lambda r: (r["best_validation_mean_dice"], -r["consistency_weight"]))
        weights[model_name] = selected["consistency_weight"]
        tuning.extend(candidates)
    write_json(output / "lambda_selection.json", {"selected_weights": weights,
               "selected_weight": weights.get("M2"), "tuning_seed": args.tuning_seed,
               "source": "validation_grid" if tuning else "explicit_or_no_consistency", "candidates": tuning,
               "selection_metric": "equal patient mean Dice across 3 regions and all 5 input conditions",
               "tuning_wall_seconds": sum(r["total_wall_seconds"] for r in tuning)})
    (output / "final_runs").mkdir(exist_ok=args.resume)
    runs = []
    for model_name in args.models:
        for seed in args.seeds:
            runs.append(train_one(args, rows, model_name, seed, weights.get(model_name, 0.0),
                                  output / "final_runs" / f"{model_name}_seed{seed}", summary, device))
            write_json(output / "runs.json", runs)
    experiment = json.loads((output / "experiment.json").read_text())
    experiment.update(status="complete", selected_consistency_weights=weights,
                      selected_consistency_weight=weights.get("M2"), models=args.models, seeds=args.seeds,
                      number_of_final_runs=len(runs), tuning_runs=len(tuning))
    write_json(output / "experiment.json", experiment)
    print(f"Training complete. Test data remain held out. Saved runs: {output}")


def patient_metric_rows(scan_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in scan_rows:
        key = tuple(row[k] for k in ("model", "seed", "patient_id", "condition", "region"))
        groups[key].append(row)
    return [{"model": k[0], "seed": k[1], "patient_id": k[2], "condition": k[3], "region": k[4],
             "scan_count": len(v), "dice": float(np.mean([r["dice"] for r in v])),
             "hd95_mm": float(np.mean([r["hd95_mm"] for r in v]))} for k, v in sorted(groups.items())]


def bootstrap_interval(values: np.ndarray, resamples: int, seed: int) -> dict:
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap needs finite patient contrasts")
    if len(values) < 2:
        return {"mean_difference": float(values.mean()), "ci95_lower": None, "ci95_upper": None,
                "patient_count": len(values), "bootstrap_resamples": 0, "bootstrap_seed": seed,
                "ci_status": "insufficient_patients", "scope": "between-patient uncertainty requires at least two patients"}
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    # One resample at a time avoids O(resamples * patients) allocation.
    for i in range(resamples):
        estimates[i] = values[rng.integers(len(values), size=len(values))].mean()
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {"mean_difference": float(values.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper),
            "patient_count": len(values), "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            "ci_status": "computed", "scope": "conditional on this patient split and trained seeds"}


def paired_contrast(patient_rows, left: str, right: str, conditions, seeds, resamples, bootstrap_seed,
                    regions=tuple(REGIONS)) -> tuple[dict, list[dict]]:
    index = {}
    patients = sorted({r["patient_id"] for r in patient_rows})
    for row in patient_rows:
        key = (row["model"], int(row["seed"]), row["patient_id"], row["condition"], row["region"])
        if key in index:
            raise ValueError(f"Duplicate patient metric: {key}")
        index[key] = float(row["dice"])
    contrasts = []
    for patient in patients:
        values = []
        for seed in seeds:
            for condition in conditions:
                for region in regions:
                    a, b = (left, seed, patient, condition, region), (right, seed, patient, condition, region)
                    if a not in index or b not in index:
                        raise ValueError(f"Unmatched paired result: {a}, {b}")
                    values.append(index[a] - index[b])
        contrasts.append({"patient_id": patient, "mean_dice_difference": float(np.mean(values))})
    report = bootstrap_interval(np.array([r["mean_dice_difference"] for r in contrasts]), resamples, bootstrap_seed)
    report.update(comparison=f"{left}-{right}", conditions=list(conditions), regions=list(regions), seeds=list(seeds))
    return report, contrasts


def evaluate_command(args) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    experiment = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    if experiment.get("status") != "complete":
        raise ValueError("Training experiment is incomplete")
    rows, summary = load_manifest(args.eda_dir, allow_partial=not experiment["full_audit"])
    for key in ("manifest_sha256", "split_sha256"):
        if summary[key] != experiment[key]:
            raise ValueError("Evaluation EDA manifest/split differs from the training experiment")
    device = choose_device(args.device)
    output = fresh_output(args.output)
    save_root = output / "predictions" if not args.no_save_predictions or args.save_probabilities else None
    if save_root:
        save_root.mkdir()
    test_rows = [r for r in rows if r["split"] == "test"]
    runs = json.loads((run_dir / "runs.json").read_text(encoding="utf-8"))
    actual = {(r["model"], int(r["seed"])) for r in runs}
    expected = {(m, int(s)) for m in experiment["models"] for s in experiment["seeds"]}
    if actual != expected or len(runs) != len(actual):
        raise ValueError("Missing or duplicate final runs")
    scan_scores = []
    for run in runs:
        # Load only trusted checkpoints created by this local training script.
        checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        for key in ("manifest_sha256", "split_sha256"):
            if config[key] != summary[key]:
                raise ValueError("Checkpoint data provenance mismatch")
        if config["model"] != run["model"] or config["seed"] != run["seed"]:
            raise ValueError("Checkpoint model/seed mismatch")
        model = build_model(config["model"], base_channels=config["base_channels"])
        model.load_state_dict(checkpoint["model_state"])
        del checkpoint
        model = model.to(device)
        scan_scores.extend(evaluate_rows(model, test_rows, device, args.batch_size, not args.no_amp,
                                         run["model"], run["seed"], True, save_root, args.save_probabilities, args.plot_cases))
        write_csv(output / "scan_scores.csv", scan_scores)
        del model
    patient_scores = patient_metric_rows(scan_scores)
    write_csv(output / "patient_scores.csv", patient_scores)
    means = defaultdict(list)
    for row in patient_scores:
        means[(row["model"], row["condition"], row["region"])].append(row)
    write_csv(output / "metric_summary.csv", [{"model": k[0], "condition": k[1], "region": k[2],
              "mean_patient_dice_across_seeds": float(np.mean([r["dice"] for r in v])),
              "mean_patient_hd95_mm_across_seeds": float(np.mean([r["hd95_mm"] for r in v])),
              "patient_count": len({r["patient_id"] for r in v})} for k, v in sorted(means.items())])
    report = {"metric_policy": METRIC_POLICY, "full_audit": experiment["full_audit"],
              "manifest_sha256": summary["manifest_sha256"], "split_sha256": summary["split_sha256"],
              "test_scans": len(test_rows), "test_patient_groups": len({r["patient_id"] for r in test_rows}),
              "models": experiment["models"], "seeds": experiment["seeds"],
              "inference_timing": "excludes loading, warmup, metrics and file writing; includes input construction and device transfer",
              "inference_precision": "CUDA mixed precision unless disabled; probabilities stored float16 when requested",
              "hd95_one_empty_count": sum(r["hd95_status"] == "one_empty_fov_penalty" for r in scan_scores)}
    if {"M1", "M2"} <= set(experiment["models"]):
        primary, contrasts = paired_contrast(patient_scores, "M2", "M1", MODALITIES, experiment["seeds"],
                                            args.bootstrap_resamples, args.bootstrap_seed)
        primary["interpretation"] = ("insufficient_patients" if primary["ci95_lower"] is None else
                                     "supports_improvement" if primary["ci95_lower"] > 0 else
                                     "supports_harm" if primary["ci95_upper"] < 0 else "benefit_uncertain")
        primary["proposal_three_seed_protocol"] = sorted(experiment["seeds"]) == [11, 22, 33]
        report["primary_missing_input_M2_minus_M1"] = primary
        write_csv(output / "primary_patient_contrasts.csv", contrasts)
        complete, _ = paired_contrast(patient_scores, "M2", "M1", ("complete",), experiment["seeds"],
                                     args.bootstrap_resamples, args.bootstrap_seed)
        complete.update(noninferiority_margin=-0.02,
                        noninferiority_supported=complete["ci95_lower"] > -0.02 if complete["ci95_lower"] is not None else None)
        report["complete_input_M2_minus_M1"] = complete
        secondary = []
        for region in REGIONS:
            for condition in MODALITIES:
                result, _ = paired_contrast(patient_scores, "M2", "M1", (condition,), experiment["seeds"],
                                            args.bootstrap_resamples, args.bootstrap_seed, (region,))
                result["analysis_role"] = "descriptive_secondary"
                secondary.append(result)
        report["region_condition_secondary_M2_minus_M1"] = secondary
    if {"M0", "M1"} <= set(experiment["models"]):
        secondary, _ = paired_contrast(patient_scores, "M1", "M0", MODALITIES, experiment["seeds"],
                                      args.bootstrap_resamples, args.bootstrap_seed)
        secondary["analysis_role"] = "descriptive_secondary"
        report["secondary_missing_input_M1_minus_M0"] = secondary
    # These extensions do not replace the proposal's original primary contrast.
    # Attention vs mean fusion isolates the added attention module; A2 vs M2
    # changes stems and fusion together and must not be called a pure attention effect.
    for left, right, key, question in (
        ("A1", "A0", "attention_A1_minus_A0", "Effect of adding cross-modal attention to matched modality stems"),
        ("A2", "A1", "consistency_in_attention_A2_minus_A1", "Effect of consistency within the attention architecture"),
        ("A2", "M2", "overall_architecture_A2_minus_M2", "Combined modality-stem and attention change versus original M2"),
    ):
        if {left, right} <= set(experiment["models"]):
            missing, contrasts = paired_contrast(patient_scores, left, right, MODALITIES, experiment["seeds"],
                                                 args.bootstrap_resamples, args.bootstrap_seed)
            complete, _ = paired_contrast(patient_scores, left, right, ("complete",), experiment["seeds"],
                                          args.bootstrap_resamples, args.bootstrap_seed)
            report[key] = {"question": question, "analysis_role": "exploratory_subset_comparison_unadjusted",
                           "missing_inputs": missing, "complete_inputs": complete,
                           "caution": "Predeclare the extension before opening test results. No multiplicity-adjusted success claim."}
            write_csv(output / f"{key}_patient_contrasts.csv", contrasts)
    write_json(output / "evaluation_summary.json", report)
    print(json.dumps({k: report[k] for k in ("test_scans", "test_patient_groups", "models", "seeds")}, indent=2))
    print(f"Saved test results in {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train and tune using train/validation only")
    train.add_argument("--eda-dir", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--models", nargs="+", choices=list(MODEL_OBJECTIVES), default=["A0", "A1"],
                       help="Original study M0/M1/M2; feedback ablation A0/A1/A2; defaults retain the original study")
    train.add_argument("--seeds", nargs="+", type=int, default=[11])
    train.add_argument("--lambda-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    train.add_argument("--consistency-weight", type=float, help="Explicit lambda, bypassing validation grid search")
    train.add_argument("--tuning-seed", type=int, default=11)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--slices-per-patient", type=int, default=16)
    train.add_argument("--validate-every", type=int, default=2)
    train.add_argument("--patience", type=int, default=5, help="Non-improving validation checks")
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--base-channels", type=int, default=16)
    train.add_argument("--cache-cases", type=int, default=2)
    train.add_argument("--allow-partial-audit", action="store_true", help="Explicit pipeline demonstration on a sample")
    train.add_argument("--no-augmentation", action="store_true")
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--resume", action="store_true", help="Resume identical settings from last.pt, skipping completed runs")
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.set_defaults(handler=train_command)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate frozen models on held-out test patients")
    evaluate.add_argument("--eda-dir", type=Path, required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--bootstrap-resamples", type=int, default=2000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=626)
    evaluate.add_argument("--save-probabilities", action="store_true")
    evaluate.add_argument("--no-save-predictions", action="store_true")
    evaluate.add_argument("--plot-cases", type=int, default=1,
                          help="Save expert/prediction/error figures for the first N test scans per run; 0 skips plotting")
    evaluate.add_argument("--no-amp", action="store_true")
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate.set_defaults(handler=evaluate_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Reject meaningless protocol parameters before opening any output directory.
    for key in ("batch_size", "epochs", "slices_per_patient", "validate_every", "patience", "base_channels", "cache_cases", "bootstrap_resamples"):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.command == "train":
        if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
            parser.error("Models and seeds must be unique")
        if any(s < 0 for s in [*args.seeds, args.tuning_seed]):
            parser.error("Seeds must be nonnegative")
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
            parser.error("Learning rate must be positive and weight decay nonnegative")
        if any(not np.isfinite(v) or v < 0 for v in args.lambda_grid) or len(set(args.lambda_grid)) != len(args.lambda_grid):
            parser.error("Lambda grid must contain unique finite nonnegative values")
        if args.consistency_weight is not None and (not np.isfinite(args.consistency_weight) or args.consistency_weight < 0):
            parser.error("Consistency weight must be finite and nonnegative")
    elif args.bootstrap_seed < 0:
        parser.error("Bootstrap seed must be nonnegative")
    if args.command == "evaluate" and args.plot_cases < 0:
        parser.error("--plot-cases must be nonnegative")
    try:
        args.handler(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
