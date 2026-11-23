"""QC and three-slice input preparation for the fixed 100-group AMCA study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


MODALITIES = ("t1n", "t1c", "t2w", "t2f")
MODALITY_NAMES = {"t1n": "T1", "t1c": "T1ce", "t2w": "T2", "t2f": "FLAIR"}
SUFFIXES = (*MODALITIES, "seg")
REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
CASE_PATTERN = re.compile(r"^BraTS-GLI-(\d+)-(\d+)$")
SPLITS = ("train", "validation", "test")


def case_paths(case_dir: Path) -> dict[str, Path]:
    """Return the five expected NIfTI paths for a BraTS GLI scan directory."""
    case_dir = Path(case_dir)
    return {name: case_dir / f"{case_dir.name}-{name}.nii.gz" for name in SUFFIXES}


def find_cases(data_root: Path) -> list[Path]:
    """Accept a scan directory, training directory, or its immediate parent."""
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError(f"Data root is not a directory: {data_root}")
    if CASE_PATTERN.fullmatch(data_root.name):
        return [data_root]
    direct = [p for p in data_root.iterdir() if p.is_dir() and CASE_PATTERN.fullmatch(p.name)]
    if direct:
        cases = sorted(direct, key=lambda p: p.name)
    else:
        cases = sorted(
            (p for folder in sorted(data_root.iterdir()) if folder.is_dir()
             for p in folder.iterdir() if p.is_dir() and CASE_PATTERN.fullmatch(p.name)),
            key=lambda p: (p.name, str(p)),
        )
    names = [p.name for p in cases]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate case_id folders found; use one training data directory.")
    return cases


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    """Z-score each modality's original nonzero voxels; preserve zero background.

    Statistics use float64 for stability. A constant foreground has zero
    normalized intensity; the separately constructed raw brain mask still
    records it. All-zero arrays normalize to zeros (QC rejects all-zero MRI).
    """
    volume = np.asarray(volume)
    if not np.isfinite(volume).all():
        raise ValueError("Cannot normalize a volume with nonfinite values")
    mask = volume != 0
    result = np.zeros(volume.shape, dtype=np.float32)
    if mask.any():
        values = volume[mask].astype(np.float64)
        mean = float(values.mean())
        scale = max(float(values.std()), 1e-6)
        result[mask] = ((values - mean) / scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite values after normalization")
    return result


def build_25d_input(volumes: dict[str, np.ndarray], z: int,
                    missing: str | None = None) -> np.ndarray:
    """Build (16,H,W), ordered by modality then z-1/z/z+1, then four flags.

    At volume boundaries the closest valid slice is replicated. The target
    and the consistency brain mask always refer to the central slice z.
    """
    if missing is not None and missing not in MODALITIES:
        raise ValueError(f"Unknown missing modality: {missing}")
    absent = [name for name in MODALITIES if name not in volumes]
    if absent:
        raise ValueError(f"Missing volume keys: {', '.join(absent)}")
    shape = np.asarray(volumes[MODALITIES[0]]).shape
    if len(shape) != 3 or any(d <= 0 for d in shape):
        raise ValueError(f"Expected nonempty 3D volumes, got {shape}")
    if any(np.asarray(volumes[name]).shape != shape for name in MODALITIES):
        raise ValueError("All four modalities must have the same 3D shape")
    if isinstance(z, bool) or not isinstance(z, (int, np.integer)) or not 0 <= int(z) < shape[2]:
        raise ValueError(f"Slice index {z} outside 0..{shape[2] - 1}")
    channels = []
    neighbors = [max(0, min(shape[2] - 1, int(z) + offset)) for offset in (-1, 0, 1)]
    for name in MODALITIES:
        for neighbor in neighbors:
            channel = (np.zeros(shape[:2], dtype=np.float32) if name == missing
                       else np.asarray(volumes[name][:, :, neighbor], dtype=np.float32))
            if not np.isfinite(channel).all():
                raise ValueError(f"Nonfinite input values in {name} at slice {neighbor}")
            channels.append(channel)
    channels.extend(np.full(shape[:2], float(name != missing), dtype=np.float32)
                    for name in MODALITIES)
    return np.stack(channels, axis=0)


def load_case(case_dir: Path, *, include_statistics: bool = False) -> dict[str, Any]:
    """Load and validate a complete scan; return normalized images and geometry.

    The brain mask is the union of RAW modality nonzero voxels. It is fixed
    before normalization or modality dropout and must be used unchanged for
    complete and missing-modality conditions. Invalid labels are checked
    before int64 conversion, so fractional labels cannot be silently rounded.
    Raw intensity statistics are EDA-only and are skipped by default to avoid
    redundant foreground extraction and percentile calculations in training.
    """
    import nibabel as nib

    paths = case_paths(Path(case_dir))
    absent = [name for name, path in paths.items() if not path.is_file()]
    if absent:
        raise ValueError(f"Missing files: {', '.join(absent)}")
    reference_shape = None
    reference_affine = None
    reference_spacing = None
    reference_orientation = None
    normalized = {}
    brain_mask = None
    segmentation = None
    raw_statistics = {}
    for name, path in paths.items():
        nii = nib.load(str(path))
        shape = tuple(int(value) for value in nii.shape)
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError(f"{name}: expected nonempty 3D volume, got {shape}")
        spacing = tuple(float(value) for value in nii.header.get_zooms()[:3])
        if len(spacing) != 3 or not np.isfinite(spacing).all() or any(value <= 0 for value in spacing):
            raise ValueError(f"{name}: spacing must be finite and positive")
        affine = np.asarray(nii.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError(f"{name}: invalid affine matrix")
        try:
            inverse = np.linalg.inv(affine)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name}: singular affine matrix") from exc
        if not np.isfinite(inverse).all() or abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12:
            raise ValueError(f"{name}: non-invertible voxel geometry")
        orientation = nib.aff2axcodes(affine)
        if any(code is None for code in orientation):
            raise ValueError(f"{name}: invalid affine orientation")
        if reference_shape is None:
            reference_shape = shape
            reference_affine = affine.copy()
            reference_spacing = spacing
            reference_orientation = orientation
            brain_mask = np.zeros(shape, dtype=bool)
        elif shape != reference_shape:
            raise ValueError(f"{name}: shape mismatch {shape} vs {reference_shape}")
        elif not np.allclose(spacing, reference_spacing, rtol=1e-5, atol=1e-3):
            raise ValueError(f"{name}: voxel spacing mismatch")
        elif orientation != reference_orientation or not np.allclose(
                affine, reference_affine, rtol=1e-5, atol=1e-3):
            raise ValueError(f"{name}: affine/orientation mismatch")
        data = np.asarray(nii.dataobj)
        if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            raise ValueError(f"{name}: expected real numeric voxel data")
        if not np.isfinite(data).all():
            raise ValueError(f"{name}: nonfinite voxel values")
        if name == "seg":
            labels = np.unique(data)
            if not np.isin(labels, (0, 1, 2, 3)).all():
                raise ValueError(f"seg: labels outside {{0,1,2,3}}: {labels.tolist()}")
            segmentation = data.astype(np.int64)
        else:
            mask = data != 0
            if not mask.any():
                raise ValueError(f"{name}: all-zero modality")
            brain_mask |= mask
            if include_statistics:
                values = data[mask].astype(np.float64)
                quantiles = np.percentile(values, (1, 50, 99))
                raw_statistics[name] = {
                    "nonzero_voxels": int(values.size), "mean": float(values.mean()),
                    "std": float(values.std()), "p01": float(quantiles[0]),
                    "median": float(quantiles[1]), "p99": float(quantiles[2]),
                }
            normalized[name] = normalize_nonzero(data)
    return {"volumes": normalized, "segmentation": segmentation,
            "brain_mask": brain_mask, "spacing_mm": reference_spacing,
            "affine": reference_affine, "raw_statistics": raw_statistics}


def inferred_patient_id(case_id: str) -> str:
    return case_id.rsplit("-", 1)[0]


def read_patient_map(path: Path | None, cases: list[Path]) -> dict[str, str]:
    """Require one nonempty mapping per discovered case; reject duplicates."""
    if path is None:
        return {case.name: inferred_patient_id(case.name) for case in cases}
    mapping = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"case_id", "patient_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Patient map must contain case_id and patient_id columns")
        for line_number, row in enumerate(reader, 2):
            case_id = (row.get("case_id") or "").strip()
            patient = (row.get("patient_id") or "").strip()
            if not case_id or not patient:
                raise ValueError(f"Patient map row {line_number}: empty case_id or patient_id")
            if case_id in mapping:
                raise ValueError(f"Patient map contains duplicate case_id: {case_id}")
            mapping[case_id] = patient
    missing = sorted(case.name for case in cases if case.name not in mapping)
    if missing:
        raise ValueError(f"Patient map missing {len(missing)} case(s): {', '.join(missing[:10])}")
    return mapping


def inventory_row(case_dir: Path, patient: str | None = None) -> dict[str, Any]:
    row = {"case_id": case_dir.name,
           "patient_id": patient if patient is not None else inferred_patient_id(case_dir.name),
           "case_dir": str(case_dir.resolve())}
    for name, path in case_paths(case_dir).items():
        row[f"has_{name}"] = path.is_file()
        row[f"{name}_bytes"] = path.stat().st_size if path.is_file() else 0
    row["complete_files"] = all(row[f"has_{name}"] for name in SUFFIXES)
    return row


def summarize_modalities(inventory: list[dict]) -> dict[str, Any]:
    """Count acquired 3D MRI volumes and natural absence, not synthetic views."""
    total = len(inventory)
    per_modality = {
        name: {"available_scan_examinations": sum(bool(row[f"has_{name}"]) for row in inventory),
               "naturally_missing_scan_examinations": sum(not row[f"has_{name}"] for row in inventory)}
        for name in MODALITIES
    }
    return {"scan_examinations": total,
            "patient_groups": len({row["patient_id"] for row in inventory}),
            "mri_3d_volume_files": sum(v["available_scan_examinations"] for v in per_modality.values()),
            "segmentation_3d_volume_files": sum(bool(row["has_seg"]) for row in inventory),
            "scans_with_any_naturally_missing_mri": sum(any(not row[f"has_{m}"] for m in MODALITIES) for row in inventory),
            "scans_missing_segmentation": sum(not row["has_seg"] for row in inventory),
            "per_modality": per_modality,
            "counting_unit": "acquired 3D volume files; axial slices and simulated conditions are not independent patients",
            "scope": "all inventoried folders; file presence alone is not image-level QC"}


def simulated_condition_rows(clean_rows: list[dict], split_rows: list[dict]) -> list[dict]:
    """Five evaluation views of each retained complete scan; no new scans."""
    output = []
    for partition in ("all_retained", *SPLITS):
        rows = clean_rows if partition == "all_retained" else [r for r in split_rows if r["split"] == partition]
        for condition in ("complete", *MODALITIES):
            output.append({"partition": partition, "condition": condition,
                           "scan_views": len(rows), "patient_groups": len({r["patient_id"] for r in rows}),
                           "available_modalities_per_view": 4 if condition == "complete" else 3,
                           "simulated_missing_modality": "none" if condition == "complete" else condition,
                           "source": "same retained complete scans, with input withholding only"})
    return output


def audit_case(case_dir: Path, patient: str | None = None) -> dict[str, Any]:
    """Audit one scan; no exclusions based on tumor size or ET absence."""
    row = inventory_row(case_dir, patient)
    if not row["complete_files"]:
        absent = [name for name in SUFFIXES if not row[f"has_{name}"]]
        row.update(qc_pass=False, issues=f"Missing files: {', '.join(absent)}")
        return row
    try:
        case = load_case(case_dir, include_statistics=True)
        import nibabel as nib
        segmentation = case["segmentation"]
        row["shape"] = "x".join(map(str, segmentation.shape))
        row["spacing_mm"] = "x".join(f"{value:.6g}" for value in case["spacing_mm"])
        row["orientation"] = "".join(nib.aff2axcodes(case["affine"]))
        row["labels"] = ",".join(map(str, np.unique(segmentation)))
        row["brain_voxels"] = int(case["brain_mask"].sum())
        row["tumor_voxels_outside_brain_mask"] = int(np.count_nonzero(
            (segmentation > 0) & ~case["brain_mask"]))
        for name, stats in case["raw_statistics"].items():
            row.update({f"{name}_{key}": value for key, value in stats.items()})
        voxel_ml = abs(float(np.linalg.det(case["affine"][:3, :3]))) / 1000.0
        for region, labels in REGIONS.items():
            count = int(np.count_nonzero(np.isin(segmentation, labels)))
            row[f"{region}_voxels"] = count
            row[f"{region}_ml"] = count * voxel_ml
        row["ET_present"] = row["ET_voxels"] > 0
        row.update(qc_pass=True, issues="")
    except Exception as exc:
        # Per-case I/O/header failures belong in the exclusion log; interruption
        # and process-exit exceptions still propagate because they are BaseException.
        row.update(qc_pass=False, issues=f"{type(exc).__name__}: {exc}")
    return row


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_splits(clean_rows: list[dict], seed: int = 626) -> list[dict]:
    """Freeze approximately 70/15/15 patient-group splits with ET stratification.

    Global quotas guarantee nonempty train/validation/test when there are at
    least three retained groups. A group's ET status is positive if ANY of
    its retained scans contains ET. Both ET strata appear in each partition
    when their sizes and partition quotas permit it.
    """
    groups: dict[str, list[dict]] = {}
    for row in sorted(clean_rows, key=lambda item: item["case_id"]):
        groups.setdefault(str(row["patient_id"]), []).append(row)
    n = len(groups)
    if n < 3:
        return []
    n_train = min(max(round(0.70 * n), 1), n - 2)
    n_validation = min(max(round(0.15 * n), 1), n - n_train - 1)
    quotas = np.array((n_train, n_validation, n - n_train - n_validation), dtype=int)
    positive = sorted(pid for pid, scans in groups.items()
                      if any(bool(row["ET_present"]) for row in scans))
    negative = sorted(set(groups) - set(positive))
    target_positive = quotas.astype(float) * (len(positive) / n)
    # Bounds preserve proportionality while retaining both ET strata where feasible.
    can_cover_both = len(positive) >= 3 and len(negative) >= 3 and np.all(quotas >= 2)
    lower = np.ones(3, dtype=int) if can_cover_both else np.zeros(3, dtype=int)
    upper = quotas - 1 if can_cover_both else quotas.copy()
    allocations = np.clip(np.floor(target_positive).astype(int), lower, upper)
    while allocations.sum() < len(positive):
        candidates = [i for i in range(3) if allocations[i] < upper[i]]
        index = max(candidates, key=lambda i: (target_positive[i] - allocations[i], -i))
        allocations[index] += 1
    while allocations.sum() > len(positive):
        candidates = [i for i in range(3) if allocations[i] > lower[i]]
        index = max(candidates, key=lambda i: (allocations[i] - target_positive[i], -i))
        allocations[index] -= 1
    rng = np.random.default_rng(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    assignment = {}
    positive_start = negative_start = 0
    for index, split in enumerate(SPLITS):
        npatients = int(allocations[index])
        nnegative = int(quotas[index] - npatients)
        for pid in positive[positive_start:positive_start + npatients]:
            assignment[pid] = split
        for pid in negative[negative_start:negative_start + nnegative]:
            assignment[pid] = split
        positive_start += npatients
        negative_start += nnegative
    return [{"case_id": row["case_id"], "patient_id": str(row["patient_id"]),
             "case_dir": str(Path(row["case_dir"]).resolve()),
             "split": assignment[str(row["patient_id"])],
             "ET_present": bool(row["ET_present"])}
            for row in sorted(clean_rows, key=lambda item: item["case_id"])]


def plot_region_distributions(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, region in zip(axes, REGIONS):
        values = [float(row[f"{region}_ml"]) for row in rows]
        ax.hist(values, bins=min(30, max(5, len(values) // 2)), color="#377eb8")
        ax.set(title=f"{region} volume (QC-passing scans)", xlabel="mL", ylabel="Scans")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(output / "region_volumes.png", dpi=140)
    plt.close(fig)


def plot_examples(rows: list[dict], output: Path, count: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    for row in rows[:count]:
        case = load_case(Path(row["case_dir"]))
        seg = case["segmentation"]
        wt_counts = np.count_nonzero(seg > 0, axis=(0, 1))
        z = int(np.argmax(wt_counts)) if wt_counts.any() else seg.shape[2] // 2
        image = case["volumes"]["t2f"][:, :, z].T
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        for ax in axes:
            ax.imshow(image, cmap="gray", origin="lower", vmin=-2, vmax=4)
            ax.axis("off")
        axes[0].set_title("Normalized T2-FLAIR")
        axes[1].imshow(np.ma.masked_where(seg[:, :, z].T == 0, seg[:, :, z].T),
                       cmap=ListedColormap(["red", "orange", "lime"]),
                       norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3),
                       alpha=0.45, origin="lower", interpolation="nearest")
        axes[1].set_title("1 core (red), 2 edema (orange), 3 ET (green)")
        fig.suptitle(f"{row['case_id']} | axial index {z}")
        fig.tight_layout()
        fig.savefig(output / f"overlay_{row['case_id']}.png", dpi=140)
        plt.close(fig)
        # Display one of the four simulated single-missing input conditions.
        complete = build_25d_input(case["volumes"], z)
        absent_t1c = build_25d_input(case["volumes"], z, missing="t1c")
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for row_index, (view, label) in enumerate(((complete, "Complete"), (absent_t1c, "T1ce missing"))):
            for modality_index, modality in enumerate(MODALITIES):
                ax = axes[row_index, modality_index]
                ax.imshow(view[3 * modality_index + 1].T, cmap="gray", origin="lower", vmin=-2, vmax=4)
                ax.set_title(f"{label}: {MODALITY_NAMES[modality]} | available={int(view[12 + modality_index, 0, 0])}")
                ax.axis("off")
        fig.suptitle(f"{row['case_id']} | 16 channels: 4 three-slice stacks + 4 indicators\nCentral MRI slices displayed")
        fig.tight_layout()
        fig.savefig(output / f"input_demo_{row['case_id']}.png", dpi=140)
        plt.close(fig)


def plot_problem_explanation(row: dict, output: Path) -> None:
    """Real expert labels explain four classes, composite regions and 2D."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    case = load_case(Path(row["case_dir"]))
    seg = case["segmentation"]
    z = int(np.argmax((seg > 0).sum(axis=(0, 1))))
    gray = case["volumes"]["t2f"][:, :, z].T
    colors = ["#0B1420", "#E85D68", "#E5AA3A", "#38B878"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
    axes[0, 0].imshow(gray, cmap="gray", origin="lower", vmin=-2, vmax=4)
    axes[0, 0].set_title("One FLAIR slice", fontsize=15)
    axes[0, 1].imshow(seg[:, :, z].T, cmap=ListedColormap(colors),
                      norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5], 4), origin="lower", interpolation="nearest")
    axes[0, 1].set_title("Four exclusive tissue classes", fontsize=15)
    axes[0, 2].text(.05, .93, "0  Background\n1  Necrotic core (NCR)\n2  Edematous / invaded tissue\n3  Enhancing tumor (ET)\n\nOne tissue label per voxel\nThree overlapping regions below",
                      transform=axes[0, 2].transAxes, va="top", fontsize=13, linespacing=1.5, color="#142D46")
    for ax, (region, labels), color in zip(axes[1], REGIONS.items(), ("#2AA4A4", "#E5AA3A", "#38B878")):
        ax.imshow(gray, cmap="gray", origin="lower", vmin=-2, vmax=4)
        mask = np.isin(seg[:, :, z].T, labels)
        ax.imshow(np.ma.masked_where(~mask, np.ones(mask.shape)), cmap=ListedColormap([color]),
                   vmin=0, vmax=1, origin="lower", interpolation="nearest", alpha=.55)
        ax.set_title(f"{region} = labels {','.join(map(str, labels))}", fontsize=15)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Four-class softmax produces WT, TC and ET by combining tissue labels", fontsize=17)
    fig.text(.5, .015, f"Expert annotation example, not model output. {row['case_id']}, axial index {z}", ha="center", fontsize=11)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(output / "segmentation_explained.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for column, name in enumerate(MODALITIES):
        volume = case["volumes"][name]
        for index, offset in enumerate((-1, 0, 1)):
            neighbor = max(0, min(volume.shape[2] - 1, z + offset))
            ax = axes[index, column]
            ax.imshow(volume[:, :, neighbor].T, cmap="gray", origin="lower", vmin=-2, vmax=4)
            ax.set_title(f"{MODALITY_NAMES[name]}, z={neighbor}")
            ax.axis("off")
    fig.suptitle("Three neighboring axial slices per MRI modality")
    fig.text(.5, .025, "12 MRI planes + 4 availability maps = 16 input channels. Predict center slice only.",
             ha="center")
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(output / "three_slice_explained.png", dpi=160)
    plt.close(fig)


def plot_prediction_comparison(case: dict, prediction: np.ndarray, output: Path, title: str,
                               condition: str = "complete") -> None:
    """Generate report-ready qualitative evidence from an evaluated prediction.

    This uses the supplied mask only after model selection, to choose a slice
    containing tumor for visualization. It does not compute a training signal.
    The context image always uses a modality available in the evaluated view.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    target = case["segmentation"]
    if prediction.shape != target.shape or not np.isin(prediction, (0,1,2,3)).all():
        raise ValueError("Prediction must match the target geometry and four labels")
    wt_counts = (target > 0).sum(axis=(0,1))
    z = int(np.argmax(wt_counts)) if wt_counts.any() else target.shape[2] // 2
    context = "t2f" if condition != "t2f" else "t2w"
    gray = case["volumes"][context][:,:,z].T
    cmap = ListedColormap(["#0B1420", "#E85D68", "#E5AA3A", "#38B878"])
    norm = BoundaryNorm([-.5,.5,1.5,2.5,3.5], 4)
    fig, axes = plt.subplots(1,4,figsize=(14,4.3))
    axes[0].imshow(gray,cmap="gray",origin="lower",vmin=-2,vmax=4)
    axes[0].set_title(f"Available context: {MODALITY_NAMES[context]}")
    for ax, mask, label in ((axes[1],target,"Expert labels"),(axes[2],prediction,"Model prediction")):
        ax.imshow(mask[:,:,z].T,cmap=cmap,norm=norm,origin="lower",interpolation="nearest")
        ax.set_title(label)
    axes[3].imshow(gray,cmap="gray",origin="lower",vmin=-2,vmax=4)
    error = (prediction[:,:,z] != target[:,:,z]).T
    axes[3].imshow(np.ma.masked_where(~error,np.ones(error.shape)),cmap=ListedColormap(["#EF4444"]),
                   vmin=0,vmax=1,origin="lower",alpha=.65,interpolation="nearest")
    axes[3].set_title("Red: any tissue-label error")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{title}, slice {z}",fontsize=14)
    fig.text(.5,.025,"Tissue colors: background dark, NCR red, ED orange, ET green. Displayed slice does not summarize whole-volume accuracy.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.92))
    fig.savefig(output,dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Extracted BraTS 2023 GLI training directory or parent")
    parser.add_argument("--output", type=Path, default=Path("eda_output"), help="New or empty output directory")
    parser.add_argument("--max-cases", type=int, help="Audit only the first N sorted scans; resulting split is a sample")
    parser.add_argument("--plot-cases", type=int, default=3, help="Number of QC-passing scan overlay/input examples")
    parser.add_argument("--inventory-only", action="store_true", help="Check filenames and file sizes without NIfTI QC or splitting")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting; matplotlib is not required")
    parser.add_argument("--seed", type=int, default=626)
    parser.add_argument("--patient-map", type=Path, help="Optional case_id,patient_id CSV with verified patient grouping")
    parser.add_argument("--subset-manifest", type=Path, help="Frozen selected cases from select_subset.py")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if args.plot_cases < 0:
        parser.error("--plot-cases must be nonnegative")
    try:
        cases = find_cases(args.data_root)
        if not cases:
            parser.error(f"No BraTS-GLI scan folders found under {args.data_root}")
        source_scan_count = len(cases)
        if args.subset_manifest:
            selection = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
            selected_ids = {r["case_id"] for r in selection["selected"]}
            cases = [case for case in cases if case.name in selected_ids]
            if {case.name for case in cases} != selected_ids:
                raise ValueError("Source dataset is missing selected examinations")
        patient_mapping = read_patient_map(args.patient_map, cases)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not args.inventory_only:
        try:
            import nibabel  # noqa: F401 -- fail early instead of marking every scan unreadable
            if not args.no_plots:
                import matplotlib  # noqa: F401 -- avoid finishing QC before a missing plotting dependency fails
        except ImportError as exc:
            parser.error(f"Missing EDA dependency: {exc}. Use --no-plots to skip matplotlib.")
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be absent or empty to prevent stale manifests/splits: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected = cases[:args.max_cases] if args.max_cases is not None else cases
    inventory = [inventory_row(case, patient_mapping[case.name]) for case in cases]
    inventory_path = output / "inventory.csv"
    write_csv(inventory_path, inventory)
    data_volume = summarize_modalities(inventory)
    write_csv(output / "modality_availability.csv", [
        {"modality": m, "display_name": MODALITY_NAMES[m], "total_scan_examinations": len(inventory), **data_volume["per_modality"][m]}
        for m in MODALITIES])
    incomplete = [row for row in inventory if not row["complete_files"]]
    write_csv(output / "incomplete_cases.csv", incomplete, list(inventory[0]))
    write_csv(output / "exclusions.csv", [], ["case_id", "patient_id", "case_dir", "reason"])
    full_audit = not args.inventory_only and len(selected) == len(cases)
    summary = {
        "source_data_root": str(args.data_root.expanduser().resolve()),
        "grouping_source": "patient_map" if args.patient_map else "filename_inference",
        "patient_map": str(args.patient_map.expanduser().resolve()) if args.patient_map else None,
        "filename_grouping_is_verified_identity": False if not args.patient_map else None,
        "total_scan_folders": len(cases),
        "unique_patient_ids": len({row["patient_id"] for row in inventory}),
        "scan_folders_with_all_five_files": len(cases) - len(incomplete),
        "naturally_incomplete_scans": len(incomplete),
        "audited_scan_folders": 0 if args.inventory_only else len(selected),
        "full_audit": full_audit,
        "partial_audit": not args.inventory_only and not full_audit,
        "audit_scope": "inventory_only" if args.inventory_only else "full" if full_audit else "sample",
        "retained_scan_folders": 0,
        "excluded_scan_folders": 0,
        "not_audited_scan_folders": len(cases) if args.inventory_only else len(cases) - len(selected),
        "split_seed": args.seed,
        "requested_split_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_unit": "patient_group",
        "stratification": "approximate; patient ET-positive if any retained scan has label 3",
        "manifest_file": "inventory.csv",
        "manifest_sha256": sha256_file(inventory_path),
        "manifest_hash_scope": "inventory CSV bytes (case paths, identities, file presence and sizes), not all NIfTI bytes",
        "split_file": None,
        "split_sha256": None,
        "label_codes": {"0": "background", "1": "non-enhancing tumor core", "2": "peritumoral edema", "3": "enhancing tumor"},
        "regions": REGIONS,
        "preprocessing": "per-volume modality nonzero z-score; background remains zero; no additional resampling",
        "input_channels": 16,
        "spatial_context": "three adjacent axial slices per modality; replicate boundary slices",
        "source_scan_count_before_selection": source_scan_count,
        "cohort_scope": "selected subset; full_audit refers to selected cohort only",
        "subset_manifest": str(args.subset_manifest) if args.subset_manifest else None,
        "data_volume": data_volume,
        "brain_mask": "union of raw nonzero voxels across all four modalities, fixed before dropout",
    }
    if not args.inventory_only:
        qc = []
        for index, case_dir in enumerate(selected, 1):
            qc.append(audit_case(case_dir, patient_mapping[case_dir.name]))
            if index % 25 == 0 or index == len(selected):
                print(f"Audited {index}/{len(selected)} scans", flush=True)
        write_csv(output / "qc.csv", qc)
        clean = [row for row in qc if row["qc_pass"]]
        failed = [row for row in qc if not row["qc_pass"]]
        exclusion_rows = [{"case_id": row["case_id"], "patient_id": row["patient_id"],
                           "case_dir": row["case_dir"], "reason": row["issues"]} for row in failed]
        write_csv(output / "exclusions.csv", exclusion_rows, ["case_id", "patient_id", "case_dir", "reason"])
        summary.update(retained_scan_folders=len(clean), excluded_scan_folders=len(failed),
                       qc_pass_scans=len(clean), qc_failed_scans=len(failed),
                       retained_patient_groups=len({row["patient_id"] for row in clean}),
                       ET_positive_scans=sum(bool(row["ET_present"]) for row in clean),
                       qc_issues=dict(sorted(Counter(row["issues"] for row in failed).items())))
        summary["region_volume_ml"] = {
            region: {"min": float(min(row[f"{region}_ml"] for row in clean)),
                     "median": float(np.median([row[f"{region}_ml"] for row in clean])),
                     "max": float(max(row[f"{region}_ml"] for row in clean))}
            for region in REGIONS} if clean else {}
        splits = make_splits(clean, args.seed)
        summary["retained_slice_dimensions"] = dict(sorted(Counter(row["shape"] for row in clean).items()))
        summary["simulated_conditions_scope"] = "full QC cohort" if full_audit else "partial QC sample only"
        summary["simulated_total_input_views"] = 5 * len(clean)
        summary["simulated_single_missing_views"] = 4 * len(clean)
        summary["simulated_views_are_new_acquisitions"] = False
        write_csv(output / "simulated_conditions.csv", simulated_condition_rows(clean, splits))
        if splits:
            split_path = output / "patient_splits.csv"
            write_csv(split_path, splits, ["case_id", "patient_id", "case_dir", "split", "ET_present"])
            summary.update(split_file=split_path.name, split_sha256=sha256_file(split_path),
                           split_is_sample=not full_audit,
                           split_scan_counts=dict(sorted(Counter(row["split"] for row in splits).items())),
                           split_patient_counts={split: len({row["patient_id"] for row in splits if row["split"] == split})
                                                 for split in SPLITS})
            summary["split_ET_positive_patient_counts"] = {
                split: len({row["patient_id"] for row in splits
                            if row["split"] == split and row["ET_present"]}) for split in SPLITS}
        else:
            summary["split_unavailable_reason"] = "Fewer than three QC-passing patient groups"
        if clean and not args.no_plots:
            plot_region_distributions(clean, output)
            if args.plot_cases:
                plot_examples(clean, output, args.plot_cases)
                plot_problem_explanation(clean[0], output)
    else:
        summary["split_unavailable_reason"] = "Inventory-only run has no NIfTI quality audit or ET labels"
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved EDA outputs in {output}")


if __name__ == "__main__":
    main()
