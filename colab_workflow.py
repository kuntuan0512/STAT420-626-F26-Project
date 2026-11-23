"""Colab orchestration: staging, reproducible profiles, audit reuse and commands.

The three scientific scripts remain usable on their own. This helper never
downloads MRI data; it stages the user's existing Drive archives/folder.
"""
from pathlib import Path, PurePosixPath
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

PROFILES = {
    "QUICK_CHECK": dict(models=["A0", "A1"], seeds=[11], epochs=1,
                        batch_size=2, base_channels=4, slices_per_patient=4,
                        validate_every=1, patience=5, max_cases=6, consistency_weight=None),
    "STUDY_100_2D": dict(models=["A0", "A1"], seeds=[11], epochs=10,
                         batch_size=4, base_channels=16, slices_per_patient=16,
                         validate_every=2, patience=5, max_cases=None, consistency_weight=None),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command, cwd=None):
    """Stream live progress; propagate failures rather than continuing cells."""
    command = list(map(str, command))
    print("Running:", subprocess.list2cmdline(command), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
        status = proc.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def safe_member(root, name):
    # ZIP paths are POSIX paths, irrespective of the host OS.
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    destination = (Path(root) / name).resolve()
    if not destination.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Archive member escapes staging directory: {name}")
    return destination


def stage_data(source, destination, expected_scans=1251):
    """Accept the existing 52-shard manifest, one ZIP, or an extracted folder.

    Archive SHA-256 is checked when a shard manifest supplies it. Resume
    markers check extracted file sizes; they are not voxel-content hashes.
    """
    from eda import find_cases, case_paths
    source, destination = Path(source), Path(destination)
    if not source.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {source}. Edit DATA_SOURCE in cell 2.")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "BraTS_Full_Manifest.json" if source.is_dir() else source
    if manifest_path.name == "BraTS_Full_Manifest.json" and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        archives = manifest["archives"]
        if manifest["scan_count"] != expected_scans:
            raise ValueError("Manifest scan count differs from EXPECTED_SCANS")
        items = []
        for row in archives:
            remote = safe_member(manifest_path.parent, row["name"])
            if not remote.is_file() or remote.stat().st_size != row["bytes"]:
                raise ValueError(f"Missing/incomplete archive: {remote}. Finish upload and rerun.")
            items.append((remote, row["sha256"]))
    elif source.is_file() and source.suffix.lower() == ".zip":
        items = [(source, None)]
        print("Single ZIP: archive CRC verified during extraction; no supplied SHA-256 manifest.")
    else:
        cases = find_cases(source)
        if len(cases) != expected_scans:
            candidate = source / "BraTS2023_GLI_Full_Dataset.zip"
            if candidate.is_file():
                return stage_data(candidate, destination, expected_scans)
            raise ValueError(f"Found {len(cases)} scans, expected {expected_scans}. Point DATA_SOURCE at the shard folder, ZIP, or extracted cohort.")
        print(f"Copying {len(cases)} scan folders to local runtime storage...", flush=True)
        for index, case in enumerate(cases, 1):
            target = destination / case.name
            target.mkdir(exist_ok=True)
            for src in case_paths(case).values():
                if not src.is_file():
                    continue  # EDA records naturally absent files.
                dest = target / src.name
                if not dest.is_file() or dest.stat().st_size != src.stat().st_size:
                    if shutil.disk_usage(destination).free < src.stat().st_size + 2**30:
                        raise OSError("Insufficient runtime disk space for staging")
                    temporary = dest.with_suffix(dest.suffix + ".tmp")
                    shutil.copyfile(src, temporary)
                    temporary.replace(dest)
            if index % 50 == 0 or index == len(cases):
                print(f"Staged {index}/{len(cases)} examinations", flush=True)
        return destination

    cache = destination.parent / (destination.name + "_archive_cache")
    cache.mkdir(exist_ok=True)
    for index, (remote, expected_hash) in enumerate(items, 1):
        marker = cache / f"{index:03d}.extracted.json"
        identity = dict(source=str(remote), bytes=remote.stat().st_size, sha256=expected_hash)
        if marker.is_file():
            saved = json.loads(marker.read_text())
            if saved["identity"] != identity:
                raise ValueError("Archive inputs changed; choose a new LOCAL_DATA_ROOT and STUDY_BASE.")
            if all(safe_member(destination, n).is_file() and safe_member(destination, n).stat().st_size == size
                   for n, size in saved["files"]):
                print(f"Reusing verified extraction {index}/{len(items)}", flush=True)
                continue
        print(f"Copying/checking archive {index}/{len(items)}: {remote.name}", flush=True)
        local = cache / f"{index:03d}.zip"
        if shutil.disk_usage(destination).free < remote.stat().st_size + 2**30:
            raise OSError("Insufficient runtime disk space for archive staging")
        shutil.copyfile(remote, local)
        actual_hash = digest(local)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch: {remote.name}; re-upload this shard.")
        with zipfile.ZipFile(local) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            names = [m.filename for m in members]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate entries in archive")
            for member in members:
                safe_member(destination, member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Symbolic links are not accepted in dataset archives")
            if shutil.disk_usage(destination).free < sum(m.file_size for m in members) + 2**30:
                raise OSError("Insufficient runtime disk space for extracted files")
            for member in members:
                target = safe_member(destination, member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".tmp")
                with archive.open(member) as incoming, temporary.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 8 * 2**20)
                temporary.replace(target)
        save_json(marker, dict(identity=identity, actual_sha256=actual_hash,
                               files=[[m.filename, m.file_size] for m in members]))
        local.unlink()  # Only the temporary runtime copy, never the Drive source.
        print(f"Extracted {index}/{len(items)}", flush=True)
    cases = find_cases(destination)
    if len(cases) != expected_scans:
        raise ValueError(f"Extracted {len(cases)} scan folders; expected {expected_scans}")
    return destination


def lock_study(study, code_dir, profile, settings, environment):
    """Reject accidental code/settings changes before modifying a resumed study."""
    study, code_dir = Path(study), Path(code_dir)
    study.mkdir(parents=True, exist_ok=True)
    sources = ["eda.py", "design_network.py", "train_model.py", "colab_workflow.py", "select_subset.py", "subset_selection.json"]
    record = dict(profile=profile, settings=settings,
                  source_sha256={name: digest(code_dir / name) for name in sources},
                  environment=environment)
    lock = study / "protocol.json"
    if lock.is_file():
        previous = json.loads(lock.read_text())
        if previous != record:
            raise ValueError("Study code/settings/environment changed. Restore original settings to resume, or choose a new STUDY_BASE for a new experiment.")
    else:
        save_json(lock, record)
    snapshot = study / "code"
    snapshot.mkdir(exist_ok=True)
    for name in sources + ["requirements_stat626.txt"]:
        target = snapshot / name
        if target.exists() and digest(target) != digest(code_dir / name):
            raise ValueError(f"Stored source snapshot changed: {target}")
        if not target.exists():
            shutil.copyfile(code_dir / name, target)


def audit(code_dir, data_root, study, settings, patient_map=""):
    from train_model import load_manifest
    study = Path(study)
    pointer = study / "eda_location.json"
    if pointer.is_file():
        directory = Path(json.loads(pointer.read_text())["path"])
        rows, summary = load_manifest(directory, allow_partial=settings["max_cases"] is not None)
        if summary["source_data_root"] != str(Path(data_root).resolve()):
            raise ValueError("Frozen audit uses a different dataset location")
        print("Reusing frozen QC and patient split:", directory)
        return directory, summary
    # Interrupted EDA is preserved; restart into a new audit directory.
    directory = study / ("eda_" + time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}")
    command = [sys.executable, "-u", str(Path(code_dir) / "eda.py"),
               "--data-root", str(data_root), "--output", str(directory), "--plot-cases", "1", "--subset-manifest", str(Path(code_dir) / "subset_selection.json")]
    if settings["max_cases"] is not None:
        command += ["--max-cases", str(settings["max_cases"])]
    if patient_map:
        command += ["--patient-map", str(patient_map)]
    run(command)
    rows, summary = load_manifest(directory, allow_partial=settings["max_cases"] is not None)
    if {r["split"] for r in rows} != {"train", "validation", "test"}:
        raise ValueError("Need at least three patient groups for the pipeline. Increase QUICK_CHECK max_cases in a new study.")
    if settings["max_cases"] is None and summary["retained_patient_groups"] != 100:
        raise ValueError("The 100-group study requires 100 QC-passing groups. Inspect exclusions; do not silently replace groups.")
    save_json(pointer, {"path": str(directory)})
    return directory, summary


def train_command(code_dir, eda_dir, study, settings, device="cuda"):
    output = Path(study) / "models"
    command = [sys.executable, "-u", str(Path(code_dir) / "train_model.py"), "train",
               "--eda-dir", str(eda_dir), "--output", str(output), "--device", device,
               "--models", *settings["models"], "--seeds", *map(str, settings["seeds"])]
    for key in ("epochs", "batch_size", "base_channels", "slices_per_patient", "validate_every", "patience"):
        command += ["--" + key.replace("_", "-"), str(settings[key])]
    if settings["max_cases"] is not None:
        command += ["--allow-partial-audit"]
    if settings["consistency_weight"] is not None:
        command += ["--consistency-weight", str(settings["consistency_weight"])]
    if (output / "experiment.json").is_file():
        command += ["--resume"]
    return command


def preflight(eda_dir, settings, device="cuda"):
    """Run an actual training-size forward/backward step before a long run."""
    import numpy as np
    import torch
    from eda import load_case, build_2d_input
    from design_network import build_model, drop_one_modality, model_objective, objective_kind, count_parameters
    from train_model import load_manifest, set_seed, amp_context
    rows, _ = load_manifest(Path(eda_dir), allow_partial=settings["max_cases"] is not None)
    row = next(r for r in rows if r["split"] == "train")
    case = load_case(Path(row["case_dir"]))
    z = int(np.argmax(np.count_nonzero(case["segmentation"], axis=(0, 1))))
    batch = settings["batch_size"]
    x = torch.from_numpy(build_2d_input(case["volumes"], z)).unsqueeze(0).repeat(batch, 1, 1, 1).to(device)
    y = torch.from_numpy(case["segmentation"][:, :, z].copy()).long().unsqueeze(0).repeat(batch, 1, 1).to(device)
    brain = torch.from_numpy(case["brain_mask"][:, :, z].copy()).bool().unsqueeze(0).repeat(batch, 1, 1).to(device)
    missing = torch.arange(batch, device=device) % 4
    target_device = torch.device(device)
    reports = []
    for name in settings["models"]:
        set_seed(626)
        model = build_model(name, settings["base_channels"]).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=target_device.type == "cuda")
        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        begin = time.perf_counter()
        with amp_context(target_device, True):
            full = model(x)
            absent = model(drop_one_modality(x, missing)) if objective_kind(name) != "M0" else None
            loss, _ = model_objective(name, full, y, absent, brain, 0.1)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {name} preflight loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite {name} gradients; check runtime/AMP before training")
        scaler.step(optimizer)
        scaler.update()
        if target_device.type == "cuda":
            torch.cuda.synchronize()
        result = dict(model=name, input_shape=list(x.shape), loss=float(loss.detach()),
                      parameters=count_parameters(model), seconds=time.perf_counter() - begin,
                      peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if target_device.type == "cuda" else None)
        reports.append(result)
        print(json.dumps(result), flush=True)
        del optimizer, scaler, model, full, absent, loss
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
    return reports


def evaluate(code_dir, eda_dir, study, settings, save_predictions=False, device="cuda"):
    study = Path(study)
    pointer = study / "evaluation_location.json"
    if pointer.is_file():
        location = Path(json.loads(pointer.read_text())["path"])
        if (location / "evaluation_summary.json").is_file():
            print("Completed evaluation already available:", location)
            return location
    location = study / ("evaluation_" + time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}")
    command = [sys.executable, "-u", str(Path(code_dir) / "train_model.py"), "evaluate",
               "--eda-dir", str(eda_dir), "--run-dir", str(study / "models"),
               "--output", str(location), "--device", device,
               "--batch-size", str(settings["batch_size"])]
    if not save_predictions:
        command += ["--no-save-predictions", "--plot-cases", "0"]
    run(command)
    save_json(pointer, {"path": str(location)})
    return location
