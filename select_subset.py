"""Freeze 100 groups and stage one complete examination per group.

Selection uses a seeded SHA-256 ranking of filename-derived patient IDs, before
reading labels. For repeated scans, choose the lexicographically first complete
examination. No data is deleted or moved. ZIP sources may be the existing 52 shards.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import shutil
import zipfile

from eda import CASE_PATTERN, SUFFIXES, find_cases, case_paths


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def select(data_root, count=100, seed=626):
    groups = {}
    cases = find_cases(Path(data_root))
    for case in cases:
        if all(p.is_file() for p in case_paths(case).values()):
            patient = CASE_PATTERN.fullmatch(case.name).group(1)
            groups.setdefault(patient, []).append(case)
    if count < 3 or count > len(groups):
        raise ValueError(f"Need 3 <= requested groups <= {len(groups)}")
    ranked = sorted(groups, key=lambda pid: (hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest(), pid))
    selected = []
    for pid in ranked[:count]:
        case = sorted(groups[pid], key=lambda p: p.name)[0]
        selected.append(dict(patient_id=pid, case_id=case.name,
                             files={p.name: p.stat().st_size for p in case_paths(case).values()}))
    return dict(selection_seed=seed, selection_method="seeded SHA-256 ranking of patient IDs; first complete examination per group",
                identity_source="filename inference; not independently verified patient identity",
                selection_uses_labels=False, source_scan_count=len(cases), eligible_patient_groups=len(groups),
                selected_patient_groups=count, selected_examinations=count,
                mri_volumes=count * 4, segmentation_masks=count,
                selected=sorted(selected, key=lambda r: r["case_id"]))


def validated_rows(manifest):
    rows = manifest["selected"]
    if len(rows) != manifest["selected_patient_groups"]:
        raise ValueError("Selection count mismatch")
    if len({r["patient_id"] for r in rows}) != len(rows) or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Subset must contain exactly one examination per group")
    for row in rows:
        match = CASE_PATTERN.fullmatch(row["case_id"])
        if not match or match.group(1) != row["patient_id"]:
            raise ValueError("Invalid filename-derived patient/case identity")
        expected = {f"{row['case_id']}-{s}.nii.gz" for s in SUFFIXES}
        if set(row["files"]) != expected or any(size < 1 for size in row["files"].values()):
            raise ValueError("Invalid selection file inventory")
    return rows


def archive_subset(source, manifest, output):
    rows = validated_rows(manifest)
    indexed = {p.name: p for p in find_cases(Path(source))}
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for i, row in enumerate(rows, 1):
            for name, size in row["files"].items():
                file = indexed[row["case_id"]] / name
                if file.stat().st_size != size:
                    raise ValueError("Source data changed after selection")
                archive.write(file, f"{row['case_id']}/{name}")
            if i % 25 == 0:
                print(f"Packaged {i}/{len(rows)} examinations", flush=True)
    return {"name": output.name, "bytes": output.stat().st_size, "sha256": sha256(output)}


def stage_subset(source, destination, selection_path):
    source, destination = Path(source), Path(destination)
    manifest = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    rows = validated_rows(manifest)
    files = {f"{row['case_id']}/{name}": size for row in rows for name, size in row["files"].items()}
    destination.mkdir(parents=True, exist_ok=True)
    unexpected = {p.name for p in find_cases(destination)} - {r["case_id"] for r in rows}
    if unexpected:
        raise ValueError("Staging directory contains other cases. Use a fresh subset staging directory.")
    marker = destination / ".subset_complete.json"
    selection_hash = sha256(selection_path)
    def complete():
        return all((destination / n).is_file() and (destination / n).stat().st_size == size
                   for n, size in files.items())
    if marker.is_file():
        old = json.loads(marker.read_text())
        if old["selection_sha256"] != selection_hash:
            raise ValueError("Selection changed. Use a new staging directory.")
        if complete():
            print("Reusing staged subset:", len(rows), "patient groups", flush=True)
            return destination
    source_archives = []
    if source.is_file() and source.suffix.lower() == ".zip":
        source_archives = [(source, None)]
    elif source.is_dir() and (source / "BraTS100_2D.zip").is_file():
        source_archives = [(source / "BraTS100_2D.zip", None)]
    elif source.is_dir() and (source / "BraTS_Full_Manifest.json").is_file():
        shard_manifest = json.loads((source / "BraTS_Full_Manifest.json").read_text(encoding="utf-8-sig"))
        for row in shard_manifest["archives"]:
            name = row["name"]
            if PurePosixPath(name).name != name or "\\" in name:
                raise ValueError("Invalid shard filename")
            file = source / name
            if not file.is_file() or file.stat().st_size != row["bytes"]:
                raise ValueError(f"Incomplete archive: {file}")
            source_archives.append((file, row["sha256"]))
    else:
        folders = {p.name: p for p in find_cases(source)}
        for i, row in enumerate(rows, 1):
            if row["case_id"] not in folders:
                raise ValueError(f"Missing selected case: {row['case_id']}")
            target = destination / row["case_id"]
            target.mkdir(exist_ok=True)
            for name, size in row["files"].items():
                src, dst = folders[row["case_id"]] / name, target / name
                if not src.is_file() or src.stat().st_size != size:
                    raise ValueError(f"Source file changed: {src}")
                tmp = dst.with_suffix(dst.suffix + ".tmp")
                shutil.copyfile(src, tmp)
                tmp.replace(dst)
            if i % 25 == 0 or i == len(rows):
                print(f"Staged {i}/{len(rows)} examinations", flush=True)
    if source_archives:
        seen = set()
        cache = destination.parent / (destination.name + "_cache")
        cache.mkdir(exist_ok=True)
        for i, (remote, expected_hash) in enumerate(source_archives, 1):
            print(f"Inspecting archive {i}/{len(source_archives)}", flush=True)
            # Read central directory first; copy only shards containing selected cases.
            with zipfile.ZipFile(remote) as archive:
                wanted = []
                for member in archive.infolist():
                    parts = PurePosixPath(member.filename).parts
                    key = "/".join(parts[-2:])
                    if key in files:
                        if key in seen:
                            raise ValueError(f"Duplicate selected file: {key}")
                        seen.add(key)
                        if member.file_size != files[key]:
                            raise ValueError(f"Changed file size: {key}")
                        wanted.append((member.filename, key))
            if not wanted:
                continue
            local = cache / "current.zip"
            required = remote.stat().st_size + sum(files[key] for _, key in wanted) + 2**30
            if shutil.disk_usage(destination).free < required:
                raise OSError("Insufficient local runtime disk space")
            shutil.copyfile(remote, local)
            if expected_hash and sha256(local) != expected_hash:
                raise ValueError(f"Checksum mismatch: {remote.name}")
            with zipfile.ZipFile(local) as archive:
                for member_name, key in wanted:
                    # Use only validated case/filename keys, never an archive-provided extraction path.
                    target = destination / key
                    target.parent.mkdir(exist_ok=True)
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    with archive.open(member_name) as incoming, temporary.open("wb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing, 8 * 2**20)
                    temporary.replace(target)
            local.unlink()
        if seen != set(files):
            raise ValueError(f"Archives lack {len(set(files) - seen)} selected files")
    if not complete():
        raise ValueError("Subset staging incomplete")
    write_json(marker, dict(selection_sha256=selection_hash, groups=len(rows),
                            verification="ZIP CRC and supplied shard SHA-256; reuse checks filenames and sizes"))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New selection JSON path")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=626)
    parser.add_argument("--archive", type=Path, help="Optional new subset ZIP")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Selection already exists; preserve it or use a new output path")
    selection = select(args.data_root, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, selection)
    print({k: v for k, v in selection.items() if k != "selected"}, flush=True)
    if args.archive:
        info = archive_subset(args.data_root, selection, args.archive)
        write_json(args.archive.with_suffix(".manifest.json"), info)
        print(info, flush=True)


if __name__ == "__main__":
    main()
