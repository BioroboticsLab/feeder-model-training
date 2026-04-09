#!/usr/bin/env python3
"""Build the feeder_only POLO dataset variant using symlinks to NFS.

Reads split_assignment.json and creates a local directory tree with symlinks
pointing to the original images/labels on NFS.

Usage:
    python build_feeder_only_dataset.py          # build (skip existing)
    python build_feeder_only_dataset.py --clean   # remove and rebuild
"""

import json
import os
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SPLIT_ASSIGNMENT = PROJECT_ROOT / "split_assignment.json"
TARGET_DIR = PROJECT_ROOT / "data" / "feeder_only"

NFS_POLO = Path("/mnt/trove/beesbook_feeder_model/feeder_bee_datasets_v1/models/polo/cvat_only")

CLASS_NAMES = {0: "UnmarkedBee", 1: "MarkedBee", 2: "BeeInCell", 3: "UpsideDownBee"}
RADII = {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0}


def build_nfs_index():
    """Scan NFS source splits and return {filename: abs_image_path}."""
    index = {}
    for split in ["train", "valid", "test"]:
        img_dir = NFS_POLO / split / "images"
        if not img_dir.exists():
            print(f"WARNING: NFS directory not found: {img_dir}")
            continue
        for p in img_dir.iterdir():
            if p.suffix == ".png":
                index[p.name] = p.resolve()
    return index


def main():
    clean = "--clean" in sys.argv

    if clean and TARGET_DIR.exists():
        print(f"Cleaning {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)

    with open(SPLIT_ASSIGNMENT) as f:
        assignments = json.load(f)

    print(f"Split assignment: {len(assignments)} entries")
    print(f"Scanning NFS: {NFS_POLO}")
    nfs_index = build_nfs_index()
    print(f"NFS index: {len(nfs_index)} files\n")

    # Create target dirs
    for split in ["train", "valid", "test"]:
        (TARGET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    created = {"train": 0, "valid": 0, "test": 0}
    missing_img = []
    missing_lbl = []

    for filename, info in sorted(assignments.items()):
        target_split = info["split"]

        if filename not in nfs_index:
            missing_img.append(filename)
            continue

        src_image = nfs_index[filename]
        src_label = src_image.parent.parent / "labels" / (src_image.stem + ".txt")

        dst_image = TARGET_DIR / target_split / "images" / filename
        dst_label = TARGET_DIR / target_split / "labels" / (src_image.stem + ".txt")

        if not dst_image.exists():
            os.symlink(src_image, dst_image)

        if src_label.exists():
            if not dst_label.exists():
                os.symlink(src_label.resolve(), dst_label)
        else:
            missing_lbl.append(filename)

        created[target_split] += 1

    # Write data.yaml
    data = {
        "names": CLASS_NAMES,
        "path": str(TARGET_DIR.resolve()),
        "radii": RADII,
        "test": "test/images",
        "train": "train/images",
        "val": "valid/images",
    }
    with open(TARGET_DIR / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=True, allow_unicode=True)

    # Summary
    print("Created symlinks:")
    for split, n in created.items():
        print(f"  {split:5s}: {n:4d}")
    print(f"  Total: {sum(created.values())}")

    if missing_img:
        print(f"\nWARNING: {len(missing_img)} images not found on NFS")
    if missing_lbl:
        print(f"WARNING: {len(missing_lbl)} label files not found on NFS")

    print(f"\nDataset ready at: {TARGET_DIR}")
    print(f"data.yaml: {TARGET_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
