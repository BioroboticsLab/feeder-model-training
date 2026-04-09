#!/usr/bin/env python3
"""Generate a new train/valid/test split assignment from existing NFS files.

Rules:
  - All exit cam images → train (extra learning data)
  - Valid and test → feeder cam only (reflects deployment)
  - Split by recording session (no temporal leakage)
  - Target: ~75% train, ~12.5% valid, ~12.5% test (feeder only)
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
NFS_POLO = Path("/mnt/trove/beesbook_feeder_model/feeder_bee_datasets_v1/models/polo/cvat_only")
OUTPUT = Path(__file__).resolve().parent / "split_assignment.json"


def scan_nfs_images():
    """Scan all NFS cvat_only splits and return list of filenames with cam_type."""
    images = []
    for split in ["train", "valid", "test"]:
        img_dir = NFS_POLO / split / "images"
        if not img_dir.exists():
            print(f"WARNING: {img_dir} not found")
            continue
        for p in sorted(img_dir.glob("*.png")):
            cam_type = "exit" if "exitcam" in p.name else "feeder"
            images.append({"filename": p.name, "cam_type": cam_type})
    return images


def main():
    random.seed(SEED)

    images = scan_nfs_images()
    print(f"Scanned {len(images)} images from NFS")

    # Group by (cam_type, session)
    sessions = defaultdict(list)
    for img in images:
        session = img["filename"].rsplit("__frame_", 1)[0]
        sessions[(img["cam_type"], session)].append(img["filename"])

    # Separate exit and feeder sessions
    exit_sessions = {k: v for k, v in sessions.items() if k[0] == "exit"}
    feeder_sessions = {k: v for k, v in sessions.items() if k[0] == "feeder"}

    # Shuffle feeder sessions
    feeder_keys = sorted(feeder_sessions.keys())
    random.shuffle(feeder_keys)

    # Count total feeder images
    total_feeder = sum(len(feeder_sessions[k]) for k in feeder_keys)

    # Assign feeder sessions to splits, targeting 12.5% valid, 12.5% test
    valid_target = total_feeder * 0.125
    test_target = total_feeder * 0.125

    valid_sessions, test_sessions, train_sessions = [], [], []
    valid_count, test_count = 0, 0

    for k in feeder_keys:
        count = len(feeder_sessions[k])
        if valid_count < valid_target:
            valid_sessions.append(k)
            valid_count += count
        elif test_count < test_target:
            test_sessions.append(k)
            test_count += count
        else:
            train_sessions.append(k)

    # Build new assignment
    new_assignment = {}

    # All exit → train
    for k, fnames in exit_sessions.items():
        for fname in fnames:
            new_assignment[fname] = {"cam_type": "exit", "split": "train"}

    # Feeder splits
    for k in train_sessions:
        for fname in feeder_sessions[k]:
            new_assignment[fname] = {"cam_type": "feeder", "split": "train"}

    for k in valid_sessions:
        for fname in feeder_sessions[k]:
            new_assignment[fname] = {"cam_type": "feeder", "split": "valid"}

    for k in test_sessions:
        for fname in feeder_sessions[k]:
            new_assignment[fname] = {"cam_type": "feeder", "split": "test"}

    # Sort by filename
    new_assignment = dict(sorted(new_assignment.items()))

    # Print summary
    from collections import Counter
    counts = Counter()
    for info in new_assignment.values():
        counts[(info["cam_type"], info["split"])] += 1

    print("\nNew split assignment:")
    for (cam, split), n in sorted(counts.items()):
        print(f"  {cam:6s} {split:5s}: {n:4d}")

    train_total = sum(n for (_, s), n in counts.items() if s == "train")
    valid_total = sum(n for (_, s), n in counts.items() if s == "valid")
    test_total = sum(n for (_, s), n in counts.items() if s == "test")
    grand_total = train_total + valid_total + test_total

    print(f"\n  Train: {train_total:4d} ({train_total / grand_total * 100:.1f}%)")
    print(f"  Valid: {valid_total:4d} ({valid_total / grand_total * 100:.1f}%)  [feeder only]")
    print(f"  Test:  {test_total:4d} ({test_total / grand_total * 100:.1f}%)  [feeder only]")
    print(f"  Total: {grand_total}")

    print(f"\nValid sessions ({len(valid_sessions)}):")
    for k in sorted(valid_sessions):
        print(f"  {k[1]} ({len(feeder_sessions[k])} frames)")
    print(f"\nTest sessions ({len(test_sessions)}):")
    for k in sorted(test_sessions):
        print(f"  {k[1]} ({len(feeder_sessions[k])} frames)")

    with open(OUTPUT, "w") as f:
        json.dump(new_assignment, f, indent=2)
    print(f"\nWritten to: {OUTPUT}")


if __name__ == "__main__":
    main()
