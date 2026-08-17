"""
duplicate_leakage_check.py

Purpose
-------
Reviewer 1 (point 2) and Reviewer 2 (point 4) both flag that the Kaggle/Mendeley
Alzheimer's MRI dataset provides no patient/scan-session identifiers, so it is not
possible to guarantee subject-level separation between train/val/test splits from
metadata alone. This script gives you a concrete, defensible substitute: it finds
EXACT and NEAR duplicate images (which is the most common source of leakage in
slice-based 2D MRI datasets, since adjacent slices from the same scan session are
often near-identical) and tells you whether any such duplicates cross your
train/val/test split boundaries.

Run this on your actual local copy of the dataset, using the exact file lists /
split assignment you used for the paper. Report the printed summary numbers
directly in the manuscript (Section 5.1) and in your response to Reviewer 1,
point 2 / Reviewer 2, point 4.

Requirements
------------
pip install imagehash pillow pandas --break-system-packages

Usage
-----
python duplicate_leakage_check.py \
    --train_dir /path/to/train --val_dir /path/to/val --test_dir /path/to/test \
    --hash_size 16 --near_dup_threshold 5 \
    --out_csv duplicate_report.csv
"""

import argparse
import itertools
import os
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image


def collect_images(directory, split_name):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    records = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in exts:
                # class label inferred from parent folder name; adjust if your
                # directory layout differs
                cls = Path(root).name
                records.append(
                    {"path": os.path.join(root, f), "split": split_name, "class": cls}
                )
    return records


def compute_hashes(records, hash_size):
    for r in records:
        try:
            with Image.open(r["path"]) as img:
                r["phash"] = imagehash.phash(img, hash_size=hash_size)
                r["ahash"] = imagehash.average_hash(img, hash_size=hash_size)
        except Exception as e:
            r["phash"] = None
            r["ahash"] = None
            r["error"] = str(e)
    return records


def find_duplicates(records, near_dup_threshold):
    exact_groups = {}
    for r in records:
        if r["phash"] is None:
            continue
        exact_groups.setdefault(str(r["phash"]), []).append(r)

    exact_dupes = {k: v for k, v in exact_groups.items() if len(v) > 1}

    # near-duplicates: pairwise Hamming distance on phash below threshold
    # (O(n^2) — fine for a few thousand images; for larger sets, bucket by
    # hash prefix first)
    near_dupe_pairs = []
    items = [r for r in records if r["phash"] is not None]
    for a, b in itertools.combinations(items, 2):
        dist = a["phash"] - b["phash"]
        if 0 < dist <= near_dup_threshold:
            near_dupe_pairs.append((a, b, dist))

    return exact_dupes, near_dupe_pairs


def summarize(exact_dupes, near_dupe_pairs, out_csv):
    rows = []

    cross_split_exact = 0
    for h, group in exact_dupes.items():
        splits = {g["split"] for g in group}
        crosses = len(splits) > 1
        cross_split_exact += int(crosses)
        for g in group:
            rows.append(
                {
                    "type": "exact",
                    "hash": h,
                    "path": g["path"],
                    "split": g["split"],
                    "class": g["class"],
                    "crosses_split": crosses,
                }
            )

    cross_split_near = 0
    for a, b, dist in near_dupe_pairs:
        crosses = a["split"] != b["split"]
        cross_split_near += int(crosses)
        rows.append(
            {
                "type": "near",
                "hash": dist,
                "path": f"{a['path']}  <->  {b['path']}",
                "split": f"{a['split']} / {b['split']}",
                "class": f"{a['class']} / {b['class']}",
                "crosses_split": crosses,
            }
        )

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(out_csv, index=False)

    print("=" * 70)
    print("DUPLICATE / NEAR-DUPLICATE LEAKAGE REPORT")
    print("=" * 70)
    print(f"Exact-duplicate groups found:          {len(exact_dupes)}")
    print(f"  ...of which cross a split boundary:  {cross_split_exact}")
    print(f"Near-duplicate pairs found:            {len(near_dupe_pairs)}")
    print(f"  ...of which cross a split boundary:  {cross_split_near}")
    print()
    if cross_split_exact or cross_split_near:
        print(
            "ACTION NEEDED: duplicates cross split boundaries. Report the exact "
            "counts above in the manuscript, and either (a) remove the offending "
            "images from all but one split and re-evaluate, or (b) report results "
            "both with and without the affected images so reviewers can see the "
            "effect size of the leakage."
        )
    else:
        print(
            "No cross-split duplicates detected at this threshold. Report this "
            "explicitly in the manuscript as your leakage-control evidence, and "
            "state the hash type / threshold used so it is reproducible."
        )
    print(f"\nFull row-level detail written to: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--hash_size", type=int, default=16)
    ap.add_argument(
        "--near_dup_threshold",
        type=int,
        default=5,
        help="Max Hamming distance between perceptual hashes to call two images "
        "near-duplicates. Lower = stricter. Start at 5 and sanity-check a sample "
        "of flagged pairs visually before trusting the threshold.",
    )
    ap.add_argument("--out_csv", default="duplicate_report.csv")
    args = ap.parse_args()

    records = []
    records += collect_images(args.train_dir, "train")
    records += collect_images(args.val_dir, "val")
    records += collect_images(args.test_dir, "test")
    print(f"Found {len(records)} images across train/val/test.")

    records = compute_hashes(records, args.hash_size)
    exact_dupes, near_dupe_pairs = find_duplicates(records, args.near_dup_threshold)
    summarize(exact_dupes, near_dupe_pairs, args.out_csv)


if __name__ == "__main__":
    main()
