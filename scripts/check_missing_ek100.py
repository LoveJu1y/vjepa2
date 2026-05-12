#!/usr/bin/env python3
"""Check missing EK100 videos against an annotation CSV.

Default is tailored to the current project layout:
- annotations csv: EPIC_100_train.csv
- dataset root: /share/project/galbot-Hotel-Model/ego-data/epic_kitchens/3h91syskeag572hl6tvuovwv4d/videos
- file format 2: <root>/<split>/<PID>/<VIDEO_ID>.MP4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def build_video_path(dataset_root: Path, video_id: str, file_format: int, split: str) -> Path:
    pid = video_id.split("_")[0]
    if file_format == 0:
        return dataset_root / pid / "videos" / f"{video_id}.MP4"
    if file_format == 1:
        return dataset_root / pid / f"{video_id}.MP4"
    if file_format == 2:
        return dataset_root / split / pid / f"{video_id}.MP4"
    raise ValueError(f"Unsupported file_format={file_format}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate missing EK100 video list.")
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("/share/project/shirc/data/ek100/epic-kitchens-100-annotations/EPIC_100_train.csv"),
        help="Path to EPIC annotations CSV (train/val).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/share/project/galbot-Hotel-Model/ego-data/epic_kitchens/3h91syskeag572hl6tvuovwv4d/videos"),
        help="Dataset root. For file_format=2 this should be the .../videos directory.",
    )
    parser.add_argument(
        "--file-format",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Path format: 0=<root>/<PID>/videos/<id>.MP4, 1=<root>/<PID>/<id>.MP4, 2=<root>/<split>/<PID>/<id>.MP4",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "val"],
        help="Split folder for file_format=2.",
    )
    parser.add_argument(
        "--out-missing-ids",
        type=Path,
        default=Path("/share/project/lvjing/vjepa2/doc/missing_ek100_train_video_ids.txt"),
        help="Output text file with one missing video_id per line.",
    )
    parser.add_argument(
        "--out-missing-paths-csv",
        type=Path,
        default=Path("/share/project/lvjing/vjepa2/doc/missing_ek100_train_paths.csv"),
        help="Output CSV containing video_id and expected path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.annotations_csv)
    unique_video_ids = list(dict.fromkeys(df["video_id"].values))

    missing: list[tuple[str, Path]] = []
    found = 0
    for vid in unique_video_ids:
        expected = build_video_path(args.dataset_root, vid, args.file_format, args.split)
        if expected.exists():
            found += 1
        else:
            missing.append((vid, expected))

    args.out_missing_ids.parent.mkdir(parents=True, exist_ok=True)
    args.out_missing_paths_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.out_missing_ids.open("w", encoding="utf-8") as f:
        for vid, _ in missing:
            f.write(f"{vid}\n")

    with args.out_missing_paths_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "expected_path"])
        for vid, expected in missing:
            writer.writerow([vid, str(expected)])

    total = len(unique_video_ids)
    print(f"annotations_csv: {args.annotations_csv}")
    print(f"dataset_root   : {args.dataset_root}")
    print(f"file_format    : {args.file_format} (split={args.split})")
    print(f"total videos   : {total}")
    print(f"found videos   : {found}")
    print(f"missing videos : {len(missing)}")
    print(f"missing ids    : {args.out_missing_ids}")
    print(f"missing paths  : {args.out_missing_paths_csv}")


if __name__ == "__main__":
    main()
