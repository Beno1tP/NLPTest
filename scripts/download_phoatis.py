#!/usr/bin/env python3
"""Download PhoATIS dataset from VinAIResearch/JointIDSF repository.

The repo structure is:
  JointIDSF/PhoATIS/
    syllable-level/   <- used for PhoBERT (syllable tokenization)
    word-level/        <- used for SVM (word-segmented by underthesea)

Each level contains:
  intent_label.txt, slot_label.txt
  train/ dev/ test/ -> each with seq.in, seq.out, label
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

REPO_URL = "https://github.com/VinAIResearch/JointIDSF.git"

# We download both levels into data/raw/
LEVELS = ["syllable-level", "word-level"]
SPLITS = ["train", "dev", "test"]
SPLIT_FILES = ["seq.in", "seq.out", "label"]
ROOT_FILES = ["intent_label.txt", "slot_label.txt"]


def download_dataset():
    """Clone repo and extract PhoATIS data (both syllable-level and word-level)."""
    # Check if already downloaded
    syllable_dir = RAW_DATA_DIR / "syllable-level"
    word_dir = RAW_DATA_DIR / "word-level"

    if syllable_dir.exists() and word_dir.exists():
        print(f"[INFO] Dataset already exists at {RAW_DATA_DIR}")
        if verify_level(syllable_dir) and verify_level(word_dir):
            print("[INFO] Dataset integrity verified. Skipping download.")
            return
        else:
            print("[WARN] Dataset incomplete. Re-downloading...")
            shutil.rmtree(syllable_dir, ignore_errors=True)
            shutil.rmtree(word_dir, ignore_errors=True)

    print(f"[INFO] Downloading PhoATIS dataset from {REPO_URL}")
    print(f"[INFO] Target directory: {RAW_DATA_DIR}")

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "JointIDSF")

        # Shallow clone
        print("[INFO] Cloning repository (shallow)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, clone_dir],
            check=True, capture_output=True, text=True
        )

        # Verify source exists
        src_phoatis = os.path.join(clone_dir, "PhoATIS")
        if not os.path.exists(src_phoatis):
            raise FileNotFoundError(
                f"PhoATIS directory not found at {src_phoatis}. "
                "Repository structure may have changed."
            )

        # Copy both levels
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        for level in LEVELS:
            src = os.path.join(src_phoatis, level)
            dst = RAW_DATA_DIR / level
            if os.path.exists(src):
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, str(dst))
                print(f"[INFO] Copied {level} -> {dst}")
            else:
                print(f"[WARN] {level} not found in repo")

    print("[INFO] Download complete!")


def verify_level(level_dir: Path) -> bool:
    """Verify dataset integrity for one level (syllable or word)."""
    all_ok = True

    for fname in ROOT_FILES:
        fpath = level_dir / fname
        if not fpath.exists():
            print(f"[ERROR] Missing: {fpath}")
            all_ok = False
        elif fpath.stat().st_size == 0:
            print(f"[ERROR] Empty: {fpath}")
            all_ok = False

    for split in SPLITS:
        split_dir = level_dir / split
        if not split_dir.exists():
            print(f"[ERROR] Missing directory: {split_dir}")
            all_ok = False
            continue

        for fname in SPLIT_FILES:
            fpath = split_dir / fname
            if not fpath.exists():
                print(f"[ERROR] Missing: {fpath}")
                all_ok = False
            elif fpath.stat().st_size == 0:
                print(f"[ERROR] Empty: {fpath}")
                all_ok = False

    return all_ok


def print_stats(level_dir: Path, level_name: str):
    """Print summary statistics for one dataset level."""
    print(f"\n  [{level_name}]")

    total = 0
    for split in SPLITS:
        seq_in = level_dir / split / "seq.in"
        if seq_in.exists():
            lines = seq_in.read_text(encoding="utf-8").strip().split("\n")
            count = len(lines)
            total += count
            print(f"    {split:>5}: {count:,} samples")
    print(f"    {'total':>5}: {total:,} samples")

    intent_file = level_dir / "intent_label.txt"
    if intent_file.exists():
        intents = intent_file.read_text(encoding="utf-8").strip().split("\n")
        print(f"    Intents: {len(intents)}")

    slot_file = level_dir / "slot_label.txt"
    if slot_file.exists():
        slots = slot_file.read_text(encoding="utf-8").strip().split("\n")
        print(f"    Slot labels: {len(slots)}")


def main():
    print("PhoATIS Dataset Downloader")
    print("-" * 40)

    download_dataset()

    # Verify both levels
    all_ok = True
    for level in LEVELS:
        level_dir = RAW_DATA_DIR / level
        if level_dir.exists():
            ok = verify_level(level_dir)
            all_ok = all_ok and ok
        else:
            print(f"[ERROR] {level} directory not found")
            all_ok = False

    if all_ok:
        print("\n[OK] All files verified successfully!")
    else:
        print("\n[FAIL] Dataset verification failed!")
        sys.exit(1)

    # Print stats
    print("\n" + "=" * 60)
    print("PhoATIS Dataset Summary")
    print("=" * 60)
    for level in LEVELS:
        level_dir = RAW_DATA_DIR / level
        if level_dir.exists():
            print_stats(level_dir, level)
    print("=" * 60)


if __name__ == "__main__":
    main()
