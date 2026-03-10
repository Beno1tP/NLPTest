"""Process raw PhoATIS data into structured format for model training.

Reads syllable-level data from data/raw/syllable-level/ and produces:
  data/processed/
    intent_labels.txt     - one intent per line
    slot_labels.txt       - one BIO slot label per line
    intent2id.json        - intent -> integer mapping
    slot2id.json          - slot label -> integer mapping
    train/ dev/ test/     - each with seq.in, seq.out, label
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "syllable-level"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ["train", "dev", "test"]


class PhoATISProcessor:
    """Process PhoATIS raw data into train/dev/test with label mappings."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DATA_DIR

        self.intent_labels: List[str] = []
        self.slot_labels: List[str] = []
        self.intent2id: Dict[str, int] = {}
        self.slot2id: Dict[str, int] = {}

    def process(self) -> None:
        """Run the full processing pipeline."""
        self._validate_raw_data()
        self._build_label_mappings()
        self._copy_splits()
        self._save_mappings()
        self._print_summary()

    def _validate_raw_data(self) -> None:
        """Ensure raw data exists and has expected structure."""
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_dir}\n"
                "Run scripts/download_phoatis.py first."
            )

        for split in SPLITS:
            split_dir = self.raw_dir / split
            for fname in ["seq.in", "seq.out", "label"]:
                fpath = split_dir / fname
                if not fpath.exists():
                    raise FileNotFoundError(f"Missing: {fpath}")

        for fname in ["intent_label.txt", "slot_label.txt"]:
            if not (self.raw_dir / fname).exists():
                raise FileNotFoundError(f"Missing: {self.raw_dir / fname}")

    def _build_label_mappings(self) -> None:
        """Build intent2id and slot2id from label files."""
        # Read intent labels from the dataset's label file
        intent_text = (self.raw_dir / "intent_label.txt").read_text(
            encoding="utf-8"
        ).strip()
        self.intent_labels = [l for l in intent_text.split("\n") if l.strip()]

        # Read slot labels
        slot_text = (self.raw_dir / "slot_label.txt").read_text(
            encoding="utf-8"
        ).strip()
        self.slot_labels = [l for l in slot_text.split("\n") if l.strip()]

        # Build mappings (0-indexed)
        self.intent2id = {label: idx for idx, label in enumerate(self.intent_labels)}
        self.slot2id = {label: idx for idx, label in enumerate(self.slot_labels)}

    def _copy_splits(self) -> None:
        """Copy split data to processed directory."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            src_dir = self.raw_dir / split
            dst_dir = self.processed_dir / split
            dst_dir.mkdir(parents=True, exist_ok=True)

            for fname in ["seq.in", "seq.out", "label"]:
                src = src_dir / fname
                dst = dst_dir / fname
                shutil.copy2(str(src), str(dst))

    def _save_mappings(self) -> None:
        """Save label lists and JSON mappings."""
        # Intent labels (one per line)
        (self.processed_dir / "intent_labels.txt").write_text(
            "\n".join(self.intent_labels) + "\n",
            encoding="utf-8",
        )

        # Slot labels (one per line)
        (self.processed_dir / "slot_labels.txt").write_text(
            "\n".join(self.slot_labels) + "\n",
            encoding="utf-8",
        )

        # JSON mappings
        with open(self.processed_dir / "intent2id.json", "w", encoding="utf-8") as f:
            json.dump(self.intent2id, f, ensure_ascii=False, indent=2)

        with open(self.processed_dir / "slot2id.json", "w", encoding="utf-8") as f:
            json.dump(self.slot2id, f, ensure_ascii=False, indent=2)

    def _print_summary(self) -> None:
        """Print processing summary."""
        print("=" * 60)
        print("PhoATIS Data Processing Complete")
        print("=" * 60)
        print(f"  Output directory: {self.processed_dir}")
        print(f"  Intent labels: {len(self.intent_labels)}")
        print(f"  Slot labels:   {len(self.slot_labels)}")

        for split in SPLITS:
            split_dir = self.processed_dir / split
            seq_in = split_dir / "seq.in"
            count = len(seq_in.read_text(encoding="utf-8").strip().split("\n"))
            print(f"  {split:>5} samples: {count:,}")

        # Print intent distribution for train
        self._print_intent_distribution("train")
        print("=" * 60)

    def _print_intent_distribution(self, split: str) -> None:
        """Print intent distribution for a split."""
        label_file = self.processed_dir / split / "label"
        labels = label_file.read_text(encoding="utf-8").strip().split("\n")

        from collections import Counter
        counts = Counter(labels)
        print(f"\n  Intent distribution ({split}, top 10):")
        for intent, count in counts.most_common(10):
            pct = 100.0 * count / len(labels)
            print(f"    {intent:<35} {count:>5} ({pct:5.1f}%)")
        if len(counts) > 10:
            print(f"    ... and {len(counts) - 10} more intents")

    # --- Public accessors ---

    def load_split(self, split: str) -> Tuple[List[str], List[List[str]], List[str]]:
        """Load a processed split.

        Returns:
            texts: list of input sentences
            slot_labels: list of BIO tag sequences (one list per sentence)
            intent_labels: list of intent strings
        """
        if split not in SPLITS:
            raise ValueError(f"Invalid split: {split}. Must be one of {SPLITS}")

        split_dir = self.processed_dir / split

        texts = (split_dir / "seq.in").read_text(encoding="utf-8").strip().split("\n")
        slot_seqs = (split_dir / "seq.out").read_text(encoding="utf-8").strip().split("\n")
        intents = (split_dir / "label").read_text(encoding="utf-8").strip().split("\n")

        slot_labels = [seq.split() for seq in slot_seqs]

        return texts, slot_labels, intents

    def get_intent2id(self) -> Dict[str, int]:
        """Get intent to ID mapping (loads from file if not in memory)."""
        if not self.intent2id:
            with open(self.processed_dir / "intent2id.json", encoding="utf-8") as f:
                self.intent2id = json.load(f)
        return self.intent2id

    def get_slot2id(self) -> Dict[str, int]:
        """Get slot label to ID mapping (loads from file if not in memory)."""
        if not self.slot2id:
            with open(self.processed_dir / "slot2id.json", encoding="utf-8") as f:
                self.slot2id = json.load(f)
        return self.slot2id

    def get_id2intent(self) -> Dict[int, str]:
        """Get ID to intent mapping."""
        return {v: k for k, v in self.get_intent2id().items()}

    def get_id2slot(self) -> Dict[int, str]:
        """Get ID to slot label mapping."""
        return {v: k for k, v in self.get_slot2id().items()}


def main():
    """Process PhoATIS dataset."""
    processor = PhoATISProcessor()
    processor.process()


if __name__ == "__main__":
    main()
