#!/usr/bin/env python3
"""Verify the complete data pipeline: download, process, and load.

Prints comprehensive stats for each loader type.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def verify_download():
    """Verify raw data exists."""
    raw_dir = PROJECT_ROOT / "data" / "raw" / "syllable-level"
    print("=" * 60)
    print("1. RAW DATA VERIFICATION")
    print("=" * 60)

    for split in ["train", "dev", "test"]:
        for fname in ["seq.in", "seq.out", "label"]:
            fpath = raw_dir / split / fname
            assert fpath.exists(), f"Missing: {fpath}"
            lines = fpath.read_text(encoding="utf-8").strip().split("\n")
            print(f"  {split}/{fname}: {len(lines)} lines")

    print("  [OK] Raw data verified\n")


def verify_processed():
    """Verify processed data and mappings."""
    from src.data.processor import PhoATISProcessor

    print("=" * 60)
    print("2. PROCESSED DATA VERIFICATION")
    print("=" * 60)

    proc = PhoATISProcessor()
    proc.process()

    # Verify mappings
    intent2id = proc.get_intent2id()
    slot2id = proc.get_slot2id()
    id2intent = proc.get_id2intent()
    id2slot = proc.get_id2slot()

    print(f"\n  intent2id: {len(intent2id)} entries")
    print(f"  slot2id:   {len(slot2id)} entries")
    print(f"  id2intent: {len(id2intent)} entries")
    print(f"  id2slot:   {len(id2slot)} entries")

    # Verify round-trip
    for intent, idx in intent2id.items():
        assert id2intent[idx] == intent, f"Round-trip failed: {intent} -> {idx} -> {id2intent[idx]}"
    for slot, idx in slot2id.items():
        assert id2slot[idx] == slot, f"Round-trip failed: {slot} -> {idx} -> {id2slot[idx]}"

    print("  [OK] Label mappings round-trip verified\n")

    # Verify data loading
    for split in ["train", "dev", "test"]:
        texts, slots, intents = proc.load_split(split)
        print(f"  {split}: {len(texts)} texts, {len(slots)} slot seqs, {len(intents)} intents")
        assert len(texts) == len(slots) == len(intents), f"Length mismatch in {split}"

        # Check token-slot alignment
        misaligned = 0
        for i, (text, slot_seq) in enumerate(zip(texts, slots)):
            words = text.split()
            if len(words) != len(slot_seq):
                misaligned += 1
        if misaligned > 0:
            print(f"    [WARN] {misaligned} samples with word-slot misalignment in {split}")

    print("  [OK] Processed data verified\n")


def verify_svm_loader():
    """Verify SVM data loader."""
    from src.data.loaders import SVMDataLoader

    print("=" * 60)
    print("3. SVM DATA LOADER")
    print("=" * 60)

    loader = SVMDataLoader()
    print(f"  Intents: {loader.num_intents}, Slots: {loader.num_slots}")

    for split in ["train", "dev", "test"]:
        texts, intent_ids, slot_labels = loader.load(split)
        print(f"\n  {split}:")
        print(f"    Texts: {len(texts)}")
        print(f"    Intent IDs: {len(intent_ids)} (range: {min(intent_ids)}-{max(intent_ids)})")
        print(f"    Slot seqs: {len(slot_labels)}")
        print(f"    Sample text: '{texts[0][:80]}...'")
        print(f"    Sample intent: {intent_ids[0]} -> {loader.id2intent[intent_ids[0]]}")
        print(f"    Sample slots: {slot_labels[0][:5]}...")

    print("\n  [OK] SVM loader verified\n")


def verify_bert_loader():
    """Verify BERT data loader with PhoBERT tokenization."""
    from src.data.loaders import BERTDataLoader

    print("=" * 60)
    print("4. BERT DATA LOADER (PhoBERT)")
    print("=" * 60)

    t0 = time.time()
    loader = BERTDataLoader(max_seq_length=128)
    print(f"  Intents: {loader.num_intents}, Slots: {loader.num_slots}")
    print(f"  Tokenizer loaded in {time.time() - t0:.1f}s")

    for split in ["train", "dev", "test"]:
        t0 = time.time()
        dataset = loader.load(split)
        elapsed = time.time() - t0
        print(f"\n  {split}: {len(dataset)} samples (encoded in {elapsed:.1f}s)")

        # Check first sample
        sample = dataset[0]
        print(f"    input_ids shape: {sample['input_ids'].shape}")
        print(f"    attention_mask shape: {sample['attention_mask'].shape}")
        print(f"    intent_label: {sample['intent_label'].item()} -> {loader.id2intent[sample['intent_label'].item()]}")
        print(f"    slot_labels shape: {sample['slot_labels'].shape}")

        # Count non-ignored slot labels
        non_ignored = (sample["slot_labels"] != -100).sum().item()
        total = sample["slot_labels"].shape[0]
        print(f"    slot labels: {non_ignored} active / {total} total (rest are -100)")

        # Verify shapes
        assert sample["input_ids"].shape[0] == 128, f"Expected 128, got {sample['input_ids'].shape[0]}"
        assert sample["attention_mask"].shape[0] == 128
        assert sample["slot_labels"].shape[0] == 128

    print("\n  [OK] BERT loader verified\n")


def verify_llm_loader():
    """Verify LLM data loader."""
    from src.data.loaders import LLMDataLoader

    print("=" * 60)
    print("5. LLM DATA LOADER")
    print("=" * 60)

    loader = LLMDataLoader()
    print(f"  Intents: {loader.num_intents}, Slots: {loader.num_slots}")

    # Full test set
    texts, intents, slots = loader.load("test")
    print(f"\n  test (full): {len(texts)} samples")

    # Sampled
    texts_s, intents_s, slots_s = loader.load("test", sample_size=50)
    print(f"  test (sampled 50): {len(texts_s)} samples")

    # Intent and slot type lists for prompting
    intent_list = loader.get_intent_list()
    slot_types = loader.get_slot_types()
    print(f"\n  Available intents (for prompt): {len(intent_list)}")
    print(f"    {intent_list[:5]}...")
    print(f"  Slot types (unique, no B-/I-): {len(slot_types)}")
    print(f"    {slot_types[:5]}...")

    print("\n  [OK] LLM loader verified\n")


def main():
    print("\n" + "#" * 60)
    print("# PhoATIS Data Pipeline - Full Verification")
    print("#" * 60 + "\n")

    verify_download()
    verify_processed()
    verify_svm_loader()
    verify_bert_loader()
    verify_llm_loader()

    print("#" * 60)
    print("# ALL VERIFICATIONS PASSED")
    print("#" * 60)


if __name__ == "__main__":
    main()
