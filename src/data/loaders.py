"""Data loaders for SVM, BERT (PhoBERT), and LLM models.

Three loader classes that read processed PhoATIS data and return
model-specific formats:
  - SVMDataLoader: tokenized texts + labels for sklearn
  - BERTDataLoader: PyTorch Dataset with PhoBERT tokenization + subword alignment
  - LLMDataLoader: raw texts + labels for zero-shot prompting
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
SPLITS = ["train", "dev", "test"]


def _read_split(
    data_dir: Path, split: str
) -> Tuple[List[str], List[List[str]], List[str]]:
    """Read a data split from processed directory.

    Returns:
        texts: input sentences
        slot_labels: BIO tag sequences per sentence
        intent_labels: intent label per sentence
    """
    split_dir = data_dir / split
    texts = (split_dir / "seq.in").read_text(encoding="utf-8").strip().split("\n")
    slots_raw = (split_dir / "seq.out").read_text(encoding="utf-8").strip().split("\n")
    intents = (split_dir / "label").read_text(encoding="utf-8").strip().split("\n")
    slot_labels = [s.split() for s in slots_raw]
    return texts, slot_labels, intents


def _load_mapping(data_dir: Path, filename: str) -> Dict[str, int]:
    """Load a JSON label-to-id mapping."""
    with open(data_dir / filename, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# SVM Data Loader
# ---------------------------------------------------------------------------

class SVMDataLoader:
    """Load data for TF-IDF + SVM baseline.

    Returns raw text strings and integer-encoded labels.
    Word segmentation (underthesea) is applied at feature extraction time,
    not here — sklearn's TfidfVectorizer handles tokenization.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else PROCESSED_DIR
        self.intent2id = _load_mapping(self.data_dir, "intent2id.json")
        self.slot2id = _load_mapping(self.data_dir, "slot2id.json")
        self.id2intent = {v: k for k, v in self.intent2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}

    def load(
        self, split: str
    ) -> Tuple[List[str], List[int], List[List[str]]]:
        """Load data for a split.

        Returns:
            texts: raw input sentences
            intent_ids: integer intent labels
            slot_labels: BIO tag sequences (strings, not encoded)
        """
        texts, slot_labels, intents = _read_split(self.data_dir, split)
        intent_ids = [self.intent2id.get(i, self.intent2id.get("UNK", 0)) for i in intents]
        return texts, intent_ids, slot_labels

    @property
    def num_intents(self) -> int:
        return len(self.intent2id)

    @property
    def num_slots(self) -> int:
        return len(self.slot2id)


# ---------------------------------------------------------------------------
# BERT (PhoBERT) Data Loader
# ---------------------------------------------------------------------------

class BERTDataLoader:
    """Load data as PyTorch Dataset with PhoBERT tokenization.

    Handles subword alignment: when a word is split into multiple subword
    tokens, only the first subword gets the slot label; the rest get
    ignore_index (-100) so they don't contribute to slot loss.

    Uses vinai/phobert-base-v2 tokenizer.
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        model_name: str = "vinai/phobert-base-v2",
        max_seq_length: int = 128,
    ):
        self.data_dir = Path(data_dir) if data_dir else PROCESSED_DIR
        self.model_name = model_name
        self.max_seq_length = max_seq_length

        self.intent2id = _load_mapping(self.data_dir, "intent2id.json")
        self.slot2id = _load_mapping(self.data_dir, "slot2id.json")
        self.id2intent = {v: k for k, v in self.intent2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}

        # Lazy tokenizer loading
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def load(self, split: str):
        """Load data as a PyTorch Dataset.

        Returns:
            PhoATISDataset with __getitem__ returning:
                input_ids, attention_mask, intent_label, slot_labels
        """
        import torch
        from torch.utils.data import Dataset

        texts, slot_labels, intents = _read_split(self.data_dir, split)

        # Encode all samples
        all_input_ids = []
        all_attention_masks = []
        all_intent_labels = []
        all_slot_labels = []

        for text, slots, intent in zip(texts, slot_labels, intents):
            encoded = self._encode_sample(text, slots, intent)
            all_input_ids.append(encoded["input_ids"])
            all_attention_masks.append(encoded["attention_mask"])
            all_intent_labels.append(encoded["intent_label"])
            all_slot_labels.append(encoded["slot_labels"])

        class PhoATISDataset(Dataset):
            def __init__(self, input_ids, attention_masks, intent_labels, slot_labels_list):
                self.input_ids = input_ids
                self.attention_masks = attention_masks
                self.intent_labels = intent_labels
                self.slot_labels = slot_labels_list

            def __len__(self):
                return len(self.input_ids)

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
                    "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
                    "intent_label": torch.tensor(self.intent_labels[idx], dtype=torch.long),
                    "slot_labels": torch.tensor(self.slot_labels[idx], dtype=torch.long),
                }

        return PhoATISDataset(
            all_input_ids, all_attention_masks,
            all_intent_labels, all_slot_labels
        )

    def _encode_sample(
        self, text: str, slot_labels: List[str], intent: str
    ) -> Dict:
        """Encode a single sample with subword alignment.

        Strategy (first-token):
          - Split text into words
          - For each word, tokenize into subwords
          - First subword gets the word's slot label
          - Remaining subwords get IGNORE_INDEX (-100)
          - CLS and SEP tokens also get IGNORE_INDEX
        """
        words = text.split()
        slot_tags = slot_labels

        # Ensure alignment
        if len(words) != len(slot_tags):
            # Fallback: truncate or pad to match
            min_len = min(len(words), len(slot_tags))
            words = words[:min_len]
            slot_tags = slot_tags[:min_len]

        # Tokenize each word individually to track subword alignment
        all_subword_ids = []
        aligned_slot_ids = []

        for word, slot_tag in zip(words, slot_tags):
            subword_tokens = self.tokenizer.tokenize(word)
            if not subword_tokens:
                # Unknown token - use UNK
                subword_tokens = [self.tokenizer.unk_token]

            subword_ids = self.tokenizer.convert_tokens_to_ids(subword_tokens)
            all_subword_ids.extend(subword_ids)

            # First subword gets the slot label, rest get IGNORE_INDEX
            slot_id = self.slot2id.get(slot_tag, self.slot2id.get("UNK", 1))
            aligned_slot_ids.append(slot_id)
            aligned_slot_ids.extend([self.IGNORE_INDEX] * (len(subword_ids) - 1))

        # Truncate to max_seq_length - 2 (for CLS and SEP)
        max_tokens = self.max_seq_length - 2
        all_subword_ids = all_subword_ids[:max_tokens]
        aligned_slot_ids = aligned_slot_ids[:max_tokens]

        # Add CLS and SEP
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        input_ids = [cls_id] + all_subword_ids + [sep_id]
        slot_ids = [self.IGNORE_INDEX] + aligned_slot_ids + [self.IGNORE_INDEX]
        attention_mask = [1] * len(input_ids)

        # Pad to max_seq_length
        pad_length = self.max_seq_length - len(input_ids)
        pad_id = self.tokenizer.pad_token_id

        input_ids += [pad_id] * pad_length
        slot_ids += [self.IGNORE_INDEX] * pad_length
        attention_mask += [0] * pad_length

        # Intent label
        intent_id = self.intent2id.get(intent, self.intent2id.get("UNK", 0))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "intent_label": intent_id,
            "slot_labels": slot_ids,
        }

    @property
    def num_intents(self) -> int:
        return len(self.intent2id)

    @property
    def num_slots(self) -> int:
        return len(self.slot2id)


# ---------------------------------------------------------------------------
# LLM Data Loader
# ---------------------------------------------------------------------------

class LLMDataLoader:
    """Load data for LLM zero-shot evaluation.

    Returns raw texts with labels for evaluation — no encoding needed
    since LLMs use their own tokenization via API.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else PROCESSED_DIR
        self.intent2id = _load_mapping(self.data_dir, "intent2id.json")
        self.slot2id = _load_mapping(self.data_dir, "slot2id.json")
        self.id2intent = {v: k for k, v in self.intent2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}

        # Load label lists for prompting
        self.intent_labels = (
            self.data_dir / "intent_labels.txt"
        ).read_text(encoding="utf-8").strip().split("\n")
        self.slot_labels = (
            self.data_dir / "slot_labels.txt"
        ).read_text(encoding="utf-8").strip().split("\n")

    def load(
        self, split: str, sample_size: Optional[int] = None, seed: int = 42
    ) -> Tuple[List[str], List[str], List[List[str]]]:
        """Load data for LLM evaluation.

        Args:
            split: data split name
            sample_size: if set, randomly sample this many examples
            seed: random seed for sampling

        Returns:
            texts: raw input sentences
            intents: intent label strings
            slot_labels: BIO tag sequences (strings)
        """
        texts, slot_labels, intents = _read_split(self.data_dir, split)

        if sample_size and sample_size < len(texts):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(texts), size=sample_size, replace=False)
            texts = [texts[i] for i in indices]
            intents = [intents[i] for i in indices]
            slot_labels = [slot_labels[i] for i in indices]

        return texts, intents, slot_labels

    def get_intent_list(self) -> List[str]:
        """Get list of all intent labels (for prompt construction)."""
        return [l for l in self.intent_labels if l not in ("UNK",)]

    def get_slot_types(self) -> List[str]:
        """Get unique slot types (without B-/I- prefix and special tokens)."""
        slot_types = set()
        for label in self.slot_labels:
            if label in ("PAD", "UNK", "O"):
                continue
            # Remove B- or I- prefix
            slot_type = label.split("-", 1)[1] if "-" in label else label
            slot_types.add(slot_type)
        return sorted(slot_types)

    @property
    def num_intents(self) -> int:
        return len(self.intent2id)

    @property
    def num_slots(self) -> int:
        return len(self.slot2id)
