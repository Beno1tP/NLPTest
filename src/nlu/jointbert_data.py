"""Dataset classes for JointBERT training and evaluation.

Provides PyTorch Dataset classes that handle:
- PhoBERT tokenization with subword alignment
- First-token strategy for slot labeling (non-first subwords get -100)
- Proper attention mask handling
- Collation for DataLoader batching
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class JointBERTDataset(Dataset):
    """PyTorch Dataset for JointBERT training with PhoBERT tokenization.

    Handles subword alignment using first-token strategy:
    - When a word is split into multiple subwords, only the first subword
      receives the slot label
    - Remaining subwords receive -100 (ignored in loss computation)
    - [CLS] and [SEP] tokens also receive -100

    Args:
        texts: List of input sentences
        slot_labels: List of BIO tag sequences (one per sentence)
        intent_labels: List of intent strings
        tokenizer: PhoBERT tokenizer
        slot2id: Slot label to ID mapping
        intent2id: Intent label to ID mapping
        max_seq_length: Maximum sequence length for padding/truncation
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        texts: List[str],
        slot_labels: List[List[str]],
        intent_labels: List[str],
        tokenizer,
        slot2id: Dict[str, int],
        intent2id: Dict[str, int],
        max_seq_length: int = 128,
    ):
        self.texts = texts
        self.slot_labels = slot_labels
        self.intent_labels = intent_labels
        self.tokenizer = tokenizer
        self.slot2id = slot2id
        self.intent2id = intent2id
        self.max_seq_length = max_seq_length

        # Pre-encode all samples for faster training
        self.encoded_samples = [
            self._encode_sample(text, slots, intent)
            for text, slots, intent in zip(texts, slot_labels, intent_labels)
        ]

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.encoded_samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "intent_label": torch.tensor(sample["intent_label"], dtype=torch.long),
            "slot_labels": torch.tensor(sample["slot_labels"], dtype=torch.long),
        }

    def _encode_sample(
        self,
        text: str,
        slot_labels: List[str],
        intent: str,
    ) -> Dict[str, List[int]]:
        """Encode a single sample with subword alignment.

        Strategy (first-token alignment):
            1. Split text into words
            2. For each word, tokenize into subwords
            3. First subword gets the word's slot label
            4. Remaining subwords get IGNORE_INDEX (-100)
            5. [CLS] and [SEP] tokens also get IGNORE_INDEX
        """
        words = text.split()
        slot_tags = slot_labels

        # Ensure alignment between words and slot tags
        if len(words) != len(slot_tags):
            min_len = min(len(words), len(slot_tags))
            words = words[:min_len]
            slot_tags = slot_tags[:min_len]

        # Tokenize each word and align slot labels
        all_subword_ids = []
        aligned_slot_ids = []

        for word, slot_tag in zip(words, slot_tags):
            # Tokenize the word into subwords
            subword_tokens = self.tokenizer.tokenize(word)
            if not subword_tokens:
                # Handle unknown tokens
                subword_tokens = [self.tokenizer.unk_token]

            subword_ids = self.tokenizer.convert_tokens_to_ids(subword_tokens)
            all_subword_ids.extend(subword_ids)

            # First-token strategy: only first subword gets the label
            slot_id = self.slot2id.get(slot_tag, self.slot2id.get("UNK", 1))
            aligned_slot_ids.append(slot_id)
            # Remaining subwords get ignore index
            aligned_slot_ids.extend([self.IGNORE_INDEX] * (len(subword_ids) - 1))

        # Truncate to max_seq_length - 2 (reserve space for [CLS] and [SEP])
        max_tokens = self.max_seq_length - 2
        all_subword_ids = all_subword_ids[:max_tokens]
        aligned_slot_ids = aligned_slot_ids[:max_tokens]

        # Add special tokens
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

        # Encode intent
        intent_id = self.intent2id.get(intent, self.intent2id.get("UNK", 0))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "intent_label": intent_id,
            "slot_labels": slot_ids,
        }


class JointBERTDataModule:
    """Data module for loading and preparing JointBERT datasets.

    Handles:
    - Loading processed PhoATIS data
    - Creating train/dev/test datasets
    - Creating DataLoaders with proper collation

    Args:
        data_dir: Path to processed data directory
        model_name: HuggingFace model name for tokenizer
        max_seq_length: Maximum sequence length
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for DataLoader
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        model_name: str = "vinai/phobert-base-v2",
        max_seq_length: int = 128,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        self.data_dir = Path(data_dir) if data_dir else PROCESSED_DIR
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load label mappings
        self.intent2id = self._load_json("intent2id.json")
        self.slot2id = self._load_json("slot2id.json")
        self.id2intent = {v: k for k, v in self.intent2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}

        # Lazy load tokenizer
        self._tokenizer = None

        # Dataset cache
        self._datasets: Dict[str, JointBERTDataset] = {}

    @property
    def tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def num_intents(self) -> int:
        return len(self.intent2id)

    @property
    def num_slots(self) -> int:
        return len(self.slot2id)

    def _load_json(self, filename: str) -> Dict:
        """Load a JSON file from data directory."""
        with open(self.data_dir / filename, encoding="utf-8") as f:
            return json.load(f)

    def _read_split(self, split: str) -> Tuple[List[str], List[List[str]], List[str]]:
        """Read data from a split directory."""
        split_dir = self.data_dir / split

        texts = (split_dir / "seq.in").read_text(encoding="utf-8").strip().split("\n")
        slots_raw = (split_dir / "seq.out").read_text(encoding="utf-8").strip().split("\n")
        intents = (split_dir / "label").read_text(encoding="utf-8").strip().split("\n")

        slot_labels = [s.split() for s in slots_raw]

        return texts, slot_labels, intents

    def get_dataset(self, split: str) -> JointBERTDataset:
        """Get dataset for a split (cached).

        Args:
            split: One of "train", "dev", "test"

        Returns:
            JointBERTDataset instance
        """
        if split not in self._datasets:
            texts, slot_labels, intents = self._read_split(split)
            self._datasets[split] = JointBERTDataset(
                texts=texts,
                slot_labels=slot_labels,
                intent_labels=intents,
                tokenizer=self.tokenizer,
                slot2id=self.slot2id,
                intent2id=self.intent2id,
                max_seq_length=self.max_seq_length,
            )
        return self._datasets[split]

    def get_dataloader(
        self,
        split: str,
        shuffle: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """Get DataLoader for a split.

        Args:
            split: One of "train", "dev", "test"
            shuffle: Whether to shuffle (defaults to True for train)
            batch_size: Override default batch size

        Returns:
            DataLoader instance
        """
        dataset = self.get_dataset(split)

        if shuffle is None:
            shuffle = (split == "train")

        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return self.get_dataloader("train", shuffle=True)

    def get_dev_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return self.get_dataloader("dev", shuffle=False)

    def get_test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        return self.get_dataloader("test", shuffle=False)


def create_data_module(
    data_dir: Optional[Path] = None,
    model_name: str = "vinai/phobert-base-v2",
    max_seq_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
) -> JointBERTDataModule:
    """Factory function to create JointBERTDataModule.

    Args:
        data_dir: Path to processed data directory
        model_name: HuggingFace model name for tokenizer
        max_seq_length: Maximum sequence length
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for DataLoader

    Returns:
        JointBERTDataModule instance
    """
    return JointBERTDataModule(
        data_dir=data_dir,
        model_name=model_name,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        num_workers=num_workers,
    )
