"""Inference wrapper for JointBERT NLU model.

Provides a high-level interface for using trained JointBERT model
for intent classification and slot filling.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from .jointbert_model import JointBERTModel, create_model


# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class JointBERTNLU:
    """High-level NLU inference wrapper for JointBERT model.

    Loads a trained JointBERT model and provides easy-to-use methods
    for intent classification and slot filling on raw text input.

    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to processed data directory (for label mappings)
        model_name: HuggingFace model name for tokenizer
        max_seq_length: Maximum sequence length
        device: Device to run inference on
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        data_dir: Optional[Path] = None,
        model_name: str = "vinai/phobert-base-v2",
        max_seq_length: int = 128,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load label mappings
        self.data_dir = Path(data_dir) if data_dir else PROCESSED_DIR
        self._load_label_mappings()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        self.model = None
        if model_path:
            self.load_model(model_path)

    def _load_label_mappings(self) -> None:
        """Load intent and slot label mappings from data directory."""
        with open(self.data_dir / "intent2id.json", encoding="utf-8") as f:
            self.intent2id = json.load(f)
        self.id2intent = {v: k for k, v in self.intent2id.items()}

        with open(self.data_dir / "slot2id.json", encoding="utf-8") as f:
            self.slot2id = json.load(f)
        self.id2slot = {v: k for k, v in self.slot2id.items()}

    def load_model(self, model_path: str) -> None:
        """Load trained model from checkpoint.

        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        # Get model config from checkpoint
        model_config = checkpoint.get("model_config", {})
        num_intents = model_config.get("num_intents", len(self.intent2id))
        num_slots = model_config.get("num_slots", len(self.slot2id))

        # Create model
        self.model = create_model(
            model_name=self.model_name,
            num_intents=num_intents,
            num_slots=num_slots,
            use_crf=model_config.get("use_crf", False),
        )

        # Load state dict
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from: {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Intents: {num_intents}")
        print(f"  Slots: {num_slots}")

    def predict(self, text: str) -> Dict:
        """Predict intent and slots for a single text input.

        Args:
            text: Input text (Vietnamese sentence)

        Returns:
            Dictionary containing:
                - intent: Predicted intent label
                - confidence: Intent prediction confidence (0-1)
                - slots: Dictionary of {slot_type: value}
                - raw_slots: List of (token, slot_label) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize input
        encoded = self._encode_text(text)

        # Run inference
        with torch.no_grad():
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Intent prediction
            intent_probs = torch.softmax(outputs["intent_logits"], dim=-1)
            intent_idx = torch.argmax(intent_probs, dim=-1).item()
            intent_confidence = intent_probs[0, intent_idx].item()
            intent_label = self.id2intent.get(intent_idx, "UNK")

            # Slot prediction
            slot_preds = torch.argmax(outputs["slot_logits"], dim=-1)[0]

        # Extract slots from predictions
        raw_slots, slot_dict = self._extract_slots(text, encoded, slot_preds.cpu().numpy())

        return {
            "intent": intent_label,
            "confidence": intent_confidence,
            "slots": slot_dict,
            "raw_slots": raw_slots,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict intent and slots for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]

    def _encode_text(self, text: str) -> Dict[str, List[int]]:
        """Encode text for model input.

        Args:
            text: Input text

        Returns:
            Dictionary with input_ids, attention_mask, and word_ids
        """
        words = text.split()

        # Track word boundaries for slot extraction
        all_subword_ids = []
        word_ids = []  # Track which word each subword belongs to

        for word_idx, word in enumerate(words):
            subword_tokens = self.tokenizer.tokenize(word)
            if not subword_tokens:
                subword_tokens = [self.tokenizer.unk_token]

            subword_ids = self.tokenizer.convert_tokens_to_ids(subword_tokens)
            all_subword_ids.extend(subword_ids)
            word_ids.extend([word_idx] * len(subword_ids))

        # Truncate
        max_tokens = self.max_seq_length - 2
        all_subword_ids = all_subword_ids[:max_tokens]
        word_ids = word_ids[:max_tokens]

        # Add special tokens
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        input_ids = [cls_id] + all_subword_ids + [sep_id]
        word_ids = [-1] + word_ids + [-1]  # -1 for special tokens
        attention_mask = [1] * len(input_ids)

        # Pad
        pad_length = self.max_seq_length - len(input_ids)
        pad_id = self.tokenizer.pad_token_id

        input_ids += [pad_id] * pad_length
        word_ids += [-1] * pad_length
        attention_mask += [0] * pad_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_ids": word_ids,
            "words": words,
        }

    def _extract_slots(
        self,
        text: str,
        encoded: Dict,
        slot_preds: List[int],
    ) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
        """Extract slot values from predictions.

        Uses first-subword labels to assign slots to original words.
        Handles BIO tagging to group multi-word slot values.

        Args:
            text: Original input text
            encoded: Encoded input with word_ids
            slot_preds: Predicted slot IDs for each token

        Returns:
            Tuple of:
                - raw_slots: List of (token, slot_label) tuples
                - slot_dict: Dictionary of {slot_type: value}
        """
        words = encoded["words"]
        word_ids = encoded["word_ids"]

        # Get first subword label for each word
        word_slot_labels = {}
        for token_idx, (word_idx, slot_id) in enumerate(zip(word_ids, slot_preds)):
            if word_idx >= 0 and word_idx not in word_slot_labels:
                # First subword for this word
                word_slot_labels[word_idx] = self.id2slot.get(slot_id, "O")

        # Build raw slots list
        raw_slots = []
        for word_idx, word in enumerate(words):
            label = word_slot_labels.get(word_idx, "O")
            raw_slots.append((word, label))

        # Extract slot values using BIO tagging
        slot_dict = {}
        current_slot_type = None
        current_slot_value = []

        for word, label in raw_slots:
            if label.startswith("B-"):
                # Save previous slot if any
                if current_slot_type and current_slot_value:
                    slot_dict[current_slot_type] = " ".join(current_slot_value)

                # Start new slot
                current_slot_type = label[2:]  # Remove "B-" prefix
                current_slot_value = [word]

            elif label.startswith("I-"):
                slot_type = label[2:]  # Remove "I-" prefix
                # Continue current slot if types match
                if current_slot_type == slot_type:
                    current_slot_value.append(word)
                else:
                    # Type mismatch - save previous and start new
                    if current_slot_type and current_slot_value:
                        slot_dict[current_slot_type] = " ".join(current_slot_value)
                    current_slot_type = slot_type
                    current_slot_value = [word]

            else:  # "O" or other
                # Save previous slot if any
                if current_slot_type and current_slot_value:
                    slot_dict[current_slot_type] = " ".join(current_slot_value)
                current_slot_type = None
                current_slot_value = []

        # Save final slot if any
        if current_slot_type and current_slot_value:
            slot_dict[current_slot_type] = " ".join(current_slot_value)

        return raw_slots, slot_dict

    def get_intent_probs(self, text: str) -> Dict[str, float]:
        """Get probability distribution over all intents.

        Args:
            text: Input text

        Returns:
            Dictionary mapping intent labels to probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        encoded = self._encode_text(text)

        with torch.no_grad():
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["intent_logits"], dim=-1)[0]

        return {
            self.id2intent[i]: prob.item()
            for i, prob in enumerate(probs)
        }

    def get_top_intents(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k most likely intents.

        Args:
            text: Input text
            top_k: Number of top intents to return

        Returns:
            List of (intent, probability) tuples, sorted by probability
        """
        probs = self.get_intent_probs(text)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:top_k]


def load_nlu(
    model_path: Optional[str] = None,
    data_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> JointBERTNLU:
    """Load JointBERT NLU model.

    Convenience function to create and load a JointBERTNLU instance.

    Args:
        model_path: Path to model checkpoint (defaults to best_jointbert.pt)
        data_dir: Path to processed data directory
        device: Device to run on

    Returns:
        Loaded JointBERTNLU instance
    """
    if model_path is None:
        model_path = str(MODELS_DIR / "best_jointbert.pt")

    nlu = JointBERTNLU(
        model_path=model_path,
        data_dir=data_dir,
        device=device,
    )

    return nlu
