"""Combined SVM + CRF NLU Pipeline for Vietnamese.

Combines SVMIntentClassifier and CRFSlotFiller into a unified NLU module
that takes text input and returns structured intent + slots output.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .crf_slot import CRFSlotFiller
from .svm_intent import SVMIntentClassifier


class SVMNLU:
    """Combined SVM-based NLU pipeline.

    Combines:
    - SVMIntentClassifier: TF-IDF + LinearSVC for intent classification
    - CRFSlotFiller: CRF for BIO slot labeling

    Usage:
        nlu = SVMNLU()
        nlu.fit(texts, intent_ids, slot_labels, id2intent)
        result = nlu.predict("đặt vé đi đà nẵng")
        # {"intent": "book_flight", "confidence": 0.95, "slots": {"toloc.city_name": "đà nẵng"}}
    """

    def __init__(
        self,
        intent_classifier: Optional[SVMIntentClassifier] = None,
        slot_filler: Optional[CRFSlotFiller] = None,
    ):
        """Initialize the NLU pipeline.

        Args:
            intent_classifier: pre-initialized intent classifier (optional)
            slot_filler: pre-initialized slot filler (optional)
        """
        self.intent_classifier = intent_classifier or SVMIntentClassifier()
        self.slot_filler = slot_filler or CRFSlotFiller()

        self._is_trained = False

    def fit(
        self,
        texts: List[str],
        intent_ids: List[int],
        slot_labels: List[List[str]],
        id2intent: Dict[int, str],
    ) -> "SVMNLU":
        """Train both intent classifier and slot filler.

        Args:
            texts: list of input sentences
            intent_ids: list of integer intent labels
            slot_labels: list of BIO tag sequences
            id2intent: mapping from intent ID to intent name

        Returns:
            self (for chaining)
        """
        # Train intent classifier
        self.intent_classifier.fit(texts, intent_ids, id2intent)

        # Train slot filler
        self.slot_filler.fit(texts, slot_labels)

        self._is_trained = True
        return self

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent and slots for a single text.

        Args:
            text: input sentence

        Returns:
            dict with keys:
                - intent: predicted intent name
                - confidence: intent prediction confidence (0-1)
                - slots: dict of slot_type -> slot_value
                - slot_labels: list of BIO tags
                - words: list of tokenized words
        """
        if not self._is_trained:
            raise RuntimeError("NLU not trained. Call fit() first.")

        # Intent classification
        intent, confidence = self.intent_classifier.predict_single(text)

        # Slot filling
        words, slot_tags = self.slot_filler.predict_single(text)
        slots = self.slot_filler.extract_slots(text)

        return {
            "intent": intent,
            "confidence": confidence,
            "slots": slots,
            "slot_labels": slot_tags,
            "words": words,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict intent and slots for multiple texts.

        Args:
            texts: list of input sentences

        Returns:
            list of prediction dicts
        """
        return [self.predict(text) for text in texts]

    def evaluate(
        self,
        texts: List[str],
        intent_ids: List[int],
        slot_labels: List[List[str]],
    ) -> Dict[str, Any]:
        """Evaluate both intent and slot performance.

        Args:
            texts: list of test sentences
            intent_ids: list of true intent IDs
            slot_labels: list of true BIO tag sequences

        Returns:
            dict with intent and slot evaluation metrics
        """
        if not self._is_trained:
            raise RuntimeError("NLU not trained. Call fit() first.")

        # Evaluate intent classifier
        intent_results = self.intent_classifier.evaluate(texts, intent_ids)

        # Evaluate slot filler
        slot_results = self.slot_filler.evaluate(texts, slot_labels)

        # Calculate sentence accuracy (both intent and all slots correct)
        sentence_correct = 0
        intent_preds = intent_results["predictions"]
        slot_preds = slot_results["predictions"]

        # Align slot predictions with original labels for comparison
        for i, (text, true_intent, true_slots) in enumerate(
            zip(texts, intent_ids, slot_labels)
        ):
            if i >= len(slot_preds):
                continue

            pred_intent = intent_preds[i]
            pred_slots = slot_preds[i]

            # Check intent match
            intent_match = pred_intent == true_intent

            # Check slot match (accounting for length differences)
            min_len = min(len(true_slots), len(pred_slots))
            slots_match = (
                len(true_slots) == len(pred_slots) and
                all(t == p for t, p in zip(true_slots[:min_len], pred_slots[:min_len]))
            )

            if intent_match and slots_match:
                sentence_correct += 1

        sentence_accuracy = sentence_correct / len(texts) if texts else 0

        return {
            "intent": {
                "accuracy": intent_results["accuracy"],
                "f1_macro": intent_results["f1_macro"],
                "classification_report": intent_results["classification_report"],
            },
            "slot": {
                "f1_weighted": slot_results["f1_weighted"],
                "f1_macro": slot_results["f1_macro"],
                "token_accuracy": slot_results["token_accuracy"],
                "classification_report": slot_results["classification_report"],
            },
            "sentence_accuracy": sentence_accuracy,
        }

    def save(self, model_dir: str) -> None:
        """Save both models to a directory.

        Args:
            model_dir: directory to save models
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained NLU.")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        self.intent_classifier.save(str(model_path / "svm_intent.joblib"))
        self.slot_filler.save(str(model_path / "crf_slot.joblib"))

    @classmethod
    def load(cls, model_dir: str) -> "SVMNLU":
        """Load both models from a directory.

        Args:
            model_dir: directory containing saved models

        Returns:
            loaded SVMNLU instance
        """
        model_path = Path(model_dir)

        intent_classifier = SVMIntentClassifier.load(
            str(model_path / "svm_intent.joblib")
        )
        slot_filler = CRFSlotFiller.load(str(model_path / "crf_slot.joblib"))

        instance = cls(
            intent_classifier=intent_classifier,
            slot_filler=slot_filler,
        )
        instance._is_trained = True

        return instance

    @property
    def num_intents(self) -> int:
        """Number of intent classes."""
        return self.intent_classifier.num_intents

    @property
    def num_slots(self) -> int:
        """Number of slot label types."""
        return self.slot_filler.num_slots

    def get_intent_labels(self) -> List[str]:
        """Get list of intent labels."""
        return list(self.intent_classifier.intent2id.keys())

    def get_slot_labels(self) -> List[str]:
        """Get list of slot labels."""
        return self.slot_filler.slot_labels
