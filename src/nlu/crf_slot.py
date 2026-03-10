"""CRF-based Slot Filler for Vietnamese NLU.

Uses sklearn-crfsuite for BIO sequence labeling with handcrafted features.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib

try:
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics as crf_metrics
except ImportError:
    sklearn_crfsuite = None
    crf_metrics = None

try:
    from underthesea import word_tokenize
except ImportError:
    def word_tokenize(text: str) -> str:
        return text


class CRFSlotFiller:
    """CRF-based slot filler for BIO sequence labeling.

    Features extracted for each word:
    - Word form (lowercase)
    - Prefixes and suffixes (2-3 chars)
    - Position in sentence
    - Capitalization features
    - Digit features
    - Context window (previous/next words)
    """

    def __init__(
        self,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
        algorithm: str = "lbfgs",
    ):
        """Initialize the CRF slot filler.

        Args:
            c1: L1 regularization coefficient
            c2: L2 regularization coefficient
            max_iterations: maximum training iterations
            algorithm: training algorithm (lbfgs, l2sgd, ap, pa, arow)
        """
        if sklearn_crfsuite is None:
            raise ImportError(
                "sklearn-crfsuite is required. Install with: pip install sklearn-crfsuite"
            )

        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.algorithm = algorithm

        # Model
        self.crf: Optional[sklearn_crfsuite.CRF] = None

        # Label info
        self.slot_labels: List[str] = []

        # Training flag
        self._is_trained = False

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words.

        For PhoATIS, text is already space-separated at syllable level,
        so we just split. We apply underthesea for potential compound words.
        """
        # First, split by whitespace (PhoATIS format)
        words = text.strip().split()
        return words

    def _word2features(
        self,
        words: List[str],
        position: int,
    ) -> Dict[str, str]:
        """Extract features for a word at given position.

        Args:
            words: list of words in the sentence
            position: index of the current word

        Returns:
            dict of feature_name -> feature_value
        """
        word = words[position]
        word_lower = word.lower()

        features = {
            "bias": "1.0",
            "word.lower": word_lower,
            "word[-3:]": word_lower[-3:] if len(word_lower) >= 3 else word_lower,
            "word[-2:]": word_lower[-2:] if len(word_lower) >= 2 else word_lower,
            "word[:3]": word_lower[:3] if len(word_lower) >= 3 else word_lower,
            "word[:2]": word_lower[:2] if len(word_lower) >= 2 else word_lower,
            "word.istitle": str(word.istitle()),
            "word.isupper": str(word.isupper()),
            "word.isdigit": str(word.isdigit()),
            "word.hasdigit": str(any(c.isdigit() for c in word)),
            "word.len": str(len(word)),
            "position": str(position),
            "position_relative": f"{position / len(words):.2f}",
        }

        # Add word length bucket
        if len(word) <= 2:
            features["word.len_bucket"] = "short"
        elif len(word) <= 5:
            features["word.len_bucket"] = "medium"
        else:
            features["word.len_bucket"] = "long"

        # Context: previous word
        if position > 0:
            prev_word = words[position - 1]
            prev_lower = prev_word.lower()
            features.update({
                "-1:word.lower": prev_lower,
                "-1:word.istitle": str(prev_word.istitle()),
                "-1:word.isdigit": str(prev_word.isdigit()),
                "-1:word[-2:]": prev_lower[-2:] if len(prev_lower) >= 2 else prev_lower,
            })
        else:
            features["BOS"] = "True"  # Beginning of sentence

        # Context: next word
        if position < len(words) - 1:
            next_word = words[position + 1]
            next_lower = next_word.lower()
            features.update({
                "+1:word.lower": next_lower,
                "+1:word.istitle": str(next_word.istitle()),
                "+1:word.isdigit": str(next_word.isdigit()),
                "+1:word[:2]": next_lower[:2] if len(next_lower) >= 2 else next_lower,
            })
        else:
            features["EOS"] = "True"  # End of sentence

        # Extended context: word at position -2
        if position > 1:
            features["-2:word.lower"] = words[position - 2].lower()

        # Extended context: word at position +2
        if position < len(words) - 2:
            features["+2:word.lower"] = words[position + 2].lower()

        return features

    def _sent2features(self, words: List[str]) -> List[Dict[str, str]]:
        """Extract features for all words in a sentence."""
        return [self._word2features(words, i) for i in range(len(words))]

    def fit(
        self,
        texts: List[str],
        slot_labels: List[List[str]],
    ) -> "CRFSlotFiller":
        """Train the CRF slot filler.

        Args:
            texts: list of input sentences
            slot_labels: list of BIO tag sequences (one per sentence)

        Returns:
            self (for chaining)
        """
        # Tokenize texts
        X_words = [self._tokenize_text(text) for text in texts]

        # Ensure alignment between words and labels
        X_features = []
        y_labels = []

        for words, labels in zip(X_words, slot_labels):
            # Handle length mismatch
            if len(words) != len(labels):
                min_len = min(len(words), len(labels))
                words = words[:min_len]
                labels = labels[:min_len]

            if len(words) == 0:
                continue

            X_features.append(self._sent2features(words))
            y_labels.append(labels)

        # Collect all unique labels
        all_labels = set()
        for labels in y_labels:
            all_labels.update(labels)
        self.slot_labels = sorted(all_labels)

        # Initialize and train CRF
        self.crf = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=True,
            verbose=False,
        )

        self.crf.fit(X_features, y_labels)
        self._is_trained = True

        return self

    def predict(self, texts: List[str]) -> List[List[str]]:
        """Predict slot labels for a batch of texts.

        Args:
            texts: list of input sentences

        Returns:
            list of BIO tag sequences
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_words = [self._tokenize_text(text) for text in texts]
        X_features = [self._sent2features(words) for words in X_words]

        return self.crf.predict(X_features)

    def predict_single(self, text: str) -> Tuple[List[str], List[str]]:
        """Predict slots for a single text.

        Args:
            text: input sentence

        Returns:
            (words, slot_labels) tuple
        """
        words = self._tokenize_text(text)
        if len(words) == 0:
            return [], []

        features = [self._sent2features(words)]
        labels = self.crf.predict(features)[0]

        return words, labels

    def extract_slots(self, text: str) -> Dict[str, str]:
        """Extract slot values from text.

        Converts BIO tags to a dict of slot_type -> slot_value.

        Args:
            text: input sentence

        Returns:
            dict mapping slot types to their values
        """
        words, labels = self.predict_single(text)

        slots = {}
        current_slot = None
        current_value = []

        for word, label in zip(words, labels):
            if label.startswith("B-"):
                # Save previous slot if exists
                if current_slot is not None:
                    slots[current_slot] = " ".join(current_value)

                # Start new slot
                current_slot = label[2:]  # Remove "B-" prefix
                current_value = [word]

            elif label.startswith("I-") and current_slot is not None:
                # Continue current slot
                slot_type = label[2:]  # Remove "I-" prefix
                if slot_type == current_slot:
                    current_value.append(word)
                else:
                    # Mismatched I- tag, save and start new
                    slots[current_slot] = " ".join(current_value)
                    current_slot = slot_type
                    current_value = [word]

            else:
                # O tag or mismatched I- tag
                if current_slot is not None:
                    slots[current_slot] = " ".join(current_value)
                    current_slot = None
                    current_value = []

        # Don't forget the last slot
        if current_slot is not None:
            slots[current_slot] = " ".join(current_value)

        return slots

    def evaluate(
        self,
        texts: List[str],
        slot_labels: List[List[str]],
    ) -> Dict:
        """Evaluate the slot filler on a test set.

        Args:
            texts: list of test sentences
            slot_labels: list of true BIO tag sequences

        Returns:
            dict with F1 scores and detailed metrics
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Tokenize and align
        X_words = [self._tokenize_text(text) for text in texts]

        y_true = []
        y_pred_all = []

        for words, labels in zip(X_words, slot_labels):
            # Handle length mismatch
            if len(words) != len(labels):
                min_len = min(len(words), len(labels))
                words = words[:min_len]
                labels = labels[:min_len]

            if len(words) == 0:
                continue

            y_true.append(labels)

        # Predict
        X_features = [self._sent2features(words) for words in X_words if len(words) > 0]
        y_pred = self.crf.predict(X_features)

        # Get labels excluding O for entity-level metrics
        labels = list(self.crf.classes_)
        labels_no_o = [l for l in labels if l != "O"]

        # Calculate F1 scores
        f1_weighted = crf_metrics.flat_f1_score(
            y_true, y_pred, average="weighted", labels=labels_no_o, zero_division=0
        )
        f1_macro = crf_metrics.flat_f1_score(
            y_true, y_pred, average="macro", labels=labels_no_o, zero_division=0
        )

        # Token-level accuracy
        y_true_flat = [label for seq in y_true for label in seq]
        y_pred_flat = [label for seq in y_pred for label in seq]
        token_accuracy = sum(
            1 for t, p in zip(y_true_flat, y_pred_flat) if t == p
        ) / len(y_true_flat) if y_true_flat else 0

        # Detailed classification report
        report = crf_metrics.flat_classification_report(
            y_true, y_pred, labels=labels_no_o, zero_division=0
        )

        return {
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "token_accuracy": token_accuracy,
            "classification_report": report,
            "predictions": y_pred,
        }

    def save(self, filepath: str) -> None:
        """Save the trained model to disk.

        Args:
            filepath: path to save the model (e.g., 'models/crf_slot.joblib')
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model.")

        model_data = {
            "crf": self.crf,
            "slot_labels": self.slot_labels,
            "config": {
                "c1": self.c1,
                "c2": self.c2,
                "max_iterations": self.max_iterations,
                "algorithm": self.algorithm,
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CRFSlotFiller":
        """Load a trained model from disk.

        Args:
            filepath: path to the saved model

        Returns:
            loaded CRFSlotFiller instance
        """
        model_data = joblib.load(filepath)

        config = model_data.get("config", {})
        instance = cls(
            c1=config.get("c1", 0.1),
            c2=config.get("c2", 0.1),
            max_iterations=config.get("max_iterations", 100),
            algorithm=config.get("algorithm", "lbfgs"),
        )

        instance.crf = model_data["crf"]
        instance.slot_labels = model_data["slot_labels"]
        instance._is_trained = True

        return instance

    @property
    def num_slots(self) -> int:
        """Number of slot label types."""
        return len(self.slot_labels)

    def get_transition_features(self, top_k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """Get top transition features from the trained CRF.

        Returns:
            dict with 'positive' and 'negative' transition weights
        """
        if not self._is_trained or self.crf is None:
            return {"positive": [], "negative": []}

        try:
            transitions = self.crf.transition_features_

            positive = sorted(
                transitions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            negative = sorted(
                transitions.items(),
                key=lambda x: x[1],
            )[:top_k]

            return {
                "positive": [(str(k), float(v)) for k, v in positive],
                "negative": [(str(k), float(v)) for k, v in negative],
            }
        except AttributeError:
            return {"positive": [], "negative": []}
