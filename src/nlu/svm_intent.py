"""TF-IDF + SVM Intent Classifier for Vietnamese NLU.

Uses underthesea for Vietnamese word segmentation and sklearn's
TfidfVectorizer + LinearSVC for classification.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC
from scipy.special import softmax

try:
    from underthesea import word_tokenize
except ImportError:
    # Fallback: simple whitespace tokenization
    def word_tokenize(text: str) -> str:
        return text


class SVMWithProba:
    """Wrapper around LinearSVC that provides probability estimates.

    Uses softmax over decision function scores to approximate probabilities.
    """

    def __init__(self, svm: LinearSVC):
        self.svm = svm
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y):
        self.svm.fit(X, y)
        self.classes_ = self.svm.classes_
        return self

    def predict(self, X):
        return self.svm.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Approximate probabilities using softmax over decision scores."""
        decision = self.svm.decision_function(X)

        # Handle binary vs multiclass
        if len(decision.shape) == 1:
            # Binary classification: decision is 1D
            # Convert to 2D with scores for both classes
            decision = np.column_stack([-decision, decision])

        # Apply softmax to get probability-like scores
        proba = softmax(decision, axis=1)
        return proba


class SVMIntentClassifier:
    """TF-IDF + SVM classifier for intent detection.

    Features:
    - Vietnamese word segmentation via underthesea
    - TF-IDF with unigrams and bigrams
    - Probability estimates via softmax over decision scores
    - Save/load with joblib
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 10000,
        svm_c: float = 1.0,
        class_weight: str = "balanced",
    ):
        """Initialize the intent classifier.

        Args:
            ngram_range: n-gram range for TF-IDF (default: unigrams + bigrams)
            max_features: maximum vocabulary size
            svm_c: SVM regularization parameter
            class_weight: class weighting strategy
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.svm_c = svm_c
        self.class_weight = class_weight

        # Models (initialized during training)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[SVMWithProba] = None

        # Label mappings
        self.intent2id: Dict[str, int] = {}
        self.id2intent: Dict[int, str] = {}

        # Training flag
        self._is_trained = False

    def _tokenize(self, text: str) -> str:
        """Tokenize Vietnamese text using underthesea.

        Returns space-separated tokens for TfidfVectorizer.
        """
        # underthesea.word_tokenize returns space-separated string
        # or list depending on version - handle both
        result = word_tokenize(text)
        if isinstance(result, list):
            return " ".join(result)
        return result

    def _preprocess(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        return [self._tokenize(text) for text in texts]

    def fit(
        self,
        texts: List[str],
        intent_ids: List[int],
        id2intent: Dict[int, str],
    ) -> "SVMIntentClassifier":
        """Train the intent classifier.

        Args:
            texts: list of input sentences
            intent_ids: list of integer intent labels
            id2intent: mapping from intent ID to intent name

        Returns:
            self (for chaining)
        """
        # Store mappings
        self.id2intent = id2intent
        self.intent2id = {v: k for k, v in id2intent.items()}

        # Preprocess texts
        processed_texts = self._preprocess(texts)

        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,  # Use log(tf) for better performance
            min_df=2,  # Ignore rare terms
            strip_accents=None,  # Keep Vietnamese diacritics
            lowercase=True,
        )

        X = self.vectorizer.fit_transform(processed_texts)
        y = np.array(intent_ids)

        # Initialize and train SVM with probability wrapper
        base_svm = LinearSVC(
            C=self.svm_c,
            class_weight=self.class_weight,
            max_iter=10000,
            random_state=42,
            dual="auto",
        )

        self.classifier = SVMWithProba(base_svm)
        self.classifier.fit(X, y)

        self._is_trained = True
        return self

    def predict(self, texts: List[str]) -> List[int]:
        """Predict intent IDs for a batch of texts.

        Args:
            texts: list of input sentences

        Returns:
            list of predicted intent IDs
        """
        if not self._is_trained:
            raise RuntimeError("Classifier not trained. Call fit() first.")

        processed_texts = self._preprocess(texts)
        X = self.vectorizer.transform(processed_texts)
        return self.classifier.predict(X).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict class probabilities for a batch of texts.

        Args:
            texts: list of input sentences

        Returns:
            array of shape (n_samples, n_classes) with probabilities
        """
        if not self._is_trained:
            raise RuntimeError("Classifier not trained. Call fit() first.")

        processed_texts = self._preprocess(texts)
        X = self.vectorizer.transform(processed_texts)
        return self.classifier.predict_proba(X)

    def predict_single(self, text: str) -> Tuple[str, float]:
        """Predict intent and confidence for a single text.

        Args:
            text: input sentence

        Returns:
            (intent_name, confidence) tuple
        """
        proba = self.predict_proba([text])[0]
        pred_id = int(np.argmax(proba))
        confidence = float(proba[pred_id])

        # Map predicted class to intent name
        # Note: SVM classes_ may not be contiguous with our id2intent
        actual_class = self.classifier.classes_[pred_id]
        intent_name = self.id2intent.get(actual_class, "UNK")

        return intent_name, confidence

    def evaluate(
        self,
        texts: List[str],
        intent_ids: List[int],
    ) -> Dict:
        """Evaluate the classifier on a test set.

        Args:
            texts: list of test sentences
            intent_ids: list of true intent IDs

        Returns:
            dict with accuracy, f1_macro, and classification_report
        """
        if not self._is_trained:
            raise RuntimeError("Classifier not trained. Call fit() first.")

        predictions = self.predict(texts)

        accuracy = accuracy_score(intent_ids, predictions)
        f1_macro = f1_score(intent_ids, predictions, average="macro", zero_division=0)

        # Get label names for report
        labels = sorted(set(intent_ids) | set(predictions))
        target_names = [self.id2intent.get(i, f"class_{i}") for i in labels]

        report = classification_report(
            intent_ids,
            predictions,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "classification_report": report,
            "predictions": predictions,
        }

    def save(self, filepath: str) -> None:
        """Save the trained model to disk.

        Args:
            filepath: path to save the model (e.g., 'models/svm_intent.joblib')
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model.")

        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "intent2id": self.intent2id,
            "id2intent": self.id2intent,
            "config": {
                "ngram_range": self.ngram_range,
                "max_features": self.max_features,
                "svm_c": self.svm_c,
                "class_weight": self.class_weight,
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "SVMIntentClassifier":
        """Load a trained model from disk.

        Args:
            filepath: path to the saved model

        Returns:
            loaded SVMIntentClassifier instance
        """
        model_data = joblib.load(filepath)

        config = model_data.get("config", {})
        instance = cls(
            ngram_range=config.get("ngram_range", (1, 2)),
            max_features=config.get("max_features", 10000),
            svm_c=config.get("svm_c", 1.0),
            class_weight=config.get("class_weight", "balanced"),
        )

        instance.vectorizer = model_data["vectorizer"]
        instance.classifier = model_data["classifier"]
        instance.intent2id = model_data["intent2id"]
        instance.id2intent = model_data["id2intent"]
        instance._is_trained = True

        return instance

    @property
    def num_intents(self) -> int:
        """Number of intent classes."""
        return len(self.intent2id)

    @property
    def feature_names(self) -> List[str]:
        """Get feature names from the vectorizer."""
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out().tolist()
