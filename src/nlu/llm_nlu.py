"""LLM-based NLU classifier for Vietnamese air travel domain.

Provides zero-shot and few-shot NLU using large language models.
Supports Claude (Anthropic), GPT (OpenAI), and mock client for testing.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import BaseLLMClient, get_client
from .llm_prompts import (
    get_system_prompt,
    format_user_prompt,
    VALID_INTENTS,
    VALID_SLOT_TYPES,
)


class LLMNLUClassifier:
    """Zero-shot NLU classifier using LLMs.

    Extracts intent and slot values from Vietnamese utterances
    using large language models with few-shot prompting.

    Usage:
        >>> classifier = LLMNLUClassifier(provider="anthropic")
        >>> result = classifier.predict("tôi muốn bay từ hà nội đến đà nẵng")
        >>> print(result)
        {"intent": "flight", "confidence": 0.95, "slots": {"fromloc.city_name": "hà nội", ...}}
    """

    def __init__(
        self,
        client: Optional[BaseLLMClient] = None,
        provider: str = "mock",
        model: Optional[str] = None,
        include_examples: bool = True,
        include_definitions: bool = True,
        rate_limit_delay: float = 0.5,
        **kwargs,
    ):
        """Initialize LLM NLU classifier.

        Args:
            client: Pre-configured LLM client (optional)
            provider: Provider name if client not given ("anthropic", "openai", "mock")
            model: Model name override
            include_examples: Whether to include few-shot examples in prompt
            include_definitions: Whether to include detailed definitions in system prompt
            rate_limit_delay: Delay between API calls (seconds)
            **kwargs: Additional arguments passed to client
        """
        self.client = client or get_client(provider, model=model, **kwargs)
        self.include_examples = include_examples
        self.include_definitions = include_definitions
        self.rate_limit_delay = rate_limit_delay

        # Cache system prompt
        self._system_prompt = get_system_prompt(include_definitions)

        # Statistics
        self.total_calls = 0
        self.total_time = 0.0
        self.errors = 0

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent and slots for a single utterance.

        Args:
            text: Vietnamese utterance

        Returns:
            Dictionary with:
                - intent: predicted intent label
                - confidence: confidence score (0-1)
                - slots: dictionary of slot_type -> value
        """
        start_time = time.time()
        self.total_calls += 1

        try:
            user_message = format_user_prompt(text, include_examples=self.include_examples)

            result = self.client.complete(
                system_prompt=self._system_prompt,
                user_message=user_message,
            )

            elapsed = time.time() - start_time
            self.total_time += elapsed

            return result

        except Exception as e:
            self.errors += 1
            # Return fallback on error
            return {
                "intent": "flight",
                "confidence": 0.0,
                "slots": {},
                "error": str(e),
            }

    def predict_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Predict intent and slots for a batch of utterances.

        Args:
            texts: List of Vietnamese utterances
            show_progress: Whether to show progress bar

        Returns:
            List of prediction dictionaries
        """
        results = []

        iterator = texts
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="LLM NLU", unit="sample")
            except ImportError:
                pass

        for i, text in enumerate(iterator):
            result = self.predict(text)
            results.append(result)

            # Rate limiting (skip for last item)
            if i < len(texts) - 1 and self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

        return results

    def predict_with_bio_slots(self, text: str) -> Tuple[str, float, List[str]]:
        """Predict with BIO-format slot labels for evaluation.

        Converts slot value extraction to BIO tagging format
        to enable direct comparison with trained models.

        Args:
            text: Vietnamese utterance

        Returns:
            Tuple of (intent, confidence, bio_labels)
            where bio_labels is a list of BIO tags aligned with words
        """
        result = self.predict(text)
        intent = result["intent"]
        confidence = result["confidence"]
        slots = result["slots"]

        # Convert to BIO format
        words = text.split()
        bio_labels = ["O"] * len(words)

        for slot_type, slot_value in slots.items():
            # Validate slot type
            if slot_type not in VALID_SLOT_TYPES:
                continue

            # Find slot value in text
            slot_words = slot_value.lower().split()
            if not slot_words:
                continue

            # Search for matching span using substring search first
            slot_value_lower = slot_value.lower()
            text_lower = text.lower()

            # Try to find exact substring match
            start_idx = text_lower.find(slot_value_lower)
            if start_idx != -1:
                # Count words before the match to find word index
                prefix = text[:start_idx]
                word_start_idx = len(prefix.split()) - (1 if prefix.endswith(" ") or not prefix else 0)
                if prefix == "":
                    word_start_idx = 0
                elif prefix.endswith(" "):
                    word_start_idx = len(prefix.split())

                # Calculate how many words the slot value spans
                num_slot_words = len(slot_words)

                # Validate indices
                if word_start_idx >= 0 and word_start_idx + num_slot_words <= len(words):
                    # Only assign if not already labeled (prioritize earlier slots)
                    if bio_labels[word_start_idx] == "O":
                        bio_labels[word_start_idx] = f"B-{slot_type}"
                        for j in range(1, num_slot_words):
                            if word_start_idx + j < len(bio_labels):
                                bio_labels[word_start_idx + j] = f"I-{slot_type}"
                continue

            # Fallback: word-by-word matching
            for i in range(len(words) - len(slot_words) + 1):
                # Check if words match (case-insensitive)
                match = True
                for j, sw in enumerate(slot_words):
                    if words[i + j].lower() != sw:
                        match = False
                        break

                if match:
                    # Assign BIO labels (only if not already labeled)
                    if bio_labels[i] == "O":
                        bio_labels[i] = f"B-{slot_type}"
                        for j in range(1, len(slot_words)):
                            bio_labels[i + j] = f"I-{slot_type}"
                    break  # Only label first occurrence

        return intent, confidence, bio_labels

    def predict_batch_with_bio(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[Tuple[str, float, List[str]]]:
        """Predict batch with BIO-format slot labels.

        Args:
            texts: List of Vietnamese utterances
            show_progress: Whether to show progress bar

        Returns:
            List of (intent, confidence, bio_labels) tuples
        """
        results = []

        iterator = texts
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="LLM NLU (BIO)", unit="sample")
            except ImportError:
                pass

        for i, text in enumerate(iterator):
            result = self.predict_with_bio_slots(text)
            results.append(result)

            if i < len(texts) - 1 and self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with call counts, timing, and error rate
        """
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0
        error_rate = self.errors / self.total_calls if self.total_calls > 0 else 0

        return {
            "total_calls": self.total_calls,
            "total_time": round(self.total_time, 2),
            "average_time": round(avg_time, 3),
            "errors": self.errors,
            "error_rate": round(error_rate, 3),
        }

    def reset_statistics(self):
        """Reset usage statistics."""
        self.total_calls = 0
        self.total_time = 0.0
        self.errors = 0

    @property
    def intent_labels(self) -> List[str]:
        """Get list of valid intent labels."""
        return VALID_INTENTS

    @property
    def slot_types(self) -> List[str]:
        """Get list of valid slot types."""
        return VALID_SLOT_TYPES


def create_llm_classifier(
    provider: str = "mock",
    config_path: Optional[str] = None,
) -> LLMNLUClassifier:
    """Create LLM classifier from config file or defaults.

    Args:
        provider: LLM provider name
        config_path: Path to YAML config file

    Returns:
        Configured LLMNLUClassifier instance
    """
    config = {}

    if config_path:
        try:
            import yaml
            from pathlib import Path

            config = yaml.safe_load(Path(config_path).read_text())
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")

    # Extract provider-specific settings
    provider_config = config.get("models", {}).get(provider, {})
    model = provider_config.get("model")
    max_tokens = provider_config.get("max_tokens", 512)
    temperature = provider_config.get("temperature", 0.0)

    # Evaluation settings
    eval_config = config.get("evaluation", {})
    rate_limit_delay = eval_config.get("delay_seconds", 0.5)

    return LLMNLUClassifier(
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        rate_limit_delay=rate_limit_delay,
    )
