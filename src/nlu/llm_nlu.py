"""LLM-based NLU classifier for Vietnamese air travel domain.

Provides zero-shot and few-shot NLU using large language models.
Supports Claude (Anthropic), GPT (OpenAI), Gemini (Google), and mock client for testing.

Key improvements over v1:
- LLM is prompted to return BIO tags directly (no reverse-engineering)
- Vietnamese diacritic normalization for robust slot matching fallback
- Correct word-boundary alignment in BIO conversion
- BIO sequence validation and auto-repair
- Robust JSON parsing with multiple fallback strategies
"""

import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import BaseLLMClient, get_client
from .llm_prompts import (
    get_system_prompt,
    format_user_prompt,
    VALID_INTENTS,
    VALID_SLOT_TYPES,
)


# ---------------------------------------------------------------------------
# Vietnamese normalization helpers
# ---------------------------------------------------------------------------

def _normalize_vietnamese(text: str) -> str:
    """Normalize Vietnamese text for fuzzy matching.

    Performs the following:
    - Lowercase
    - NFC unicode normalization (canonical composition)
    - Strip leading/trailing whitespace
    - Collapse multiple spaces

    Does NOT strip diacritics — we use this for comparing two
    Vietnamese strings where both may have diacritics but differ
    slightly in composition form (NFC vs NFD).

    Args:
        text: Input Vietnamese string

    Returns:
        Normalized string
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _strip_diacritics(text: str) -> str:
    """Remove all diacritic marks from text (for last-resort matching).

    Converts e.g. "hà nội" -> "ha noi", "đà nẵng" -> "da nang".
    Used only as a last-resort fallback when NFC matching fails.

    Args:
        text: Input string possibly containing diacritics

    Returns:
        ASCII-approximated string
    """
    # Decompose to NFD so diacritics become separate characters
    nfd = unicodedata.normalize("NFD", text)
    # Keep only ASCII characters and the special Vietnamese đ/Đ
    result = []
    i = 0
    while i < len(nfd):
        ch = nfd[i]
        # Replace đ/Đ with d/D before stripping
        if ch in ("đ", "Đ"):
            result.append("d")
            i += 1
            continue
        # Drop combining diacritical marks (category Mn)
        if unicodedata.category(ch) == "Mn":
            i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result).lower()


# ---------------------------------------------------------------------------
# BIO prompt templates
# ---------------------------------------------------------------------------

_BIO_SYSTEM_PROMPT = """You are an expert Vietnamese Natural Language Understanding (NLU) system \
specializing in air travel customer support.

Your task: given a Vietnamese utterance, output a JSON object with:
1. "intent"     — one intent label from the VALID INTENTS list
2. "confidence" — float 0.0-1.0 reflecting your certainty
3. "bio_tags"   — a list of BIO labels, ONE label per whitespace-split word

═══════════════════════════════════════════════
BIO TAGGING RULES  (read carefully)
═══════════════════════════════════════════════
• "O"             = word is not part of any slot
• "B-<slot_type>" = FIRST word of a slot span
• "I-<slot_type>" = continuation word of the SAME slot span
• Every "I-X" tag MUST immediately follow a "B-X" or another "I-X" of the same type
• The bio_tags array length MUST equal the number of words exactly
• If a slot spans multiple words, the FIRST word gets "B-", all others get "I-"

═══════════════════════════════════════════════
VALID INTENTS
═══════════════════════════════════════════════
abbreviation, aircraft, aircraft#flight#flight_no, airfare, airfare#flight,
airline, airline#flight_no, airport, capacity, city, city#flight_time,
distance, flight, flight#flight_no, flight#flight_time, flight_no,
flight_no#flight_time, flight_time, ground_fare, ground_fare#ground_service,
ground_service, meal, quantity, restriction

═══════════════════════════════════════════════
VALID SLOT TYPES
═══════════════════════════════════════════════
Location : airport_code, airport_name, city_name,
           fromloc.airport_code, fromloc.airport_name, fromloc.city_name,
           fromloc.state_code, fromloc.state_name,
           state_code, state_name,
           stoploc.airport_name, stoploc.city_name, stoploc.state_code,
           toloc.airport_code, toloc.airport_name, toloc.city_name,
           toloc.country_name, toloc.state_code, toloc.state_name

Date     : arrive_date.date_relative, arrive_date.day_name,
           arrive_date.day_number, arrive_date.month_name,
           arrive_date.today_relative, day_name, day_number,
           depart_date.date_relative, depart_date.day_name,
           depart_date.day_number, depart_date.month_name,
           depart_date.today_relative, depart_date.year, month_name,
           return_date.date_relative, return_date.day_name,
           return_date.day_number, return_date.month_name,
           return_date.today_relative

Time     : arrive_time.end_time, arrive_time.period_mod,
           arrive_time.period_of_day, arrive_time.start_time,
           arrive_time.time, arrive_time.time_relative,
           depart_time.end_time, depart_time.period_mod,
           depart_time.period_of_day, depart_time.start_time,
           depart_time.time, depart_time.time_relative,
           return_time.period_mod, return_time.period_of_day,
           time, time_relative

Flight   : aircraft_code, airline_code, airline_name, class_type,
           flight_days, flight_mod, flight_number, flight_stop,
           flight_time, round_trip

Other    : connect, cost_relative, days_code, economy, fare_amount,
           fare_basis_code, meal, meal_code, meal_description,
           mod, or, period_of_day, restriction_code, transport_type

═══════════════════════════════════════════════
OUTPUT FORMAT  (JSON only, no markdown fences)
═══════════════════════════════════════════════
{"intent": "<intent>", "confidence": <float>, "bio_tags": ["<tag>", ...]}
"""


def _format_bio_user_prompt(text: str) -> str:
    """Build a user prompt that lists words with indices.

    Listing words with indices helps the LLM align BIO tags
    precisely to word positions.

    Args:
        text: Vietnamese utterance

    Returns:
        Formatted prompt string
    """
    words = text.split()
    n = len(words)
    word_lines = "\n".join(f"  {i:>3}: {w}" for i, w in enumerate(words))
    return (
        f'Utterance: "{text}"\n\n'
        f"Words ({n} total — your bio_tags array MUST have exactly {n} elements):\n"
        f"{word_lines}\n\n"
        "Return ONLY the JSON object. No explanation, no markdown."
    )


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON from LLM response with multiple strategies.

    Strategy order:
    1. Direct json.loads on stripped text
    2. Extract first {...} block with regex
    3. Strip markdown code fences then parse

    Args:
        raw: Raw string from LLM

    Returns:
        Parsed dict or None if all strategies fail
    """
    import json

    text = raw.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    fenced = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first complete {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# BIO sequence validation
# ---------------------------------------------------------------------------

def _validate_and_repair_bio(bio_tags: List[str], n_words: int) -> List[str]:
    """Validate BIO sequence and repair common LLM mistakes.

    Repairs performed:
    - Wrong length: pad with "O" or truncate
    - "I-X" without preceding "B-X" or "I-X" of same type → convert to "B-X"
    - Unknown tag formats → convert to "O"

    Args:
        bio_tags: Raw BIO tags from LLM
        n_words: Expected number of words

    Returns:
        Repaired BIO tag list of length n_words
    """
    # Length fix
    if len(bio_tags) < n_words:
        bio_tags = list(bio_tags) + ["O"] * (n_words - len(bio_tags))
    elif len(bio_tags) > n_words:
        bio_tags = list(bio_tags[:n_words])

    repaired: List[str] = []
    current_type: Optional[str] = None

    for tag in bio_tags:
        if not isinstance(tag, str):
            repaired.append("O")
            current_type = None
            continue

        tag = tag.strip()

        if tag == "O":
            repaired.append("O")
            current_type = None

        elif tag.startswith("B-"):
            slot_type = tag[2:]
            if slot_type in VALID_SLOT_TYPES:
                repaired.append(tag)
                current_type = slot_type
            else:
                # Unknown slot type — treat as O
                repaired.append("O")
                current_type = None

        elif tag.startswith("I-"):
            slot_type = tag[2:]
            if slot_type in VALID_SLOT_TYPES:
                if slot_type == current_type:
                    # Valid continuation
                    repaired.append(tag)
                else:
                    # Wrong type or no preceding B- — promote to B-
                    repaired.append(f"B-{slot_type}")
                    current_type = slot_type
            else:
                repaired.append("O")
                current_type = None

        else:
            # Completely unknown format
            repaired.append("O")
            current_type = None

    return repaired


# ---------------------------------------------------------------------------
# Slot-dict → BIO fallback converter (kept for predict() backward compat)
# ---------------------------------------------------------------------------

def _slots_dict_to_bio(
    text: str,
    slots: Dict[str, str],
) -> List[str]:
    """Convert a slot dictionary to BIO labels aligned with text words.

    This is the FALLBACK path used when the LLM returns the old
    dict-format response instead of bio_tags.  It uses three
    matching strategies in order:

    1. NFC-normalized exact substring match (handles accent composition)
    2. Diacritic-stripped fuzzy match (handles LLM dropping accents)
    3. Word-by-word match with both normalized and stripped comparison

    Args:
        text: Original Vietnamese utterance
        slots: Dict mapping slot_type -> slot_value

    Returns:
        List of BIO tags aligned to text.split()
    """
    words = text.split()
    n = len(words)
    bio_labels = ["O"] * n

    # Pre-compute normalized versions of words once
    words_nfc = [_normalize_vietnamese(w) for w in words]
    words_stripped = [_strip_diacritics(w) for w in words]

    def _assign(start: int, length: int, slot_type: str) -> bool:
        """Assign BIO labels if the span is unoccupied. Returns True on success."""
        if start < 0 or start + length > n:
            return False
        if bio_labels[start] != "O":
            return False  # already occupied — skip
        bio_labels[start] = f"B-{slot_type}"
        for k in range(1, length):
            bio_labels[start + k] = f"I-{slot_type}"
        return True

    def _find_word_index_from_char(text_variant: str, slot_variant: str) -> int:
        """Return word-start index for a character-level match, or -1.

        Ensures the match starts on a word boundary.
        """
        idx = text_variant.find(slot_variant)
        if idx == -1:
            return -1
        # Check left word boundary
        if idx > 0 and text_variant[idx - 1] != " ":
            return -1
        # Check right word boundary
        end = idx + len(slot_variant)
        if end < len(text_variant) and text_variant[end] != " ":
            return -1
        # Count words before match
        prefix = text_variant[:idx]
        return len(prefix.split()) if prefix else 0

    for slot_type, slot_value in slots.items():
        if slot_type not in VALID_SLOT_TYPES:
            continue
        if not slot_value or not slot_value.strip():
            continue

        slot_value = slot_value.strip()
        slot_words_nfc = _normalize_vietnamese(slot_value).split()
        slot_words_stripped = _strip_diacritics(slot_value).split()
        n_slot = len(slot_words_nfc)

        if n_slot == 0:
            continue

        # ── Strategy 1: NFC normalized substring search ──
        text_nfc = " ".join(words_nfc)
        slot_nfc = " ".join(slot_words_nfc)
        wi = _find_word_index_from_char(text_nfc, slot_nfc)
        if wi != -1:
            _assign(wi, n_slot, slot_type)
            continue

        # ── Strategy 2: diacritic-stripped substring search ──
        text_stripped = " ".join(words_stripped)
        slot_stripped = " ".join(slot_words_stripped)
        wi = _find_word_index_from_char(text_stripped, slot_stripped)
        if wi != -1:
            _assign(wi, n_slot, slot_type)
            continue

        # ── Strategy 3: word-by-word sliding window ──
        for i in range(n - n_slot + 1):
            # Try NFC match first
            if all(words_nfc[i + j] == slot_words_nfc[j] for j in range(n_slot)):
                if _assign(i, n_slot, slot_type):
                    break
            # Try stripped match
            elif all(
                words_stripped[i + j] == slot_words_stripped[j]
                for j in range(n_slot)
            ):
                if _assign(i, n_slot, slot_type):
                    break

    return bio_labels


# ---------------------------------------------------------------------------
# Main classifier class
# ---------------------------------------------------------------------------

class LLMNLUClassifier:
    """Zero-shot NLU classifier using LLMs.

    Extracts intent and slot values from Vietnamese utterances
    using large language models with direct BIO-tag prompting.

    The classifier supports three LLM backends transparently:
    - Anthropic Claude  (provider="anthropic")
    - OpenAI GPT        (provider="openai")
    - Google Gemini     (provider="gemini")
    - Mock client       (provider="mock")   ← for testing

    Usage:
        >>> classifier = LLMNLUClassifier(provider="anthropic")
        >>> result = classifier.predict("tôi muốn bay từ hà nội đến đà nẵng")
        >>> print(result)
        {"intent": "flight", "confidence": 0.95, "slots": {"fromloc.city_name": "hà nội", ...}}

        >>> intent, conf, bio = classifier.predict_with_bio_slots(
        ...     "tôi muốn bay từ hà nội đến đà nẵng"
        ... )
        >>> print(bio)
        ['O', 'O', 'O', 'O', 'B-fromloc.city_name', 'I-fromloc.city_name',
         'O', 'B-toloc.city_name', 'I-toloc.city_name']
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
            provider: Provider name if client not given
                      ("anthropic", "openai", "gemini", "mock")
            model: Model name override
            include_examples: Whether to include few-shot examples in the
                              dict-format prompt (used by predict())
            include_definitions: Whether to include detailed definitions
                                 in the dict-format system prompt
            rate_limit_delay: Delay in seconds between API calls
            **kwargs: Additional arguments forwarded to the LLM client
                      (e.g. max_tokens, temperature, api_key)
        """
        self.client = client or get_client(provider, model=model, **kwargs)
        self.include_examples = include_examples
        self.include_definitions = include_definitions
        self.rate_limit_delay = rate_limit_delay

        # Cache system prompts
        # Dict-format prompt (used by predict() for backward compatibility)
        self._dict_system_prompt = get_system_prompt(include_definitions)
        # BIO-format prompt (used by predict_with_bio_slots())
        self._bio_system_prompt = _BIO_SYSTEM_PROMPT

        # Statistics
        self.total_calls = 0
        self.total_time = 0.0
        self.errors = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent and slots for a single utterance.

        Uses the original dict-format prompt for backward compatibility.
        The returned 'slots' dict maps slot_type -> extracted text span.

        Args:
            text: Vietnamese utterance

        Returns:
            Dictionary with keys:
                intent     (str)  : predicted intent label
                confidence (float): confidence score 0–1
                slots      (dict) : {slot_type: slot_value, ...}
        """
        start_time = time.time()
        self.total_calls += 1

        try:
            user_message = format_user_prompt(
                text, include_examples=self.include_examples
            )

            result = self.client.complete(
                system_prompt=self._dict_system_prompt,
                user_message=user_message,
            )

            self.total_time += time.time() - start_time
            return result

        except Exception as e:
            self.errors += 1
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
            show_progress: Whether to show tqdm progress bar

        Returns:
            List of prediction dicts (same format as predict())
        """
        results = []
        iterator = self._make_iterator(texts, "LLM NLU", show_progress)

        for i, text in enumerate(iterator):
            results.append(self.predict(text))
            if i < len(texts) - 1 and self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

        return results

    def predict_with_bio_slots(self, text: str) -> Tuple[str, float, List[str]]:
        """Predict with BIO-format slot labels for evaluation.

        This method sends a DIFFERENT prompt than predict(): it asks the
        LLM to return one BIO tag per word directly, avoiding any
        post-hoc character-level alignment.

        Fallback chain when LLM response cannot be used as-is:
        1. Parse LLM JSON → extract bio_tags array → validate/repair
        2. If bio_tags missing → extract slots dict → _slots_dict_to_bio()
           (which uses NFC + diacritic-stripped matching)
        3. If JSON parse fails entirely → all "O" tags

        Args:
            text: Vietnamese utterance

        Returns:
            Tuple of:
                intent    (str)       : predicted intent label
                confidence(float)     : confidence score 0–1
                bio_labels(List[str]) : BIO tags aligned to text.split()
        """
        words = text.split()
        n_words = len(words)
        start_time = time.time()
        self.total_calls += 1

        # Default fallback values
        intent = "flight"
        confidence = 0.0
        bio_labels = ["O"] * n_words

        try:
            user_message = _format_bio_user_prompt(text)

            # Call LLM — the client.complete() may return a parsed dict
            # (if the underlying client already parses JSON) or a raw string.
            raw_result = self.client.complete(
                system_prompt=self._bio_system_prompt,
                user_message=user_message,
            )

            self.total_time += time.time() - start_time

            # ── Case A: client already parsed the JSON into a dict ──
            if isinstance(raw_result, dict):
                parsed = raw_result

            # ── Case B: client returned a raw string → parse ourselves ──
            elif isinstance(raw_result, str):
                parsed = _try_parse_json(raw_result)
                if parsed is None:
                    # Unparseable — use all-O fallback
                    self.errors += 1
                    return intent, confidence, bio_labels

            else:
                # Unexpected type
                self.errors += 1
                return intent, confidence, bio_labels

            # Extract intent and confidence
            intent = self._extract_intent(parsed)
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            # ── Prefer bio_tags if present ──
            if "bio_tags" in parsed and isinstance(parsed["bio_tags"], list):
                bio_labels = _validate_and_repair_bio(parsed["bio_tags"], n_words)

            # ── Fallback: bio_tags missing → use slots dict ──
            elif "slots" in parsed and isinstance(parsed["slots"], dict):
                bio_labels = _slots_dict_to_bio(text, parsed["slots"])

            # ── Nothing usable → keep all-O ──
            # bio_labels already = ["O"] * n_words

        except Exception as e:
            self.errors += 1
            # Keep defaults

        return intent, confidence, bio_labels

    def predict_batch_with_bio(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[Tuple[str, float, List[str]]]:
        """Predict batch with BIO-format slot labels.

        Args:
            texts: List of Vietnamese utterances
            show_progress: Whether to show tqdm progress bar

        Returns:
            List of (intent, confidence, bio_labels) tuples
        """
        results = []
        iterator = self._make_iterator(texts, "LLM NLU (BIO)", show_progress)

        for i, text in enumerate(iterator):
            results.append(self.predict_with_bio_slots(text))
            if i < len(texts) - 1 and self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with:
                total_calls  (int)  : number of LLM calls made
                total_time   (float): cumulative seconds spent on API calls
                average_time (float): mean seconds per call
                errors       (int)  : number of failed calls
                error_rate   (float): fraction of calls that failed
        """
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0.0
        error_rate = self.errors / self.total_calls if self.total_calls > 0 else 0.0

        return {
            "total_calls": self.total_calls,
            "total_time": round(self.total_time, 2),
            "average_time": round(avg_time, 3),
            "errors": self.errors,
            "error_rate": round(error_rate, 3),
        }

    def reset_statistics(self) -> None:
        """Reset all usage statistics to zero."""
        self.total_calls = 0
        self.total_time = 0.0
        self.errors = 0

    @property
    def intent_labels(self) -> List[str]:
        """List of valid intent labels."""
        return VALID_INTENTS

    @property
    def slot_types(self) -> List[str]:
        """List of valid slot types."""
        return VALID_SLOT_TYPES

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _extract_intent(self, parsed: Dict[str, Any]) -> str:
        """Extract and validate intent from parsed LLM response.

        Falls back to "flight" (most common class) if intent is
        missing or not in the valid intent list.

        Args:
            parsed: Parsed JSON dict from LLM

        Returns:
            Valid intent string
        """
        intent = parsed.get("intent", "flight")
        if not isinstance(intent, str):
            return "flight"
        intent = intent.strip()
        # Accept if valid; otherwise return most-common fallback
        return intent if intent in VALID_INTENTS else "flight"

    @staticmethod
    def _make_iterator(texts: List[str], desc: str, show_progress: bool):
        """Wrap texts in a tqdm progress bar if requested and available."""
        if show_progress:
            try:
                from tqdm import tqdm
                return tqdm(texts, desc=desc, unit="sample")
            except ImportError:
                pass
        return texts


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_llm_classifier(
    provider: str = "mock",
    config_path: Optional[str] = None,
) -> LLMNLUClassifier:
    """Create an LLM classifier from a config file or sensible defaults.

    Supported providers: "anthropic", "openai", "gemini", "mock"

    The optional YAML config file may contain:

        models:
          anthropic:
            model: claude-3-5-sonnet-20241022
            max_tokens: 512
            temperature: 0.0
          openai:
            model: gpt-4o-mini
            max_tokens: 512
            temperature: 0.0
          gemini:
            model: gemini-1.5-flash
            max_tokens: 512
            temperature: 0.0
        evaluation:
          delay_seconds: 0.5

    Args:
        provider: LLM provider name
        config_path: Path to YAML config file (optional)

    Returns:
        Configured LLMNLUClassifier instance
    """
    config: Dict[str, Any] = {}

    if config_path:
        try:
            import yaml
            from pathlib import Path

            config = yaml.safe_load(Path(config_path).read_text()) or {}
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