"""API clients for LLM-based NLU.

Provides unified interface for different LLM providers:
  - AnthropicClient: Uses Claude with tool_use for structured output
  - OpenAIClient: Uses GPT with JSON mode
  - MockClient: For testing without API keys

All clients implement the same interface: complete(system_prompt, user_message) -> dict
"""

import json
import os
import time
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_message: str,
        tool_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send completion request and return structured NLU output.

        Args:
            system_prompt: System instructions
            user_message: User input with few-shot examples
            tool_schema: Optional schema for structured output

        Returns:
            Dictionary with intent, confidence, and slots
        """
        pass

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common issues.

        Args:
            text: Raw response text

        Returns:
            Parsed dictionary

        Raises:
            ValueError: If JSON parsing fails
        """
        # Clean up common issues
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nResponse: {text[:200]}")

    def _validate_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize NLU output.

        Args:
            output: Raw parsed output

        Returns:
            Validated and normalized output
        """
        from .llm_prompts import VALID_INTENTS

        # Ensure required fields
        intent = output.get("intent", "flight")
        confidence = output.get("confidence", 0.5)
        slots = output.get("slots", {})

        # Validate intent
        if intent not in VALID_INTENTS:
            # Try to find closest match
            intent_lower = intent.lower()
            for valid in VALID_INTENTS:
                if valid.lower() == intent_lower:
                    intent = valid
                    break
            else:
                # Default to most common
                intent = "flight"
                confidence = min(confidence, 0.5)

        # Validate confidence
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        # Ensure slots is a dict with string values
        if not isinstance(slots, dict):
            slots = {}
        slots = {str(k): str(v) for k, v in slots.items() if v}

        return {
            "intent": intent,
            "confidence": confidence,
            "slots": slots,
        }


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client using tool_use for structured output."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 512,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, max_tokens, temperature, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )

        # Lazy import
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        tool_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send completion using Claude with tool_use."""
        from .llm_prompts import get_anthropic_tool_schema

        tool = tool_schema or get_anthropic_tool_schema()

        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool["name"]},
                    messages=[{"role": "user", "content": user_message}],
                )

                # Extract tool use result
                for block in response.content:
                    if block.type == "tool_use":
                        return self._validate_output(block.input)

                # Fallback: try to parse text response
                for block in response.content:
                    if hasattr(block, "text"):
                        parsed = self._parse_json_response(block.text)
                        return self._validate_output(parsed)

                raise ValueError("No valid response from Claude")

            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Anthropic API failed after {self.retry_attempts} attempts: {last_error}")


class OpenAIClient(BaseLLMClient):
    """OpenAI client using JSON mode for structured output."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, max_tokens, temperature, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        tool_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send completion using OpenAI with JSON mode."""
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )

                text = response.choices[0].message.content
                parsed = self._parse_json_response(text)
                return self._validate_output(parsed)

            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"OpenAI API failed after {self.retry_attempts} attempts: {last_error}")


class GeminiClient(BaseLLMClient):
    """Google Gemini client using JSON mode for structured output."""

    def __init__(
        self,
        model: str = "gemini-3.1-pro-latest",
        max_tokens: int = 512,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, max_tokens, temperature, **kwargs)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable."
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.generation_config = genai.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                response_mime_type="application/json",
            )
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        tool_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send completion using Gemini with JSON mode."""
        last_error = None

        # Gemini combines system + user prompt as a single turn
        full_prompt = f"{system_prompt}\n\n{user_message}"

        for attempt in range(self.retry_attempts):
            try:
                model = self.genai.GenerativeModel(
                    model_name=self.model,
                    generation_config=self.generation_config,
                )
                response = model.generate_content(full_prompt)
                parsed = self._parse_json_response(response.text)
                return self._validate_output(parsed)

            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Gemini API failed after {self.retry_attempts} attempts: {last_error}")


class MockClient(BaseLLMClient):
    """Mock client for testing without API keys.

    Returns reasonable mock results based on simple keyword matching.
    Useful for development, testing, and demos without API costs.
    """

    # Keyword patterns for intent detection
    INTENT_KEYWORDS = {
        "flight": ["chuyến bay", "bay", "flight", "vé", "khởi hành", "đặt vé", "tìm"],
        "airfare": ["giá", "bao nhiêu", "tiền", "phí", "fare", "price", "cost"],
        "airline": ["hãng", "airline", "vietnam airlines", "vietjet", "bamboo", "jetstar"],
        "flight_time": ["mấy giờ", "bao lâu", "thời gian", "duration", "time"],
        "airport": ["sân bay", "airport", "terminal"],
        "ground_service": ["xe đưa đón", "shuttle", "ground", "taxi"],
        "meal": ["bữa ăn", "meal", "ăn"],
        "aircraft": ["máy bay", "aircraft", "boeing", "airbus"],
    }

    # Slot extraction patterns
    CITY_KEYWORDS = [
        "hà nội", "hồ chí minh", "đà nẵng", "đà lạt", "phú quốc", "nha trang",
        "cần thơ", "huế", "hải phòng", "quy nhơn", "pleiku", "buôn ma thuột",
        "vinh", "thanh hóa", "sài gòn", "cam ranh", "côn đảo", "hạ long",
        "tuy hòa", "điện biên phủ", "cà mau", "sevilla", "manila", "bắc kinh"
    ]

    DAY_KEYWORDS = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]

    TIME_KEYWORDS = ["sáng", "trưa", "chiều", "tối", "đêm"]

    AIRLINE_KEYWORDS = ["vietnam airlines", "vietjet", "bamboo airways", "jetstar", "pacific airlines"]

    def __init__(
        self,
        model: str = "mock",
        simulate_latency: bool = True,
        latency_range: tuple = (0.1, 0.3),
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.simulate_latency = simulate_latency
        self.latency_range = latency_range

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        tool_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return mock NLU result based on keyword matching."""
        if self.simulate_latency:
            time.sleep(random.uniform(*self.latency_range))

        # Extract the actual utterance from the user message
        utterance = user_message.lower()
        if 'input: "' in utterance:
            start = utterance.find('input: "') + 8
            end = utterance.find('"', start)
            if end > start:
                utterance = utterance[start:end]

        # Detect intent
        intent = self._detect_intent(utterance)

        # Check for composite intents
        has_price = any(kw in utterance for kw in ["giá", "bao nhiêu", "tiền"])
        has_flight = any(kw in utterance for kw in ["chuyến bay", "bay từ", "bay đến"])

        if has_price and has_flight:
            intent = "airfare#flight"
        elif intent == "airfare" and has_flight:
            intent = "airfare#flight"

        # Extract slots
        slots = self._extract_slots(utterance)

        # Calculate mock confidence
        confidence = self._calculate_confidence(utterance, intent, slots)

        return {
            "intent": intent,
            "confidence": confidence,
            "slots": slots,
        }

    def _detect_intent(self, text: str) -> str:
        """Detect intent using keyword matching."""
        scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[intent] = score

        if not scores:
            return "flight"  # Default

        return max(scores, key=scores.get)

    def _extract_slots(self, text: str) -> Dict[str, str]:
        """Extract slots using keyword matching."""
        slots = {}

        # Extract cities
        found_cities = [city for city in self.CITY_KEYWORDS if city in text]

        # Try to determine from/to based on context
        if found_cities:
            # Look for "từ X" pattern for fromloc
            for city in found_cities:
                idx = text.find(city)
                context_start = max(0, idx - 10)
                context = text[context_start:idx]

                if "từ" in context:
                    slots["fromloc.city_name"] = city
                elif "đến" in context or "tới" in context or "đi" in context:
                    slots["toloc.city_name"] = city

            # If only found cities without clear context, assign first as from, second as to
            if not slots and len(found_cities) >= 2:
                slots["fromloc.city_name"] = found_cities[0]
                slots["toloc.city_name"] = found_cities[1]
            elif not slots and len(found_cities) == 1:
                # Single city - check if "đến" or "đi" precedes it
                city = found_cities[0]
                idx = text.find(city)
                if idx > 0:
                    before = text[:idx]
                    if "đến" in before or "đi" in before or "tới" in before:
                        slots["toloc.city_name"] = city
                    elif "từ" in before:
                        slots["fromloc.city_name"] = city
                    else:
                        # Default to destination
                        slots["toloc.city_name"] = city

        # Extract day names
        for day in self.DAY_KEYWORDS:
            if day in text:
                slots["depart_date.day_name"] = day
                break

        # Extract time of day
        for time_period in self.TIME_KEYWORDS:
            if time_period in text:
                slots["depart_time.period_of_day"] = time_period
                break

        # Extract airlines
        for airline in self.AIRLINE_KEYWORDS:
            if airline in text:
                slots["airline_name"] = airline
                break

        # Check for round trip
        if "khứ hồi" in text or "hai chiều" in text:
            slots["round_trip"] = "khứ hồi"
        elif "một chiều" in text:
            slots["round_trip"] = "một chiều"

        # Extract relative dates
        if "hôm nay" in text:
            slots["depart_date.today_relative"] = "hôm nay"
        elif "ngày mai" in text:
            slots["depart_date.today_relative"] = "ngày mai"
        elif "tuần tới" in text or "tới" in text:
            slots["depart_date.date_relative"] = "tới"

        # Extract month
        months = [
            "tháng một", "tháng hai", "tháng ba", "tháng tư", "tháng năm",
            "tháng sáu", "tháng bảy", "tháng tám", "tháng chín", "tháng mười",
            "tháng mười một", "tháng mười hai"
        ]
        for i, month in enumerate(months, 1):
            if month in text:
                slots["depart_date.month_name"] = month
                break
            # Also check "tháng X" format
            short_form = f"tháng {i}"
            if short_form in text:
                slots["depart_date.month_name"] = short_form
                break

        return slots

    def _calculate_confidence(
        self, text: str, intent: str, slots: Dict[str, str]
    ) -> float:
        """Calculate mock confidence based on match quality."""
        base_confidence = 0.75

        # Boost for more slots
        slot_boost = min(0.15, len(slots) * 0.03)

        # Boost for keyword matches
        keyword_count = 0
        if intent in self.INTENT_KEYWORDS:
            keyword_count = sum(
                1 for kw in self.INTENT_KEYWORDS[intent] if kw in text
            )
        keyword_boost = min(0.10, keyword_count * 0.03)

        confidence = base_confidence + slot_boost + keyword_boost

        # Add small random variation
        confidence += random.uniform(-0.05, 0.05)

        return round(min(0.98, max(0.50, confidence)), 2)


def get_client(
    provider: str = "mock",
    model: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Factory function to get appropriate LLM client.

    Args:
        provider: One of "anthropic", "openai", "gemini", or "mock"
        model: Optional model name override
        **kwargs: Additional arguments for client

    Returns:
        Configured LLM client instance
    """
    provider = provider.lower()

    if provider == "anthropic":
        model = model or "claude-3-haiku-20240307"
        try:
            return AnthropicClient(model=model, **kwargs)
        except (ValueError, ImportError) as e:
            print(f"Warning: Could not initialize Anthropic client: {e}")
            print("Falling back to mock client.")
            return MockClient(**kwargs)

    elif provider == "openai":
        model = model or "gpt-4o-mini"
        try:
            return OpenAIClient(model=model, **kwargs)
        except (ValueError, ImportError) as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")
            print("Falling back to mock client.")
            return MockClient(**kwargs)

    elif provider == "gemini":
        model = model or "gemini-3.1-pro-latest"
        try:
            return GeminiClient(model=model, **kwargs)
        except (ValueError, ImportError) as e:
            print(f"Warning: Could not initialize Gemini client: {e}")
            print("Falling back to mock client.")
            return MockClient(**kwargs)

    elif provider == "mock":
        return MockClient(**kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic', 'openai', 'gemini', or 'mock'.")