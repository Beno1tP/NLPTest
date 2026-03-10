"""Dialogue State Tracker for Vietnamese AI Call Center.

Tracks conversation state across multiple turns, accumulating slot values
and maintaining intent history for context-aware dialogue management.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from copy import deepcopy


# Slot types that are relevant for flight booking domain
BOOKING_SLOT_TYPES = {
    # Location slots
    "fromloc.city_name",
    "fromloc.airport_name",
    "fromloc.airport_code",
    "toloc.city_name",
    "toloc.airport_name",
    "toloc.airport_code",
    "stoploc.city_name",
    # Date slots
    "depart_date.day_name",
    "depart_date.day_number",
    "depart_date.month_name",
    "depart_date.date_relative",
    "depart_date.today_relative",
    "depart_date.year",
    "arrive_date.day_name",
    "arrive_date.day_number",
    "arrive_date.month_name",
    "return_date.day_name",
    "return_date.day_number",
    "return_date.month_name",
    # Time slots
    "depart_time.time",
    "depart_time.start_time",
    "depart_time.end_time",
    "depart_time.period_of_day",
    "arrive_time.time",
    "arrive_time.period_of_day",
    # Flight attributes
    "airline_name",
    "airline_code",
    "flight_number",
    "class_type",
    "round_trip",
    "flight_mod",  # nonstop, direct
}

# Required slots for common intents
REQUIRED_SLOTS = {
    "flight": ["fromloc.city_name", "toloc.city_name"],
    "airfare": ["fromloc.city_name", "toloc.city_name"],
    "airfare#flight": ["fromloc.city_name", "toloc.city_name"],
    "flight#airfare": ["fromloc.city_name", "toloc.city_name"],
    "flight_time": [],  # Usually needs flight_number or route
    "airline": [],
    "airport": [],
    "ground_service": ["fromloc.city_name"],
}

# Default required slots for unknown intents
DEFAULT_REQUIRED_SLOTS = ["fromloc.city_name", "toloc.city_name"]


@dataclass
class DialogueState:
    """Represents the current state of the dialogue.

    Attributes:
        slots: Accumulated slot values across turns
        intent_history: List of (intent, confidence) tuples from each turn
        turn_count: Number of dialogue turns
        completed: Whether the dialogue goal is achieved
        confirmed: Whether the user has confirmed the action
        last_action: The last system action taken
        context: Additional context information
    """

    slots: Dict[str, str] = field(default_factory=dict)
    intent_history: List[tuple] = field(default_factory=list)
    turn_count: int = 0
    completed: bool = False
    confirmed: bool = False
    last_action: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_intent(self) -> Optional[str]:
        """Get the most recent intent."""
        if self.intent_history:
            return self.intent_history[-1][0]
        return None

    @property
    def current_confidence(self) -> float:
        """Get the confidence of the most recent intent."""
        if self.intent_history:
            return self.intent_history[-1][1]
        return 0.0

    @property
    def dominant_intent(self) -> Optional[str]:
        """Get the most frequent intent in the conversation."""
        if not self.intent_history:
            return None
        intent_counts: Dict[str, int] = {}
        for intent, _ in self.intent_history:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        return max(intent_counts.items(), key=lambda x: x[1])[0]

    def get_slot(self, slot_type: str, default: Any = None) -> Any:
        """Get a slot value with optional default."""
        return self.slots.get(slot_type, default)

    def has_slot(self, slot_type: str) -> bool:
        """Check if a slot is filled."""
        return slot_type in self.slots and self.slots[slot_type]

    def get_missing_required_slots(self, intent: Optional[str] = None) -> List[str]:
        """Get list of required slots that are not yet filled.

        Args:
            intent: Intent to check requirements for. Uses dominant_intent if not provided.

        Returns:
            List of missing required slot types.
        """
        intent = intent or self.dominant_intent
        if intent is None:
            return []

        required = REQUIRED_SLOTS.get(intent, DEFAULT_REQUIRED_SLOTS)
        return [slot for slot in required if not self.has_slot(slot)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "slots": deepcopy(self.slots),
            "intent_history": list(self.intent_history),
            "turn_count": self.turn_count,
            "completed": self.completed,
            "confirmed": self.confirmed,
            "last_action": deepcopy(self.last_action) if self.last_action else None,
            "context": deepcopy(self.context),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueState":
        """Create state from dictionary."""
        return cls(
            slots=data.get("slots", {}),
            intent_history=[(i, c) for i, c in data.get("intent_history", [])],
            turn_count=data.get("turn_count", 0),
            completed=data.get("completed", False),
            confirmed=data.get("confirmed", False),
            last_action=data.get("last_action"),
            context=data.get("context", {}),
        )


class StateTracker:
    """Dialogue State Tracker that accumulates information across turns.

    Handles:
    - Slot accumulation across multiple turns
    - User corrections (new value for existing slot)
    - Intent history tracking
    - Confirmation and completion state management

    Usage:
        tracker = StateTracker()

        # Process each turn
        nlu_output = {"intent": "flight", "confidence": 0.95, "slots": {"toloc.city_name": "da nang"}}
        state = tracker.update(nlu_output)

        # Check state
        print(state.slots)  # {"toloc.city_name": "da nang"}
        print(state.get_missing_required_slots())  # ["fromloc.city_name"]

        # Reset for new conversation
        tracker.reset()
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize the state tracker.

        Args:
            confidence_threshold: Minimum confidence to accept NLU output.
        """
        self.confidence_threshold = confidence_threshold
        self._state = DialogueState()

    @property
    def state(self) -> DialogueState:
        """Get current dialogue state."""
        return self._state

    def reset(self) -> DialogueState:
        """Reset the dialogue state for a new conversation.

        Returns:
            Fresh DialogueState.
        """
        self._state = DialogueState()
        return self._state

    def update(self, nlu_output: Dict[str, Any], user_text: Optional[str] = None) -> DialogueState:
        """Update dialogue state with NLU output from current turn.

        Args:
            nlu_output: NLU prediction containing:
                - intent: Predicted intent label
                - confidence: Intent confidence score (0-1)
                - slots: Dictionary of {slot_type: value}
            user_text: Original user utterance (optional, for context)

        Returns:
            Updated DialogueState.
        """
        self._state.turn_count += 1

        # Extract NLU components
        intent = nlu_output.get("intent", "unknown")
        confidence = nlu_output.get("confidence", 0.0)
        slots = nlu_output.get("slots", {})

        # Track intent history
        self._state.intent_history.append((intent, confidence))

        # Update slots (new values override old ones - user correction)
        self._update_slots(slots)

        # Handle special intents
        self._handle_special_intents(intent, user_text)

        # Store user text in context
        if user_text:
            if "user_utterances" not in self._state.context:
                self._state.context["user_utterances"] = []
            self._state.context["user_utterances"].append(user_text)

        return self._state

    def _update_slots(self, new_slots: Dict[str, str]) -> None:
        """Update slots with new values.

        New values override existing ones (handles user corrections).

        Args:
            new_slots: New slot values from NLU output.
        """
        for slot_type, value in new_slots.items():
            if value and slot_type in BOOKING_SLOT_TYPES:
                # Clean the value
                cleaned_value = self._clean_slot_value(value)
                if cleaned_value:
                    self._state.slots[slot_type] = cleaned_value

    def _clean_slot_value(self, value: str) -> str:
        """Clean and normalize slot value.

        Args:
            value: Raw slot value.

        Returns:
            Cleaned slot value.
        """
        if not value:
            return ""
        # Strip whitespace and normalize
        cleaned = value.strip().lower()
        return cleaned

    def _handle_special_intents(self, intent: str, user_text: Optional[str]) -> None:
        """Handle special intents that affect dialogue state.

        Args:
            intent: Current intent.
            user_text: Original user utterance.
        """
        # Handle confirmation
        if user_text:
            user_lower = user_text.lower()

            # Check for positive confirmation
            positive_words = ["dung", "đúng", "ok", "yes", "co", "có", "phai", "phải", "vang", "vâng"]
            if any(word in user_lower for word in positive_words):
                if self._state.last_action and self._state.last_action.get("type") == "confirm":
                    self._state.confirmed = True

            # Check for negative / correction
            negative_words = ["khong", "không", "sai", "chua", "chưa", "no"]
            if any(word in user_lower for word in negative_words):
                self._state.confirmed = False

            # Check for reset / start over
            reset_words = ["lai", "lại", "moi", "mới", "reset", "huy", "hủy"]
            if any(word in user_lower for word in reset_words):
                self._state.slots = {}
                self._state.confirmed = False
                self._state.completed = False

    def clear_slot(self, slot_type: str) -> None:
        """Clear a specific slot value.

        Args:
            slot_type: Slot type to clear.
        """
        if slot_type in self._state.slots:
            del self._state.slots[slot_type]

    def set_slot(self, slot_type: str, value: str) -> None:
        """Manually set a slot value.

        Args:
            slot_type: Slot type.
            value: Slot value.
        """
        cleaned = self._clean_slot_value(value)
        if cleaned:
            self._state.slots[slot_type] = cleaned

    def mark_completed(self) -> None:
        """Mark the dialogue as completed."""
        self._state.completed = True

    def set_last_action(self, action: Dict[str, Any]) -> None:
        """Store the last system action.

        Args:
            action: Action dictionary from policy.
        """
        self._state.last_action = deepcopy(action)

    def get_summary(self) -> str:
        """Get a human-readable summary of current state.

        Returns:
            Summary string.
        """
        lines = [
            f"Turn: {self._state.turn_count}",
            f"Intent: {self._state.current_intent} ({self._state.current_confidence:.2f})",
            f"Slots: {self._state.slots}",
            f"Confirmed: {self._state.confirmed}",
            f"Completed: {self._state.completed}",
        ]
        missing = self._state.get_missing_required_slots()
        if missing:
            lines.append(f"Missing: {missing}")
        return "\n".join(lines)
