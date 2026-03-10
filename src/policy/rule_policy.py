"""Rule-based Policy for Vietnamese AI Call Center.

Determines system actions based on dialogue state and NLU output,
using a decision tree approach for flight booking scenarios.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..dst.tracker import DialogueState, REQUIRED_SLOTS, DEFAULT_REQUIRED_SLOTS


class ActionType(str, Enum):
    """Types of system actions."""

    GREET = "greet"
    CLARIFY = "clarify"
    REQUEST_SLOT = "request_slot"
    CONFIRM = "confirm"
    EXECUTE = "execute"
    RESPOND = "respond"
    ESCALATE = "escalate"
    GOODBYE = "goodbye"


# Intents that typically need flight booking flow
BOOKING_INTENTS = {"flight", "airfare", "airfare#flight", "flight#airfare"}

# Intents that are informational queries
INFO_INTENTS = {
    "flight_time",
    "airline",
    "airport",
    "aircraft",
    "meal",
    "capacity",
    "distance",
    "city",
    "abbreviation",
    "quantity",
    "restriction",
}

# Intents for ground transportation
GROUND_INTENTS = {"ground_service", "ground_fare", "ground_fare#ground_service"}

# Slot request priority order
SLOT_REQUEST_ORDER = [
    "fromloc.city_name",
    "toloc.city_name",
    "depart_date.day_name",
    "depart_date.month_name",
    "depart_date.today_relative",
    "depart_time.period_of_day",
    "airline_name",
    "class_type",
]


@dataclass
class PolicyConfig:
    """Configuration for rule-based policy.

    Attributes:
        confidence_threshold: Minimum confidence to accept intent
        low_confidence_threshold: Below this, always clarify
        max_clarify_attempts: Maximum consecutive clarifications
        require_confirmation: Whether to confirm before execution
    """

    confidence_threshold: float = 0.5
    low_confidence_threshold: float = 0.3
    max_clarify_attempts: int = 3
    require_confirmation: bool = True


class RuleBasedPolicy:
    """Rule-based policy for dialogue action selection.

    Implements a decision tree for determining the next system action
    based on the current dialogue state and NLU output.

    Decision flow:
    1. Check if greeting needed (first turn)
    2. Check confidence level (clarify if too low)
    3. Check for required slots (request if missing)
    4. Confirm parameters (if configured)
    5. Execute action or provide information

    Usage:
        policy = RuleBasedPolicy()
        action = policy.select_action(state, nlu_output)
        # action = {"type": "request_slot", "slot": "fromloc.city_name", ...}
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        """Initialize policy.

        Args:
            config: Policy configuration. Uses defaults if not provided.
        """
        self.config = config or PolicyConfig()
        self._clarify_count = 0

    def reset(self) -> None:
        """Reset policy state for new conversation."""
        self._clarify_count = 0

    def select_action(
        self,
        state: DialogueState,
        nlu_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Select the next system action based on current state.

        Args:
            state: Current dialogue state.
            nlu_output: NLU prediction for current turn.

        Returns:
            Action dictionary containing:
                - type: ActionType value
                - Additional fields depending on action type
        """
        intent = nlu_output.get("intent", "unknown")
        confidence = nlu_output.get("confidence", 0.0)
        slots = nlu_output.get("slots", {})

        # First turn greeting
        if state.turn_count == 1 and not state.slots:
            # If user already provided info, skip greeting
            if not slots and confidence < self.config.confidence_threshold:
                return self._create_action(ActionType.GREET)

        # Check if user confirmed after a confirm action
        if state.confirmed and state.last_action:
            last_action_type = state.last_action.get("type")
            if last_action_type == "confirm":
                # User confirmed - execute the action
                dominant_intent = state.dominant_intent or "flight"
                return self._create_action(
                    ActionType.EXECUTE,
                    intent=dominant_intent,
                    params=state.slots,
                )

        # Low confidence - clarify
        if confidence < self.config.low_confidence_threshold:
            self._clarify_count += 1
            if self._clarify_count >= self.config.max_clarify_attempts:
                return self._create_action(ActionType.ESCALATE, reason="max_clarify_exceeded")
            return self._create_action(ActionType.CLARIFY, reason="low_confidence")

        # Reset clarify count on successful understanding
        self._clarify_count = 0

        # Handle different intent categories
        if intent in BOOKING_INTENTS:
            return self._handle_booking_intent(state, intent, confidence, slots)
        elif intent in INFO_INTENTS:
            return self._handle_info_intent(state, intent, confidence, slots)
        elif intent in GROUND_INTENTS:
            return self._handle_ground_intent(state, intent, confidence, slots)
        else:
            # Unknown or composite intent
            return self._handle_general_intent(state, intent, confidence, slots)

    def _handle_booking_intent(
        self,
        state: DialogueState,
        intent: str,
        confidence: float,
        slots: Dict[str, str],
    ) -> Dict[str, Any]:
        """Handle flight booking related intents.

        Args:
            state: Current dialogue state.
            intent: Detected intent.
            confidence: Intent confidence.
            slots: Detected slots.

        Returns:
            Selected action.
        """
        # Check for missing required slots
        missing_slots = state.get_missing_required_slots(intent)

        if missing_slots:
            # Request the first missing slot in priority order
            for slot in SLOT_REQUEST_ORDER:
                if slot in missing_slots:
                    return self._create_action(
                        ActionType.REQUEST_SLOT,
                        slot=slot,
                        intent=intent,
                    )
            # Request first missing slot if not in priority list
            return self._create_action(
                ActionType.REQUEST_SLOT,
                slot=missing_slots[0],
                intent=intent,
            )

        # All required slots filled - confirm or execute
        if self.config.require_confirmation and not state.confirmed:
            return self._create_action(
                ActionType.CONFIRM,
                intent=intent,
                params=state.slots,
            )

        # Execute the booking
        return self._create_action(
            ActionType.EXECUTE,
            intent=intent,
            params=state.slots,
        )

    def _handle_info_intent(
        self,
        state: DialogueState,
        intent: str,
        confidence: float,
        slots: Dict[str, str],
    ) -> Dict[str, Any]:
        """Handle informational query intents.

        Args:
            state: Current dialogue state.
            intent: Detected intent.
            confidence: Intent confidence.
            slots: Detected slots.

        Returns:
            Selected action.
        """
        # For info queries, provide response directly
        return self._create_action(
            ActionType.RESPOND,
            intent=intent,
            params=state.slots,
            query_type=intent,
        )

    def _handle_ground_intent(
        self,
        state: DialogueState,
        intent: str,
        confidence: float,
        slots: Dict[str, str],
    ) -> Dict[str, Any]:
        """Handle ground transportation intents.

        Args:
            state: Current dialogue state.
            intent: Detected intent.
            confidence: Intent confidence.
            slots: Detected slots.

        Returns:
            Selected action.
        """
        # Check for location
        if not state.has_slot("fromloc.city_name") and not state.has_slot("toloc.city_name"):
            return self._create_action(
                ActionType.REQUEST_SLOT,
                slot="fromloc.city_name",
                intent=intent,
            )

        return self._create_action(
            ActionType.RESPOND,
            intent=intent,
            params=state.slots,
            query_type="ground_transport",
        )

    def _handle_general_intent(
        self,
        state: DialogueState,
        intent: str,
        confidence: float,
        slots: Dict[str, str],
    ) -> Dict[str, Any]:
        """Handle general or unknown intents.

        Args:
            state: Current dialogue state.
            intent: Detected intent.
            confidence: Intent confidence.
            slots: Detected slots.

        Returns:
            Selected action.
        """
        # If some slots are filled, try to help with booking
        if state.slots:
            missing_slots = state.get_missing_required_slots("flight")
            if missing_slots:
                return self._create_action(
                    ActionType.REQUEST_SLOT,
                    slot=missing_slots[0],
                    intent="flight",
                )

        # Clarify user intent
        if confidence < self.config.confidence_threshold:
            return self._create_action(ActionType.CLARIFY, reason="unclear_intent")

        # Provide general response
        return self._create_action(
            ActionType.RESPOND,
            intent=intent,
            params=state.slots,
        )

    def _create_action(self, action_type: ActionType, **kwargs) -> Dict[str, Any]:
        """Create an action dictionary.

        Args:
            action_type: Type of action.
            **kwargs: Additional action parameters.

        Returns:
            Action dictionary.
        """
        action = {"type": action_type.value, **kwargs}
        return action

    def get_required_slots(self, intent: str) -> List[str]:
        """Get required slots for an intent.

        Args:
            intent: Intent label.

        Returns:
            List of required slot types.
        """
        return REQUIRED_SLOTS.get(intent, DEFAULT_REQUIRED_SLOTS)


def create_policy(config: Optional[Dict[str, Any]] = None) -> RuleBasedPolicy:
    """Factory function to create a policy instance.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured RuleBasedPolicy instance.
    """
    if config:
        policy_config = PolicyConfig(
            confidence_threshold=config.get("confidence_threshold", 0.5),
            low_confidence_threshold=config.get("low_confidence_threshold", 0.3),
            max_clarify_attempts=config.get("max_clarify_attempts", 3),
            require_confirmation=config.get("require_confirmation", True),
        )
        return RuleBasedPolicy(config=policy_config)
    return RuleBasedPolicy()
