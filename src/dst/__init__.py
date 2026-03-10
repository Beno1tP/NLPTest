"""Dialogue State Tracking module.

Provides components for tracking conversation state across multiple turns,
accumulating slot values, and managing dialogue context.

Exports:
    DialogueState   - Dataclass representing current dialogue state
    StateTracker    - Tracks and updates dialogue state based on NLU output
    BOOKING_SLOT_TYPES - Set of valid slot types for flight booking
    REQUIRED_SLOTS  - Dict mapping intents to required slot types
"""

from .tracker import (
    DialogueState,
    StateTracker,
    BOOKING_SLOT_TYPES,
    REQUIRED_SLOTS,
    DEFAULT_REQUIRED_SLOTS,
)

__all__ = [
    "DialogueState",
    "StateTracker",
    "BOOKING_SLOT_TYPES",
    "REQUIRED_SLOTS",
    "DEFAULT_REQUIRED_SLOTS",
]
