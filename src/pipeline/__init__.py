"""Dialogue Pipeline module.

Provides the full dialogue pipeline orchestrator that integrates
STT, NLU, DST, Policy, NLG, and TTS components.

Exports:
    DialoguePipeline     - Main pipeline orchestrator
    PipelineConfig       - Configuration for pipeline
    TurnResult           - Result of processing a dialogue turn
    ConversationManager  - Multi-session conversation manager
    create_pipeline      - Factory function to create pipeline
"""

from .orchestrator import (
    DialoguePipeline,
    PipelineConfig,
    TurnResult,
    ConversationManager,
    create_pipeline,
)

__all__ = [
    "DialoguePipeline",
    "PipelineConfig",
    "TurnResult",
    "ConversationManager",
    "create_pipeline",
]
