"""NLU (Natural Language Understanding) module for Vietnamese AI Call Center.

This module provides three approaches for joint intent classification and slot filling:

1. SVM Baseline (TF-IDF + LinearSVC + CRF)
   - SVMIntentClassifier: TF-IDF vectorization with calibrated SVM
   - CRFSlotFiller: CRF-based BIO sequence labeling
   - SVMNLU: Combined pipeline

2. JointBERT + PhoBERT
   - JointBERTModel: Fine-tuned PhoBERT with joint intent/slot heads
   - JointBERTNLU: High-level inference wrapper
   - JointBERTTrainer: Training loop with early stopping

3. LLM Zero-Shot
   - LLMNLUClassifier: Claude/GPT prompting for intent and slot extraction
   - Supports Anthropic (Claude), OpenAI (GPT), and mock client for testing

Usage:
    # SVM Baseline
    from src.nlu import SVMNLU, SVMIntentClassifier, CRFSlotFiller
    nlu = SVMNLU()
    nlu.fit(texts, intent_ids, slot_labels, id2intent)
    result = nlu.predict("dat ve di da nang")

    # JointBERT (primary model)
    from src.nlu import JointBERTNLU, load_nlu
    nlu = load_nlu("models/best_jointbert.pt")
    result = nlu.predict("toi muon dat ve di da nang")
    # {"intent": "flight", "confidence": 0.95, "slots": {"toloc.city_name": "da nang"}}

    # LLM Zero-Shot
    from src.nlu import LLMNLUClassifier, create_llm_classifier
    llm_nlu = create_llm_classifier(provider="mock")  # or "anthropic", "openai"
    result = llm_nlu.predict("toi muon bay tu ha noi den da nang")
"""

from .crf_slot import CRFSlotFiller
from .svm_intent import SVMIntentClassifier
from .svm_nlu import SVMNLU

# JointBERT + PhoBERT
from .jointbert_model import (
    JointBERTModel,
    JointBERTWithIntentSlotAttention,
    create_model,
)
from .jointbert_data import (
    JointBERTDataset,
    JointBERTDataModule,
    create_data_module,
)
from .jointbert_trainer import (
    JointBERTTrainer,
    evaluate_model,
)
from .jointbert_nlu import (
    JointBERTNLU,
    load_nlu,
)

# LLM Zero-Shot NLU
from .llm_nlu import LLMNLUClassifier, create_llm_classifier
from .llm_client import (
    BaseLLMClient,
    AnthropicClient,
    OpenAIClient,
    MockClient,
    get_client,
)
from .llm_prompts import (
    VALID_INTENTS,
    VALID_SLOT_TYPES,
    INTENT_DEFINITIONS,
    SLOT_TYPE_DEFINITIONS,
    get_system_prompt,
    format_user_prompt,
    get_few_shot_examples,
    get_anthropic_tool_schema,
    get_openai_json_schema,
)

__all__ = [
    # SVM Baseline
    "SVMIntentClassifier",
    "CRFSlotFiller",
    "SVMNLU",
    # JointBERT + PhoBERT
    "JointBERTModel",
    "JointBERTWithIntentSlotAttention",
    "create_model",
    "JointBERTDataset",
    "JointBERTDataModule",
    "create_data_module",
    "JointBERTTrainer",
    "evaluate_model",
    "JointBERTNLU",
    "load_nlu",
    # LLM Zero-Shot
    "LLMNLUClassifier",
    "create_llm_classifier",
    # LLM Clients
    "BaseLLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "MockClient",
    "get_client",
    # Prompts and constants
    "VALID_INTENTS",
    "VALID_SLOT_TYPES",
    "INTENT_DEFINITIONS",
    "SLOT_TYPE_DEFINITIONS",
    "get_system_prompt",
    "format_user_prompt",
    "get_few_shot_examples",
    "get_anthropic_tool_schema",
    "get_openai_json_schema",
]
