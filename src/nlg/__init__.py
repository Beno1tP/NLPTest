"""Natural Language Generation module.

Provides template-based response generation for Vietnamese AI Call Center,
with polite language and proper Vietnamese diacritics.

Exports:
    TemplateNLG     - Template-based NLG with slot filling
    NLGConfig       - Configuration for NLG behavior
    create_nlg      - Factory function to create NLG instance
    TEMPLATES       - Dictionary of response templates
    CITY_DISPLAY_NAMES   - City name formatting mappings
    AIRLINE_DISPLAY_NAMES - Airline name formatting mappings
"""

from .templates import (
    TemplateNLG,
    NLGConfig,
    create_nlg,
    TEMPLATES,
    CITY_DISPLAY_NAMES,
    AIRLINE_DISPLAY_NAMES,
    DAY_DISPLAY_NAMES,
    MONTH_DISPLAY_NAMES,
)

__all__ = [
    "TemplateNLG",
    "NLGConfig",
    "create_nlg",
    "TEMPLATES",
    "CITY_DISPLAY_NAMES",
    "AIRLINE_DISPLAY_NAMES",
    "DAY_DISPLAY_NAMES",
    "MONTH_DISPLAY_NAMES",
]
