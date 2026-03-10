"""Prompt templates for LLM-based NLU.

Contains system prompts, few-shot examples, and output format specifications
for zero-shot intent classification and slot filling using Claude or GPT.
"""

from typing import List, Dict, Any

# PhoATIS Intent definitions with descriptions
INTENT_DEFINITIONS = {
    "flight": "Query about flight information, schedules, availability",
    "airfare": "Query about ticket prices, fares, costs",
    "ground_service": "Query about ground transportation services",
    "airline": "Query about airline information",
    "abbreviation": "Query about abbreviation meanings",
    "aircraft": "Query about aircraft types",
    "flight_time": "Query about flight duration or times",
    "quantity": "Query about quantities (seats, flights)",
    "distance": "Query about distance between locations",
    "city": "Query about city information",
    "airport": "Query about airport information",
    "ground_fare": "Query about ground transportation costs",
    "capacity": "Query about aircraft capacity",
    "flight_no": "Query about flight numbers",
    "meal": "Query about meal services",
    "restriction": "Query about restrictions or limitations",
    # Composite intents
    "airline#flight_no": "Query about airline and flight number",
    "airfare#flight": "Query about flight with fare information",
    "flight#airfare": "Query about flight with fare information (alternate)",
    "ground_fare#ground_service": "Query about ground service and fare",
    "ground_service#ground_fare": "Query about ground service and fare (alternate)",
    "flight#flight_no": "Query about flight with flight number",
    "flight#flight_time": "Query about flight with time information",
    "flight_no#flight_time": "Query about flight number with time",
    "city#flight_time": "Query about city with flight time",
    "aircraft#flight#flight_no": "Query about aircraft, flight and flight number",
}

# All valid intent labels from PhoATIS
VALID_INTENTS = [
    "abbreviation", "aircraft", "aircraft#flight#flight_no", "airfare",
    "airfare#flight", "airline", "airline#flight_no", "airport", "capacity",
    "city", "city#flight_time", "distance", "flight", "flight#flight_no",
    "flight#flight_time", "flight_no", "flight_no#flight_time", "flight_time",
    "ground_fare", "ground_fare#ground_service", "ground_service", "meal",
    "quantity", "restriction"
]

# Slot type definitions for air travel domain
SLOT_TYPE_DEFINITIONS = {
    # Location slots
    "fromloc.city_name": "Departure city name",
    "fromloc.airport_name": "Departure airport name",
    "fromloc.airport_code": "Departure airport code",
    "fromloc.state_name": "Departure state name",
    "fromloc.state_code": "Departure state code",
    "toloc.city_name": "Arrival city name",
    "toloc.airport_name": "Arrival airport name",
    "toloc.airport_code": "Arrival airport code",
    "toloc.state_name": "Arrival state name",
    "toloc.state_code": "Arrival state code",
    "toloc.country_name": "Arrival country name",
    "stoploc.city_name": "Stopover city name",
    "stoploc.airport_name": "Stopover airport name",
    "stoploc.state_code": "Stopover state code",
    "city_name": "Generic city name",
    "airport_name": "Generic airport name",
    "airport_code": "Generic airport code",
    "state_name": "Generic state name",
    "state_code": "Generic state code",
    # Date slots
    "depart_date.day_name": "Departure day of week (e.g., Monday, Tuesday)",
    "depart_date.day_number": "Departure day number",
    "depart_date.month_name": "Departure month name",
    "depart_date.date_relative": "Relative departure date (e.g., next week)",
    "depart_date.today_relative": "Today-relative departure (e.g., today, tomorrow)",
    "depart_date.year": "Departure year",
    "arrive_date.day_name": "Arrival day of week",
    "arrive_date.day_number": "Arrival day number",
    "arrive_date.month_name": "Arrival month name",
    "arrive_date.date_relative": "Relative arrival date",
    "arrive_date.today_relative": "Today-relative arrival",
    "return_date.day_name": "Return day of week",
    "return_date.day_number": "Return day number",
    "return_date.month_name": "Return month name",
    "return_date.date_relative": "Relative return date",
    "return_date.today_relative": "Today-relative return",
    "day_name": "Generic day name",
    "day_number": "Generic day number",
    "month_name": "Generic month name",
    # Time slots
    "depart_time.time": "Departure time (e.g., 7 AM, 15:00)",
    "depart_time.start_time": "Departure time range start",
    "depart_time.end_time": "Departure time range end",
    "depart_time.period_of_day": "Departure time period (morning, evening)",
    "depart_time.period_mod": "Departure period modifier (early, late)",
    "depart_time.time_relative": "Relative departure time (before, after)",
    "arrive_time.time": "Arrival time",
    "arrive_time.start_time": "Arrival time range start",
    "arrive_time.end_time": "Arrival time range end",
    "arrive_time.period_of_day": "Arrival time period",
    "arrive_time.period_mod": "Arrival period modifier",
    "arrive_time.time_relative": "Relative arrival time",
    "return_time.period_of_day": "Return time period",
    "return_time.period_mod": "Return period modifier",
    "time": "Generic time",
    "time_relative": "Generic relative time",
    # Flight attributes
    "airline_name": "Airline name (e.g., Vietnam Airlines, Jetstar)",
    "airline_code": "Airline code (e.g., VN, BL)",
    "flight_number": "Flight number",
    "aircraft_code": "Aircraft type code",
    "class_type": "Ticket class (economy, business)",
    "round_trip": "Round trip indicator",
    "flight_mod": "Flight modifier (nonstop, direct)",
    "flight_stop": "Number of stops",
    "flight_days": "Days flight operates",
    "flight_time": "Flight duration",
    "connect": "Connection information",
    # Fare and restrictions
    "fare_amount": "Fare amount",
    "fare_basis_code": "Fare basis code",
    "cost_relative": "Relative cost (cheapest, expensive)",
    "restriction_code": "Restriction code",
    "economy": "Economy class indicator",
    # Meal
    "meal": "Meal type",
    "meal_code": "Meal code",
    "meal_description": "Meal description",
    # Other
    "transport_type": "Transport type",
    "mod": "Generic modifier",
    "period_of_day": "Generic period of day",
    "today_relative": "Generic today-relative",
    "days_code": "Days code",
    "or": "Disjunction marker",
}

# All valid slot types (without B-/I- prefix)
VALID_SLOT_TYPES = sorted(set(SLOT_TYPE_DEFINITIONS.keys()))


def get_system_prompt(include_definitions: bool = True) -> str:
    """Generate system prompt for NLU task.

    Args:
        include_definitions: Whether to include detailed definitions

    Returns:
        System prompt string
    """
    intent_list = ", ".join(VALID_INTENTS)

    # Group slot types by category for readability
    slot_categories = {
        "Location": [s for s in VALID_SLOT_TYPES if any(
            x in s for x in ["fromloc", "toloc", "stoploc", "city", "airport", "state"]
        )],
        "Date": [s for s in VALID_SLOT_TYPES if "date" in s or s in [
            "day_name", "day_number", "month_name"
        ]],
        "Time": [s for s in VALID_SLOT_TYPES if "time" in s and "flight_time" not in s],
        "Flight": [s for s in VALID_SLOT_TYPES if any(
            x in s for x in ["airline", "flight", "aircraft", "class", "round_trip"]
        )],
        "Other": [s for s in VALID_SLOT_TYPES if not any(
            x in s for x in ["loc", "city", "airport", "state", "date", "time",
                             "airline", "flight", "aircraft", "class", "round_trip",
                             "day_", "month"]
        )],
    }

    slot_section = ""
    for category, slots in slot_categories.items():
        if slots:
            slot_section += f"\n  {category}: {', '.join(slots)}"

    prompt = f"""You are a Vietnamese Natural Language Understanding (NLU) system for air travel customer support.

Your task is to extract the user's intent and slot values from Vietnamese utterances about flights, airlines, and travel.

## Available Intents
{intent_list}

Note: Composite intents (with #) indicate multiple intents in one utterance.

## Available Slot Types{slot_section}

## Output Format
You MUST respond with valid JSON only, no other text:
{{"intent": "intent_name", "confidence": 0.0-1.0, "slots": {{"slot_type": "extracted_value"}}}}

## Rules
1. Intent must be one of the valid intents listed above
2. Confidence should reflect how certain you are (0.0-1.0)
3. Slots should contain ONLY the exact text spans from the input
4. For multi-word slot values, include the full phrase
5. If no slots are detected, return empty slots: {{}}
6. For Vietnamese text, preserve diacritics exactly as in input"""

    return prompt


def get_few_shot_examples() -> List[Dict[str, Any]]:
    """Return few-shot examples for in-context learning.

    Returns:
        List of example dictionaries with input and expected output
    """
    return [
        {
            "input": "tôi muốn tìm chuyến bay từ hà nội đến đà nẵng vào thứ hai",
            "output": {
                "intent": "flight",
                "confidence": 0.95,
                "slots": {
                    "fromloc.city_name": "hà nội",
                    "toloc.city_name": "đà nẵng",
                    "depart_date.day_name": "thứ hai"
                }
            }
        },
        {
            "input": "giá vé máy bay vietnam airlines đi phú quốc bao nhiêu",
            "output": {
                "intent": "airfare",
                "confidence": 0.92,
                "slots": {
                    "airline_name": "vietnam airlines",
                    "toloc.city_name": "phú quốc"
                }
            }
        },
        {
            "input": "chuyến bay số hiệu VN123 khởi hành lúc mấy giờ",
            "output": {
                "intent": "flight_time",
                "confidence": 0.90,
                "slots": {
                    "flight_number": "VN123"
                }
            }
        },
        {
            "input": "cho tôi xem các chuyến bay và giá vé từ sài gòn đến đà lạt ngày mai",
            "output": {
                "intent": "airfare#flight",
                "confidence": 0.88,
                "slots": {
                    "fromloc.city_name": "sài gòn",
                    "toloc.city_name": "đà lạt",
                    "depart_date.today_relative": "ngày mai"
                }
            }
        },
        {
            "input": "tôi cần đặt vé khứ hồi từ cần thơ đến hà nội vào sáng thứ tư",
            "output": {
                "intent": "flight",
                "confidence": 0.93,
                "slots": {
                    "fromloc.city_name": "cần thơ",
                    "toloc.city_name": "hà nội",
                    "round_trip": "khứ hồi",
                    "depart_time.period_of_day": "sáng",
                    "depart_date.day_name": "thứ tư"
                }
            }
        }
    ]


def format_user_prompt(utterance: str, include_examples: bool = True) -> str:
    """Format user message with optional few-shot examples.

    Args:
        utterance: Vietnamese input text
        include_examples: Whether to include few-shot examples

    Returns:
        Formatted user prompt
    """
    examples_text = ""
    if include_examples:
        examples = get_few_shot_examples()
        examples_text = "## Examples\n"
        for i, ex in enumerate(examples, 1):
            import json
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Input: \"{ex['input']}\"\n"
            examples_text += f"Output: {json.dumps(ex['output'], ensure_ascii=False)}\n"
        examples_text += "\n---\n"

    return f"""{examples_text}Now extract intent and slots from the following utterance:

Input: "{utterance}"

Output (JSON only):"""


def get_anthropic_tool_schema() -> Dict[str, Any]:
    """Get Anthropic tool_use schema for structured output.

    Returns:
        Tool definition for Claude's tool_use feature
    """
    return {
        "name": "extract_nlu",
        "description": "Extract intent and slot values from Vietnamese utterance",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": VALID_INTENTS,
                    "description": "The detected intent"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score for the intent"
                },
                "slots": {
                    "type": "object",
                    "description": "Extracted slot values as key-value pairs",
                    "additionalProperties": {
                        "type": "string"
                    }
                }
            },
            "required": ["intent", "confidence", "slots"]
        }
    }


def get_openai_json_schema() -> Dict[str, Any]:
    """Get JSON schema for OpenAI's JSON mode.

    Returns:
        JSON schema for response format
    """
    return {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": VALID_INTENTS
            },
            "confidence": {
                "type": "number"
            },
            "slots": {
                "type": "object",
                "additionalProperties": {
                    "type": "string"
                }
            }
        },
        "required": ["intent", "confidence", "slots"],
        "additionalProperties": False
    }
