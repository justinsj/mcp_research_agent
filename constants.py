"""
Constants & Type Definitions
============================
Defines the domain-specific data structures and process configuration
for the MCP template agentic loop.

Customisation Guide:
    1. Define your output schema as TypedDicts (replace FinalAnswerDict).
    2. Add intermediate data-gathering steps (like GatheredData).
    3. Set PROCESS to the ordered list of TypedDict classes the LLM
       must complete, ending with your final answer type.
    4. Write domain-specific INSTRUCTIONS for the LLM.
    5. Adjust MAX_LOOPS if your task needs more/fewer iterations.

Exports:
    - MAX_LOOPS: int
    - ResearchItem, GatheredData, FinalAnswerDict, QualityControlCheckResult
    - PROCESS: List[TypedDict classes]
    - FINAL_ANSWER_STRUCTURE: str (human-readable schema description)
    - INSTRUCTIONS: str (LLM instructions)
    - format_enum_values(), get_typeddict_structure()
"""

from typing_extensions import (
    Any, TypedDict, List, Tuple, get_type_hints,
    Literal, Optional, Union,
)
from enum import Enum
import logging


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

# Maximum agentic loops before the client forces a final answer.
MAX_LOOPS = 8
LOG_LEVEL = logging.INFO

# ─────────────────────────────────────────────────────────────────────
# Data Structures (TypedDicts)
# ─────────────────────────────────────────────────────────────────────
# Each TypedDict represents one step the LLM must fill out IN ORDER.
# Replace or extend these with your own domain-specific structures.

class ReferenceItem(TypedDict):
    """A single reference item."""
    intended_query: str
    source_url: str
    relevant_snippet: str

class DisputeCase(TypedDict):
    """A single piece of information gathered from the web."""
    reference_items: List[ReferenceItem]
    defendant_name: str
    defendant_url: Optional[str]
    plaintiff_name: str
    plaintiff_url: Optional[str]
    
class EntityInformation(TypedDict):
    entity_name: str
    entity_url: str
    description: str
    founding_or_birth_year: int
    profession: Optional[str]

class DefendantInformation(EntityInformation):
    pass

class PlaintiffInformation(EntityInformation):
    pass

class FinalAnswerDict(TypedDict):
    """The shape of the final output the LLM must produce."""
    result: str
    defendant_name: str
    plaintiff_name: str
    reference_items: List[ReferenceItem]

class QualityControlCheckResult(TypedDict):
    """Result returned by the instruction_following_check MCP tool."""
    is_quality_standard_met: bool
    issues: List[str]
    other_notes: str


# ─────────────────────────────────────────────────────────────────────
# Ordered Enrichment Process
# ─────────────────────────────────────────────────────────────────────
# The LLM completes these steps one at a time, in order.
# The last entry MUST be the final answer type.
PROCESS: List[Any] = [
    DisputeCase,
    DefendantInformation,
    PlaintiffInformation,
    FinalAnswerDict,
]

# Lookup map: TypedDict class name -> class (for runtime validation by name)
PROCESS_BY_NAME = {cls.__name__: cls for cls in PROCESS}


# ─────────────────────────────────────────────────────────────────────
# Helpers: pretty-print TypedDicts for the LLM
# ─────────────────────────────────────────────────────────────────────


def format_enum_values(enum_class: type) -> str:
    """Format an Enum class's values as a pipe-separated string for display."""
    return " | ".join(repr(m.value) for m in enum_class)


def get_typeddict_structure(
    td_class: Any, indent: int = 0, include_name: bool = True,
) -> str:
    """
    Recursively render a TypedDict class as a human-readable schema string.

    Handles nested TypedDicts, Lists, Tuples, Unions, Optionals, Literals,
    and Enums. The output is intended for inclusion in LLM system prompts
    so the model knows what JSON structure to produce.

    Args:
        td_class: The TypedDict class to render.
        indent: Current indentation level (for recursion).
        include_name: Whether to include the class name as a header line.

    Returns:
        A multi-line string describing the structure.
    """
    prefix = "  " * indent
    result = ""
    if include_name:
        result += f"{prefix}{td_class.__name__}:\n"
    hints = get_type_hints(td_class)

    for key, value_type in hints.items():
        field_prefix = "  " * (indent + 1)

        if isinstance(value_type, type) and issubclass(value_type, Enum):
            result += (
                f"{field_prefix}{key}: "
                f"{format_enum_values(value_type)}\n"
            )
        elif hasattr(value_type, "__annotations__"):
            result += f"{field_prefix}{key}:\n"
            result += get_typeddict_structure(
                value_type, indent + 2, include_name=False,
            )
        elif getattr(value_type, "__origin__", None) in (list, List):
            inner = value_type.__args__[0]
            result += f"{field_prefix}{key}: List\n"
            if hasattr(inner, "__annotations__"):
                result += (
                    f"{field_prefix}  └─ {inner.__name__}:\n"
                )
                result += get_typeddict_structure(
                    inner, indent + 3, include_name=False,
                )
            else:
                type_name = getattr(inner, "__name__", str(inner))
                result += f"{field_prefix}  └─ {type_name}\n"
        elif getattr(value_type, "__origin__", None) in (tuple, Tuple):
            inner_types = ", ".join(
                t.__name__ for t in value_type.__args__
            )
            result += (
                f"{field_prefix}{key}: Tuple[{inner_types}]\n"
            )
        elif getattr(value_type, "__origin__", None) is Union:
            args = value_type.__args__
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1 and len(args) == 2:
                inner = non_none[0]
                type_name = getattr(inner, "__name__", str(inner))
                result += (
                    f"{field_prefix}{key}: Optional[{type_name}]\n"
                )
            else:
                union_types = ", ".join(
                    getattr(a, "__name__", str(a)) for a in args
                )
                result += (
                    f"{field_prefix}{key}: Union[{union_types}]\n"
                )
        elif getattr(value_type, "__origin__", None) is Literal:
            literals = " | ".join(repr(l) for l in value_type.__args__)
            result += f"{field_prefix}{key}: {literals}\n"
        else:
            type_name = getattr(
                value_type, "__name__", str(value_type),
            )
            result += f"{field_prefix}{key}: {type_name}\n"

    return result


FINAL_ANSWER_STRUCTURE = (
    "Desired data structure:\n"
    + get_typeddict_structure(FinalAnswerDict)
)


# ─────────────────────────────────────────────────────────────────────
# Domain-Specific Instructions
# ─────────────────────────────────────────────────────────────────────
# Customise these for your use-case. They are injected into the LLM's
# system prompt and also passed to the instruction_following_check tool.
INSTRUCTIONS = """
GENERAL GUIDELINES
- Only reference facts that can be verified from actual sources.
- Do not hallucinate or fabricate any information.
- If you cannot find information, state that clearly.
"""
