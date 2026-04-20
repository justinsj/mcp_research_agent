"""
Validation Helpers
==================
Provides recursive TypedDict validation for checking whether a plain
dict conforms to a given TypedDict schema at runtime.

Used by the MCP server's `final_answer_check` and
`instruction_following_check` tools to verify that the LLM's output
matches the expected structure before accepting it.

Supported types:
    - Nested TypedDicts
    - List[T] (with element-level validation)
    - Tuple[T1, T2, ...] (fixed-length, typed)
    - Literal["a", "b"] (enum-like values)
    - Basic types (str, int, float, bool)

Usage:
    from helpers import validate_type
    result = validate_type(data, FinalAnswerDict)
    # result == {"status": "ok", "message": data}
    # or      {"status": "error", "message": ["Missing key: x", ...]}
"""

from typing import Any, Dict, List, Tuple, get_type_hints, Literal


def _validate_typed_dict(
    data: Any, td_class: Any, path: str = "",
) -> List[str]:
    """
    Recursively validate that `data` conforms to `td_class` schema.

    Checks for:
        - Missing required keys
        - Unexpected extra keys
        - Type mismatches (including nested dicts, lists, tuples, literals)

    Args:
        data: The dict (or value) to validate.
        td_class: The TypedDict class defining the expected schema.
        path: Dot-separated path for nested error messages (internal use).

    Returns:
        A list of human-readable error strings. Empty list means valid.
    """
    errors: List[str] = []
    hints = get_type_hints(td_class)

    if not isinstance(data, dict):
        return [
            f"{path or td_class.__name__} should be a dict, "
            f"got {type(data).__name__}"
        ]

    for key, expected_type in hints.items():
        full_path = f"{path}.{key}" if path else key
        if key not in data:
            errors.append(f"Missing key: {full_path}")
            continue

        value = data[key]

        # Nested TypedDict
        if hasattr(expected_type, "__annotations__"):
            if not isinstance(value, dict):
                errors.append(
                    f"{full_path} should be a dict, "
                    f"got {type(value).__name__}"
                )
            else:
                errors += _validate_typed_dict(
                    value, expected_type, full_path,
                )

        # List
        elif getattr(expected_type, "__origin__", None) in (list, List):
            inner_type = expected_type.__args__[0]
            if not isinstance(value, list):
                errors.append(
                    f"{full_path} should be a list, "
                    f"got {type(value).__name__}"
                )
            else:
                for i, item in enumerate(value):
                    item_path = f"{full_path}[{i}]"
                    if hasattr(inner_type, "__annotations__"):
                        errors += _validate_typed_dict(
                            item, inner_type, item_path,
                        )
                    elif getattr(
                        inner_type, "__origin__", None,
                    ) is Literal:
                        if item not in inner_type.__args__:
                            errors.append(
                                f"{item_path} should be one of "
                                f"{inner_type.__args__}, got {item!r}"
                            )
                    elif not isinstance(item, inner_type):
                        errors.append(
                            f"{item_path} should be "
                            f"{inner_type.__name__}, "
                            f"got {type(item).__name__}"
                        )

        # Tuple
        elif getattr(expected_type, "__origin__", None) in (
            tuple, Tuple,
        ):
            if (
                not isinstance(value, (tuple, list))
                or len(value) != len(expected_type.__args__)
            ):
                errors.append(
                    f"{full_path} should be "
                    f"Tuple{expected_type.__args__}, got {value}"
                )
            else:
                for i, (subval, subtype) in enumerate(
                    zip(value, expected_type.__args__),
                ):
                    if not isinstance(subval, subtype):
                        errors.append(
                            f"{full_path}[{i}] should be "
                            f"{subtype.__name__}, "
                            f"got {type(subval).__name__}"
                        )

        # Literal
        elif getattr(expected_type, "__origin__", None) is Literal:
            if value not in expected_type.__args__:
                errors.append(
                    f"{full_path} should be one of "
                    f"{expected_type.__args__}, got {value!r}"
                )

        # Basic types
        else:
            # Accept int where float is expected (standard numeric coercion)
            check_type = (
                (int, float) if expected_type is float else expected_type
            )
            if not isinstance(value, check_type):
                errors.append(
                    f"{full_path} should be "
                    f"{expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

    # Unexpected keys
    for key in data.keys():
        if key not in hints:
            full_path = f"{path}.{key}" if path else key
            errors.append(f"Unexpected key: {full_path}")

    return errors


def validate_type(
    profile: Dict[str, Any], typed_class: Any,
) -> Dict[str, Any]:
    """
    Validate a dict against a TypedDict schema and return a status dict.

    This is the main public interface for validation. Returns a dict
    with "status" ("ok" or "error") and "message" (the data on success,
    or a list of error strings on failure).

    Args:
        profile: The dict to validate.
        typed_class: The TypedDict class to validate against.

    Returns:
        {"status": "ok", "message": profile} on success, or
        {"status": "error", "message": [error strings]} on failure.
    """
    errors = _validate_typed_dict(profile, typed_class)
    if errors:
        return {"status": "error", "message": errors}
    return {"status": "ok", "message": profile}
