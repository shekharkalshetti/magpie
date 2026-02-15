"""
Schema Guard — Output format validation for LLM responses.

Validates LLM output against a developer-defined schema (Pydantic model or JSON Schema dict).
Supports two failure modes:
  - "block": Raise SchemaValidationError (prevents invalid output from reaching the app)
  - "flag": Allow output through, include validation info in log for review queue
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Type


@dataclass
class SchemaValidationResult:
    """Result of schema validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    parsed_output: Any = None
    schema_name: str = ""

    def to_dict(self) -> dict:
        """Convert to dict for log payload."""
        return {
            "valid": self.valid,
            "schema_name": self.schema_name,
            "errors": self.errors,
        }


class SchemaValidationError(Exception):
    """Raised when LLM output fails schema validation in block mode."""

    def __init__(
        self,
        message: str,
        output_text: str,
        schema_name: str,
        validation_errors: list[str],
    ):
        super().__init__(message)
        self.output_text = output_text
        self.schema_name = schema_name
        self.validation_errors = validation_errors


def _extract_json_from_text(text: str) -> str:
    """
    Extract JSON from LLM output that may contain markdown fences or surrounding text.

    Handles common patterns:
    - ```json ... ```
    - ``` ... ```
    - Raw JSON (starts with { or [)
    - JSON embedded in surrounding text
    """
    # Strip whitespace
    text = text.strip()

    # Try markdown code fences: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # If it already looks like JSON, return as-is
    if text.startswith(("{", "[")):
        return text

    # Try to find JSON object or array in the text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        if start_idx != -1:
            # Find matching closing bracket
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[start_idx : i + 1]

    # Return original text as fallback
    return text


def validate_output_schema(output_text: str, schema: Any) -> SchemaValidationResult:
    """
    Validate LLM output against a schema.

    Args:
        output_text: The raw text output from the LLM
        schema: A Pydantic BaseModel class or a JSON Schema dict

    Returns:
        SchemaValidationResult with valid/invalid status and any errors
    """
    if output_text is None:
        return SchemaValidationResult(
            valid=False,
            errors=["Output is None — nothing to validate"],
            schema_name=_get_schema_name(schema),
        )

    # Extract JSON from the output
    json_text = _extract_json_from_text(output_text)

    # Determine schema type and validate
    if _is_pydantic_model(schema):
        return _validate_pydantic(json_text, schema)
    elif isinstance(schema, dict):
        return _validate_json_schema(json_text, schema)
    else:
        return SchemaValidationResult(
            valid=False,
            errors=[
                f"Unsupported schema type: {type(schema).__name__}. "
                "Use a Pydantic BaseModel class or a JSON Schema dict."
            ],
            schema_name=str(type(schema).__name__),
        )


def _is_pydantic_model(schema: Any) -> bool:
    """Check if schema is a Pydantic BaseModel class."""
    try:
        from pydantic import BaseModel

        return isinstance(schema, type) and issubclass(schema, BaseModel)
    except ImportError:
        return False


def _get_schema_name(schema: Any) -> str:
    """Get a human-readable name for the schema."""
    if isinstance(schema, type):
        return schema.__name__
    elif isinstance(schema, dict):
        return schema.get("title", "JSONSchema")
    return str(type(schema).__name__)


def _validate_pydantic(json_text: str, schema: Type) -> SchemaValidationResult:
    """Validate against a Pydantic model."""
    from pydantic import ValidationError

    schema_name = schema.__name__

    try:
        parsed = schema.model_validate_json(json_text)
        return SchemaValidationResult(
            valid=True,
            parsed_output=parsed,
            schema_name=schema_name,
        )
    except ValidationError as e:
        errors = [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()]
        return SchemaValidationResult(
            valid=False,
            errors=errors,
            schema_name=schema_name,
        )
    except json.JSONDecodeError as e:
        return SchemaValidationResult(
            valid=False,
            errors=[f"Invalid JSON: {str(e)}"],
            schema_name=schema_name,
        )


def _validate_json_schema(json_text: str, schema: dict) -> SchemaValidationResult:
    """Validate against a JSON Schema dict."""
    schema_name = schema.get("title", "JSONSchema")

    # Parse JSON first
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return SchemaValidationResult(
            valid=False,
            errors=[f"Invalid JSON: {str(e)}"],
            schema_name=schema_name,
        )

    # Validate against schema
    try:
        import jsonschema

        jsonschema.validate(instance=data, schema=schema)
        return SchemaValidationResult(
            valid=True,
            parsed_output=data,
            schema_name=schema_name,
        )
    except ImportError:
        return SchemaValidationResult(
            valid=False,
            errors=[
                "jsonschema package required for dict schema validation. "
                "Install with: pip install jsonschema"
            ],
            schema_name=schema_name,
        )
    except jsonschema.ValidationError as e:
        return SchemaValidationResult(
            valid=False,
            errors=[e.message],
            schema_name=schema_name,
        )
