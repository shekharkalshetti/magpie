"""
PII Detection and Redaction Module.

Uses local LLM to detect and redact PII (Personally Identifiable Information).
When enabled, automatically redacts PII from inputs before LLM execution.
"""

import json
import httpx
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PIIResult:
    """Result of PII detection and redaction."""

    contains_pii: bool
    redacted_text: Optional[str]
    pii_types: list[str]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/metadata."""
        return {
            "contains_pii": self.contains_pii,
            "pii_types": self.pii_types,
            "redacted": self.redacted_text is not None,
            "error": self.error,
        }


class PIIDetector:
    """
    Handles PII detection and redaction using local LLM.

    When PII is detected, it is automatically redacted from the input
    before passing to the wrapped LLM function.
    """

    def __init__(
        self, llm_url: str = "http://localhost:1234", model: str = "qwen2.5-1.5b-instruct"
    ):
        """
        Initialize PII detector.

        Args:
            llm_url: URL of local LM Studio instance
            model: Model to use for PII detection (default: qwen2.5-1.5b-instruct)
        """
        self.llm_url = llm_url
        self.model = model
        self.api_endpoint = f"{llm_url}/v1/chat/completions"
        self._cache: dict[str, PIIResult] = {}  # Cache for analyzed texts

    def _create_detection_prompt(self, text: str) -> str:
        """Create prompt for PII detection."""
        return f"""Replace ALL personally identifiable information with [REDACTED].

PII types: email addresses, phone numbers, names, SSN, credit cards, addresses, IP addresses, dates of birth, IDs.

Text: {text}

Return JSON in this exact format:
{{"contains_pii": true, "pii_types": ["email", "phone"], "redacted_text": "text with values replaced by [REDACTED]"}}

Example:
Input: "Email: john@test.com, Phone: 555-1234"
Output: {{"contains_pii": true, "pii_types": ["email", "phone"], "redacted_text": "Email: [REDACTED], Phone: [REDACTED]"}}

Now process the text above and return ONLY the JSON:"""

    def detect_and_redact(self, text: str) -> PIIResult:
        """
        Detect PII in text and redact if found.

        Args:
            text: Input text to analyze

        Returns:
            PIIResult with detection info and redacted text
        """
        if not text or not isinstance(text, str):
            return PIIResult(contains_pii=False, redacted_text=None, pii_types=[])

        # Check cache first (use first 100 chars as key)
        cache_key = text[:100]
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Call LM Studio API
            prompt = self._create_detection_prompt(text)

            response_http = httpx.post(
                self.api_endpoint,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 500,
                    "stream": False,
                },
                timeout=30,
            )

            response_http.raise_for_status()
            api_result = response_http.json()

            # Parse response from LM Studio format
            response_text = (
                api_result.get("choices", [{}])[0].get(
                    "message", {}).get("content", "{}")
            )

            # Clean up response text - remove markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            try:
                detection_result = json.loads(response_text)
            except json.JSONDecodeError as e:
                # Fail open - no PII detected if parsing fails
                error_result = PIIResult(
                    contains_pii=False,
                    redacted_text=None,
                    pii_types=[],
                    error=f"Failed to parse response: {str(e)}",
                )
                self._cache[cache_key] = error_result
                return error_result

            contains_pii = detection_result.get("contains_pii", False)
            pii_types = detection_result.get("pii_types", [])
            redacted_text = detection_result.get(
                "redacted_text") if contains_pii else None

            result: PIIResult = PIIResult(
                contains_pii=contains_pii, redacted_text=redacted_text, pii_types=pii_types
            )

            # Cache the result
            self._cache[cache_key] = result

            # Limit cache size to prevent memory issues
            if len(self._cache) > 100:
                # Remove oldest entry
                self._cache.pop(next(iter(self._cache)))

            return result

        except httpx.RequestError as e:
            # Fail open - if LM Studio is not available, don't block execution
            return PIIResult(
                contains_pii=False,
                redacted_text=None,
                pii_types=[],
                error=f"LM Studio connection failed: {str(e)}",
            )
        except Exception as e:
            # Fail open for other errors
            return PIIResult(
                contains_pii=False,
                redacted_text=None,
                pii_types=[],
                error=f"PII detection failed: {str(e)}",
            )

    def process_input(self, input_data: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        Process input data for PII detection and redaction.

        Args:
            input_data: Input data (can be dict, list, str, etc.)

        Returns:
            Tuple of (processed_input, pii_info)
        """
        # Convert input to analyzable text
        if isinstance(input_data, str):
            text_to_analyze = input_data
        elif isinstance(input_data, dict):
            # Look for common prompt fields
            text_to_analyze = (
                input_data.get("prompt")
                or input_data.get("message")
                or input_data.get("text")
                or input_data.get("content")
                or json.dumps(input_data)
            )
        elif isinstance(input_data, (list, tuple)):
            text_to_analyze = " ".join(str(item) for item in input_data)
        else:
            text_to_analyze = str(input_data)

        result = self.detect_and_redact(text_to_analyze)

        # If PII found and redacted, update input
        if result.contains_pii and result.redacted_text:
            if isinstance(input_data, str):
                return result.redacted_text, result.to_dict()
            elif isinstance(input_data, dict):
                # Update the dict with redacted text
                redacted_input = input_data.copy()
                for key in ["prompt", "message", "text", "content"]:
                    if key in redacted_input:
                        redacted_input[key] = result.redacted_text
                        break
                return redacted_input, result.to_dict()

        return input_data, result.to_dict() if result.contains_pii or result.error else None


# Global detector instance
_detector: Optional[PIIDetector] = None


def get_detector(
    llm_url: str = "http://localhost:1234", model: str = "qwen2.5-1.5b-instruct"
) -> PIIDetector:
    """Get global PII detector instance."""
    global _detector
    if _detector is None:
        _detector = PIIDetector(llm_url=llm_url, model=model)
    return _detector
