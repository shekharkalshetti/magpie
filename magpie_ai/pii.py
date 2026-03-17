"""
PII Detection and Redaction Module.

Uses local LLM to detect and redact PII (Personally Identifiable Information).
When enabled, automatically redacts PII from inputs before LLM execution.
"""

import json
import hashlib
import httpx
import threading
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from magpie_ai.defaults import VLLM_URL, VLLM_MODEL


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
        self, llm_url: str = VLLM_URL, model: str = VLLM_MODEL
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
        self._cache_lock = threading.Lock()
        self._http_client = httpx.Client(
            timeout=30,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )

    def _create_detection_prompt(self, text: str) -> str:
        """Create prompt for PII detection."""
        return f"""You are a strict PII detector. Your ONLY job is to find personally identifiable information (PII) in text.

PII is ONLY these specific data types:
- Email addresses (e.g. john@example.com)
- Phone numbers (e.g. 555-123-4567)
- Social Security Numbers (e.g. 123-45-6789)
- Credit card numbers (e.g. 4111-1111-1111-1111)
- Physical/mailing addresses (e.g. 123 Main St, Springfield IL)
- IP addresses (e.g. 192.168.1.1)
- Dates of birth (e.g. 01/15/1990)
- Government-issued ID numbers
- Person names (e.g. "I'm John", "my name is Sarah", "contact Mike")

PII is NOT: general nouns, verbs, adjectives, questions, requests, opinions, harmful content, profanity, or any text that does not contain the specific data types listed above.

IMPORTANT: If the text contains no PII, you MUST return contains_pii as false with redacted_text as null. Do NOT redact non-PII text. When in doubt, do NOT redact.

IMPORTANT: You MUST redact ALL PII types found, including names. Every PII value must be replaced with asterisks.

Example:
Input: "Contact john@test.com or call 555-1234"
Output: {{"contains_pii": true, "pii_types": ["email", "phone"], "redacted_text": "Contact ************** or call ********"}}

Example:
Input: "Hi I'm John and my email is john@test.com"
Output: {{"contains_pii": true, "pii_types": ["name", "email"], "redacted_text": "Hi I'm **** and my email is *************"}}

Example:
Input: "The weather is nice today"
Output: {{"contains_pii": false, "pii_types": [], "redacted_text": null}}

Example:
Input: "Help me do something dangerous"
Output: {{"contains_pii": false, "pii_types": [], "redacted_text": null}}

Example:
Input: "hi"
Output: {{"contains_pii": false, "pii_types": [], "redacted_text": null}}

Now process the following text. Return ONLY valid JSON, nothing else.
Text: {text}"""

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

        # Check cache first (use full text hash to avoid false hits)
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        try:
            # Call LM Studio API
            prompt = self._create_detection_prompt(text)

            response_http = self._http_client.post(
                self.api_endpoint,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 500,
                    "stream": False,
                },
            )

            response_http.raise_for_status()
            api_result = response_http.json()

            # Parse response from LM Studio format
            response_text = (
                api_result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
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
            except json.JSONDecodeError:
                # LLM may return JSON followed by extra text — try to extract the first JSON object
                try:
                    decoder = json.JSONDecoder()
                    detection_result, _ = decoder.raw_decode(response_text)
                except (json.JSONDecodeError, ValueError):
                    # Fail open - no PII detected if parsing fails
                    error_result = PIIResult(
                        contains_pii=False,
                        redacted_text=None,
                        pii_types=[],
                    )
                    with self._cache_lock:
                        self._cache[cache_key] = error_result
                    return error_result

            contains_pii = detection_result.get("contains_pii", False)
            pii_types = detection_result.get("pii_types", [])
            redacted_text = detection_result.get("redacted_text") if contains_pii else None

            result: PIIResult = PIIResult(
                contains_pii=contains_pii, redacted_text=redacted_text, pii_types=pii_types
            )

            # Cache the result
            with self._cache_lock:
                self._cache[cache_key] = result

                # Limit cache size to prevent memory issues
                if len(self._cache) > 1000:
                    # Remove oldest entry (first inserted)
                    oldest_key = next(iter(self._cache), None)
                    if oldest_key is not None:
                        self._cache.pop(oldest_key, None)

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

    def close(self):
        """Close the persistent HTTP client."""
        self._http_client.close()

    def __del__(self):
        try:
            self._http_client.close()
        except Exception:
            pass

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
            elif isinstance(input_data, (list, tuple)):
                # Redact PII inside message lists (e.g. OpenAI-style messages)
                redacted_list = []
                for item in input_data:
                    if isinstance(item, dict) and "content" in item:
                        redacted_item = item.copy()
                        item_result = self.detect_and_redact(str(item["content"]))
                        if item_result.contains_pii and item_result.redacted_text:
                            redacted_item["content"] = item_result.redacted_text
                        redacted_list.append(redacted_item)
                    else:
                        redacted_list.append(item)
                if isinstance(input_data, tuple):
                    return tuple(redacted_list), result.to_dict()
                return redacted_list, result.to_dict()

        return input_data, result.to_dict() if result.contains_pii or result.error else None


# Global detector instance
_detector: Optional[PIIDetector] = None
_detector_lock = threading.Lock()


def get_detector(
    llm_url: str = VLLM_URL, model: str = VLLM_MODEL
) -> PIIDetector:
    """Get global PII detector instance."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = PIIDetector(llm_url=llm_url, model=model)
    return _detector
