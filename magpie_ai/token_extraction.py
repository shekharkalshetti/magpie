"""
Token extraction utilities for various LLM providers.

Handles response inspection for OpenAI, Anthropic, and other formats.
"""

from typing import Optional, Any, Tuple


def extract_tokens_from_response(
    response: Any, input_text: Optional[str] = None
) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract input and output tokens from LLM response.

    Supports:
    - OpenAI API responses (ChatCompletion, Completion)
    - Anthropic API responses (MessageResponse)
    - Generic objects with .usage attribute
    - Fallback: text-based token estimation from output text

    Args:
        response: Response object from LLM API (or string if just text output)
        input_text: Optional input text for estimation if response is string

    Returns:
        Tuple of (input_tokens, output_tokens), or (None, None) if cannot estimate
    """
    if not response:
        return None, None

    # If response is a string, it's the output text - estimate tokens from it
    if isinstance(response, str):
        output_tokens = estimate_tokens_from_text(response)
        # If we have input text, estimate input tokens too
        input_tokens = estimate_tokens_from_text(input_text) if input_text else None
        return input_tokens, output_tokens

    # Try to extract from common attributes
    input_tokens, output_tokens = _try_extract_usage(response)
    if input_tokens is not None and output_tokens is not None:
        return input_tokens, output_tokens

    # Fallback: if we extracted text, estimate from that
    output_text = extract_text_from_response(response)
    if output_text:
        output_tokens = estimate_tokens_from_text(output_text)
        input_tokens = estimate_tokens_from_text(input_text) if input_text else None
        return input_tokens, output_tokens

    return None, None


def _try_extract_usage(response: Any) -> Tuple[Optional[int], Optional[int]]:
    """Try to extract usage from response object."""

    # OpenAI format: response.usage.prompt_tokens, response.usage.completion_tokens
    if hasattr(response, "usage"):
        usage = response.usage
        if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
            return usage.prompt_tokens, usage.completion_tokens
        # Also try alternative names
        if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
            return usage.input_tokens, usage.output_tokens

    # Anthropic format: response.usage.input_tokens, response.usage.output_tokens
    if hasattr(response, "usage"):
        usage = response.usage
        if isinstance(usage, dict):
            if "input_tokens" in usage and "output_tokens" in usage:
                return usage["input_tokens"], usage["output_tokens"]
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                return usage["prompt_tokens"], usage["completion_tokens"]

    # Dictionary format
    if isinstance(response, dict):
        if "usage" in response:
            usage = response["usage"]
            if isinstance(usage, dict):
                # Try various naming conventions
                input_key = None
                output_key = None

                for inp in ["input_tokens", "prompt_tokens", "input"]:
                    if inp in usage:
                        input_key = inp
                        break

                for out in ["output_tokens", "completion_tokens", "output"]:
                    if out in usage:
                        output_key = out
                        break

                if input_key and output_key:
                    return usage[input_key], usage[output_key]

    return None, None


def extract_text_from_response(response: Any) -> Optional[str]:
    """
    Extract generated text from LLM response.

    Supports:
    - Plain strings (returns as-is)
    - OpenAI ChatCompletion (response.choices[0].message.content)
    - OpenAI Completion (response.choices[0].text)
    - Anthropic (response.content[0].text)
    - Dictionary formats

    Args:
        response: Response object from LLM API (or string)

    Returns:
        Generated text, or None if not found
    """
    if not response:
        return None

    # If response is already a string, return it
    if isinstance(response, str):
        return response

    # OpenAI ChatCompletion
    if hasattr(response, "choices") and len(response.choices) > 0:
        choice = response.choices[0]
        # Try message.content (ChatCompletion)
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            result = choice.message.content
            if isinstance(result, str):
                return result
        # Try text (Completion)
        if hasattr(choice, "text"):
            result = choice.text
            if isinstance(result, str):
                return result

    # Anthropic
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            # Try first block
            first = content[0]
            if hasattr(first, "text"):
                result = first.text
                if isinstance(result, str):
                    return result
            if isinstance(first, dict) and "text" in first:
                result = first["text"]
                if isinstance(result, str):
                    return result

    # Dictionary format
    if isinstance(response, dict):
        # OpenAI style
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                result = choice["message"]["content"]
                if isinstance(result, str):
                    return result
            if "text" in choice:
                result = choice["text"]
                if isinstance(result, str):
                    return result

        # Anthropic style
        if "content" in response and isinstance(response["content"], list):
            if len(response["content"]) > 0:
                first = response["content"][0]
                if isinstance(first, dict) and "text" in first:
                    result = first["text"]
                    if isinstance(result, str):
                        return result

    return None


def extract_model_from_response(response: Any) -> Optional[str]:
    """
    Extract model name from response if available.

    Args:
        response: Response object from LLM API

    Returns:
        Model name, or None if not found
    """
    if not response:
        return None

    # Check for direct model attribute
    if hasattr(response, "model"):
        result = response.model
        if isinstance(result, str):
            return result

    # Dictionary format
    if isinstance(response, dict) and "model" in response:
        result = response["model"]
        if isinstance(result, str):
            return result

    return None


def estimate_tokens_from_text(text: Optional[str]) -> Optional[int]:
    """
    Estimate token count from text using character-based heuristic.

    OpenAI's tokenizer typically converts text to ~1 token per 4 characters.
    This is a rough approximation for estimation purposes.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count, or None if text is None/empty
    """
    if not text:
        return None

    # Remove common whitespace padding
    text = text.strip()
    if not text:
        return None

    # Rough heuristic: 1 token per 4 characters (OpenAI tokenizer average)
    # Add some buffer for special tokens
    estimated = max(1, (len(text) + 3) // 4)
    return estimated
