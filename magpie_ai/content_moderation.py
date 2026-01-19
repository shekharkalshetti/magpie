"""
Content Moderation Module.

Uses local LLM with system prompts from shared policy files
to detect policy violations in content.
"""

import json
import httpx
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from magpie_ai.prompt import PromptCache, get_system_prompt_or_default


class ModerationSeverity(Enum):
    """Severity levels for content moderation."""
    CRITICAL = "critical"  # Block immediately
    HIGH = "high"          # Should be blocked
    MEDIUM = "medium"      # Flag for review
    LOW = "low"            # Warning only


class ModerationAction(Enum):
    """Recommended actions for detected violations."""
    BLOCK = "block"       # Prevent execution
    FLAG = "flag"         # Allow but flag for review
    WARN = "warn"         # Allow with warning
    ALLOW = "allow"       # No issues detected


@dataclass
class ModerationViolation:
    """Represents a single content moderation violation."""
    category: str
    severity: ModerationSeverity
    description: str
    action: ModerationAction


@dataclass
class ModerationResult:
    """Result of content moderation analysis."""
    is_safe: bool
    action: ModerationAction
    violations: List[ModerationViolation]
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/metadata."""
        return {
            "is_safe": self.is_safe,
            "action": self.action.value,
            "violations": [
                {
                    "category": v.category,
                    "severity": v.severity.value,
                    "description": v.description,
                    "action": v.action.value
                }
                for v in self.violations
            ],
            "error": self.error
        }


class ContentModerationError(Exception):
    """Raised when content is blocked by moderation."""

    def __init__(self, message: str, result: ModerationResult):
        super().__init__(message)
        self.result = result


class ContentModerator:
    """
    Handles content moderation using local LLM with policy-based system prompts.

    Uses PromptCache for efficient file-based prompt loading with auto-reload.
    """

    def __init__(
        self,
        project_id: str,
        llm_url: str = "http://localhost:1234",
        model: str = "qwen2.5-1.5b-instruct"
    ):
        """
        Initialize content moderator.

        Args:
            project_id: Project ID to load policy from
            llm_url: URL of local LM Studio instance
            model: Model to use for content moderation
        """
        self.project_id = project_id
        self.llm_url = llm_url
        self.model = model
        self.api_endpoint = f"{llm_url}/v1/chat/completions"
        self._prompt_cache = PromptCache(project_id)

    def _create_moderation_prompt(self, content: str) -> str:
        """Create user prompt for content moderation analysis."""
        return f"""Analyze this content for policy violations:

<content>
{content}
</content>

Return JSON in this exact format:
{{"is_safe": true, "action": "allow", "violations": []}}

If violations found:
{{"is_safe": false, "action": "block", "violations": [{{"category": "...", "severity": "critical|high|medium|low", "description": "...", "action": "block|flag|warn"}}]}}

Return ONLY the JSON:"""

    def moderate(
        self,
        content: str,
        block_on_violation: bool = True
    ) -> ModerationResult:
        """
        Analyze content for policy violations.

        Args:
            content: Content to analyze
            block_on_violation: If True, raise ContentModerationError on critical/high violations

        Returns:
            ModerationResult with analysis details

        Raises:
            ContentModerationError: If block_on_violation=True and critical/high violations found
        """
        if not content or not isinstance(content, str):
            return ModerationResult(
                is_safe=True,
                action=ModerationAction.ALLOW,
                violations=[]
            )

        try:
            # Get system prompt from cached policy file
            system_prompt = self._prompt_cache.get()
            if not system_prompt:
                system_prompt = get_system_prompt_or_default(self.project_id)

            # Call LM Studio API
            user_prompt = self._create_moderation_prompt(content)

            response = httpx.post(
                self.api_endpoint,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 500,
                    "stream": False
                },
                timeout=30
            )

            response.raise_for_status()
            api_result = response.json()

            # Parse response
            response_text = api_result.get("choices", [{}])[0].get(
                "message", {}
            ).get("content", "{}")

            # Clean up response - remove markdown code blocks
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError:
                # Default to safe if parsing fails (fail open)
                return ModerationResult(
                    is_safe=True,
                    action=ModerationAction.ALLOW,
                    violations=[],
                    raw_response=response_text[:200],
                    error="Failed to parse moderation response"
                )

            # Build result
            is_safe = parsed.get("is_safe", True)
            action_str = parsed.get("action", "allow").lower()
            action = ModerationAction(action_str) if action_str in [
                a.value for a in ModerationAction] else ModerationAction.ALLOW

            violations = []
            for v in parsed.get("violations", []):
                try:
                    severity_str = v.get("severity", "low").lower()
                    severity = ModerationSeverity(severity_str) if severity_str in [
                        s.value for s in ModerationSeverity] else ModerationSeverity.LOW

                    action_v_str = v.get("action", "warn").lower()
                    action_v = ModerationAction(action_v_str) if action_v_str in [
                        a.value for a in ModerationAction] else ModerationAction.WARN

                    violations.append(ModerationViolation(
                        category=v.get("category", "unknown"),
                        severity=severity,
                        description=v.get("description", ""),
                        action=action_v
                    ))
                except (KeyError, ValueError):
                    continue

            result = ModerationResult(
                is_safe=is_safe,
                action=action,
                violations=violations,
                raw_response=response_text[:200]
            )

            # Block if requested and severe violations found
            if block_on_violation and not is_safe:
                has_blocking = any(
                    v.severity in (ModerationSeverity.CRITICAL,
                                   ModerationSeverity.HIGH)
                    or v.action == ModerationAction.BLOCK
                    for v in violations
                )
                if has_blocking or action == ModerationAction.BLOCK:
                    categories = ", ".join(v.category for v in violations[:3])
                    raise ContentModerationError(
                        f"Content blocked by moderation policy. Violations: {categories}",
                        result
                    )

            return result

        except ContentModerationError:
            raise
        except httpx.RequestError as e:
            # Fail open - if LM Studio unavailable, allow content
            return ModerationResult(
                is_safe=True,
                action=ModerationAction.ALLOW,
                violations=[],
                error=f"LM Studio connection failed: {str(e)}"
            )
        except Exception as e:
            # Fail open for other errors
            return ModerationResult(
                is_safe=True,
                action=ModerationAction.ALLOW,
                violations=[],
                error=f"Content moderation failed: {str(e)}"
            )

    def process_input(
        self,
        input_data: Any,
        block_on_violation: bool = True
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        """
        Process input data for content moderation.

        Args:
            input_data: Input data (str, dict, list, etc.)
            block_on_violation: Whether to raise error on violations

        Returns:
            Tuple of (input_data, moderation_info)

        Raises:
            ContentModerationError: If block_on_violation=True and violations found
        """
        # Extract text to analyze
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, dict):
            text = (
                input_data.get("prompt") or
                input_data.get("message") or
                input_data.get("text") or
                input_data.get("content") or
                json.dumps(input_data)
            )
        elif isinstance(input_data, (list, tuple)):
            text = " ".join(str(item) for item in input_data)
        else:
            text = str(input_data)

        result = self.moderate(text, block_on_violation=block_on_violation)

        # Return original data with moderation info
        # Content moderation doesn't modify input, just analyzes it
        return input_data, result.to_dict()


# Global moderator instances per project
_moderators: Dict[str, ContentModerator] = {}


def get_moderator(
    project_id: str,
    llm_url: str = "http://localhost:1234",
    model: str = "qwen2.5-1.5b-instruct"
) -> ContentModerator:
    """Get or create content moderator for a project."""
    key = f"{project_id}:{llm_url}:{model}"
    if key not in _moderators:
        _moderators[key] = ContentModerator(
            project_id=project_id,
            llm_url=llm_url,
            model=model
        )
    return _moderators[key]
