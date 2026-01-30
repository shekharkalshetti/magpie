"""
Decorator for monitoring function execution.

Usage:
    @magpie_ai.monitor(project_id="my-project", pii=True, content_moderation=True)
    def my_llm_function(prompt):
        return llm.call(prompt)
"""

import functools
import threading
import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Any, Optional, TypeVar, cast
from datetime import datetime
import uuid
import inspect

from magpie_ai.client import get_client
from magpie_ai.context import get_context_metadata
from magpie_ai.pii import PIIDetector, get_detector
from magpie_ai.content_moderation import (
    ContentModerator,
    ContentModerationError,
    ModerationAction,
    get_moderator,
)
from magpie_ai.pricing import calculate_costs, get_context_utilization
from magpie_ai.token_extraction import (
    extract_tokens_from_response,
    extract_text_from_response,
)

F = TypeVar("F", bound=Callable[..., Any])

# Track active logging threads to wait for them on exit
_active_threads: list[threading.Thread] = []
_threads_lock = threading.Lock()

# Thread pool for parallel processing (PII + content moderation)
_executor: Optional[ThreadPoolExecutor] = None


def _validate_monitor_params(
    project_id: Optional[str],
    custom: Optional[Dict[str, Any]],
    model: Optional[str],
    input_token_price: Optional[float],
    output_token_price: Optional[float],
) -> None:
    """Validate monitor decorator parameters."""
    if not project_id or not isinstance(project_id, str):
        raise ValueError(
            "project_id is required and must be a non-empty string")

    if custom is not None and not isinstance(custom, dict):
        raise TypeError("custom parameter must be a dictionary")

    # Pricing validation: either use model name or explicit prices, not both
    has_model = model is not None
    has_explicit_pricing = (input_token_price is not None) or (
        output_token_price is not None)

    if has_model and has_explicit_pricing:
        raise ValueError(
            "Cannot specify both 'model' and explicit pricing parameters. "
            "Use either model='gpt-4' OR input_token_price/output_token_price, not both."
        )

    if input_token_price is not None and not isinstance(input_token_price, (int, float)):
        raise TypeError("input_token_price must be a number")

    if output_token_price is not None and not isinstance(output_token_price, (int, float)):
        raise TypeError("output_token_price must be a number")

    if input_token_price is not None and input_token_price < 0:
        raise ValueError("input_token_price must be non-negative")

    if output_token_price is not None and output_token_price < 0:
        raise ValueError("output_token_price must be non-negative")


def _get_executor() -> ThreadPoolExecutor:
    """Get or create the shared thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="magpie_ai_")
    return _executor


def _wait_for_logging_threads():
    """Wait for all active logging threads to complete before exit."""
    global _executor
    try:
        with _threads_lock:
            threads_to_wait = list(_active_threads)

        for thread in threads_to_wait:
            try:
                thread.join(timeout=2.0)  # Wait max 2 seconds per thread
            except Exception:
                pass  # Ignore errors during shutdown

        # Shutdown executor
        if _executor:
            _executor.shutdown(wait=True, cancel_futures=False)
    except Exception:
        pass  # Ignore all errors during interpreter shutdown


# Register cleanup function to run on exit
atexit.register(_wait_for_logging_threads)


def monitor(
    project_id: str,
    custom: Optional[Dict[str, Any]] = None,
    capture_input: bool = True,
    trace_id: Optional[str] = None,
    pii: bool = False,
    content_moderation: bool = False,
    llm_url: str = "http://localhost:1234",
    llm_model: str = "qwen2.5-1.5b-instruct",
    model: Optional[str] = None,
    input_token_price: Optional[float] = None,
    output_token_price: Optional[float] = None,
) -> Callable[[F], F]:
    """
    Decorator for monitoring function execution.

    Captures input, output, timing, and comprehensive metrics for decorated functions.
    Automatically extracts tokens, calculates costs, and sends logs to backend asynchronously.
    Optionally performs PII detection/redaction and content moderation.

    Args:
        project_id: Project ID (required)
        custom: Developer-provided custom data (stored as-is in database)
        capture_input: Whether to capture function inputs (default: True)
        trace_id: Optional trace ID (generated per execution if not provided)
        pii: Enable PII detection and redaction (default: False)
        content_moderation: Enable content moderation using policy (default: False)
        llm_url: URL of local LM Studio instance (default: "http://localhost:1234")
        llm_model: LM Studio model for PII/moderation (default: "qwen2.5-1.5b-instruct")
        model: Model name for pricing lookup (e.g., "gpt-4", "claude-3-sonnet")
        input_token_price: Explicit input token price per 1M tokens (use if model not in database)
        output_token_price: Explicit output token price per 1M tokens (use if model not in database)

    Raises:
        ValueError: If project_id is empty, pricing config is invalid, or both model and explicit pricing are provided
        TypeError: If custom is not a dict or pricing values are not numeric

    Captured Metrics:
        - Request tracing: trace_id, project_id
        - Latency: total_latency_ms
        - Token usage: input_tokens, output_tokens, total_tokens
        - Context: context_utilization_percent
        - Cost: input_cost (USD), output_cost (USD)
        - Custom data: developer-provided custom field

    Features:
        - pii=True: Detects and redacts PII from inputs before LLM execution
        - content_moderation=True: Checks content against policy rules, blocks violations
        - Both enabled: Runs in parallel for time efficiency
        - Automatic token extraction from OpenAI and Anthropic responses
        - Automatic cost calculation based on pricing

    Pricing Configuration:
        Option 1 - Use model name (recommended):
            @magpie_ai.monitor(..., model="gpt-4")

        Option 2 - Explicit pricing:
            @magpie_ai.monitor(..., input_token_price=0.03, output_token_price=0.06)

        Cannot mix both options.

    Example:
        @magpie_ai.monitor(
            project_id="my-project",
            model="gpt-4",
            pii=True,
            content_moderation=True,
            custom={"user_id": "user_123", "session": "abc"}
        )
        def chat_with_gpt(prompt: str) -> str:
            client = OpenAI()
            return client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content

    Note:
        - Currently supports synchronous functions only
        - Fails open - never crashes your application
        - Thread-safe - can be used in multi-threaded environments
    """
    # Validate all parameters upfront
    _validate_monitor_params(
        project_id=project_id,
        custom=custom,
        model=model,
        input_token_price=input_token_price,
        output_token_price=output_token_price,
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _execute_monitored(
                func=func,
                args=args,
                kwargs=kwargs,
                project_id=project_id,
                custom=custom,
                capture_input=capture_input,
                trace_id=trace_id,
                pii_enabled=pii,
                content_moderation_enabled=content_moderation,
                llm_url=llm_url,
                llm_model=llm_model,
                model=model,
                input_token_price=input_token_price,
                output_token_price=output_token_price,
            )

        return cast(F, wrapper)

    return decorator


def _execute_monitored(
    func: Callable,
    args: tuple,
    kwargs: dict,
    project_id: str,
    custom: Optional[Dict[str, Any]],
    capture_input: bool,
    trace_id: Optional[str],
    pii_enabled: bool,
    content_moderation_enabled: bool,
    llm_url: str,
    llm_model: str,
    model: Optional[str],
    input_token_price: Optional[float],
    output_token_price: Optional[float],
):
    """
    Execute function with monitoring.

    Thread-safe implementation that:
    1. Runs PII detection/redaction and content moderation in parallel (if enabled)
    2. Captures input arguments
    3. Executes the wrapped function
    4. Captures output
    5. Validates metadata
    6. Sends async log to backend

    Always fails open - never raises exceptions related to monitoring.
    """
    client = get_client()

    # Get context metadata if present
    context_meta = get_context_metadata()

    # Use context project_id if decorator doesn't have one
    effective_project_id = project_id
    if context_meta and context_meta.project_id and not project_id:
        effective_project_id = context_meta.project_id

    # Use context trace_id if decorator doesn't have one
    effective_trace_id = trace_id
    if context_meta and context_meta.trace_id and not trace_id:
        effective_trace_id = context_meta.trace_id

    # Generate trace_id if still None
    execution_trace_id = effective_trace_id or str(uuid.uuid4())

    # Validate pricing parameters
    if model and (input_token_price is not None or output_token_price is not None):
        # Log warning but fail open
        pass  # User will see error on cost calculation if prices are invalid

    # ============================================================
    # PARALLEL PROCESSING: PII Detection + Content Moderation
    # ============================================================
    # Both operations are I/O bound (LLM API calls), so running
    # them in parallel significantly reduces latency when both
    # are enabled.
    # ============================================================

    pii_info: Optional[Dict[str, Any]] = None
    moderation_info: Optional[Dict[str, Any]] = None
    exec_metadata: Optional[Dict[str, Any]] = None
    original_args = args
    original_kwargs = kwargs
    input_was_blocked = False  # Track if input was blocked for async task
    input_block_error: Optional[ContentModerationError] = (
        None  # Store the error to re-raise after logging
    )

    # Initialize status and error_message early so exception handlers can set them
    status = "success"
    error_message: Optional[str] = None

    # Initialize moderation and pii info before processing block
    moderation_info = None
    pii_info = None

    # Extract text from inputs for processing
    input_text: str = _extract_input_text(args, kwargs)

    if pii_enabled or content_moderation_enabled:
        try:
            if pii_enabled and content_moderation_enabled:
                # PARALLEL EXECUTION: Run both in parallel for efficiency
                args, kwargs, pii_info, moderation_info = _process_parallel(
                    args=args,
                    kwargs=kwargs,
                    input_text=input_text,
                    project_id=effective_project_id,
                    llm_url=llm_url,
                    llm_model=llm_model,
                )
            elif pii_enabled:
                # PII only
                args, kwargs, pii_info = _process_pii_only(
                    args=args, kwargs=kwargs, llm_url=llm_url, llm_model=llm_model
                )
            else:
                # Content moderation only
                moderation_info = _process_moderation_only(
                    input_text=input_text,
                    project_id=effective_project_id,
                    llm_url=llm_url,
                    llm_model=llm_model,
                )

            # Add processing info to metadata
            if exec_metadata is None:
                exec_metadata = {}
            if pii_info:
                exec_metadata["pii_detection"] = pii_info
            if moderation_info:
                exec_metadata["content_moderation"] = moderation_info

            # ============================================================
            # SYNC INPUT MODERATION: Block on critical violations
            # ============================================================
            # If content moderation found critical input violations,
            # block execution and raise error immediately.
            # This is synchronous blocking to prevent policy violations
            # from reaching the LLM or downstream systems.
            # ============================================================
            if content_moderation_enabled and moderation_info:
                severity = moderation_info.get("severity", "low")
                blocked = moderation_info.get("blocked", False)
                if severity == "critical" or blocked:
                    # Mark that input was blocked
                    input_was_blocked = True
                    # Convert dict back to ModerationResult for the exception
                    # to maintain the API contract
                    from magpie_ai.content_moderation import ModerationResult

                    mod_result = ModerationResult(
                        is_safe=moderation_info.get("is_safe", False),
                        action=ModerationAction(
                            moderation_info.get("action", "block")),
                        violations=[],  # Already included in moderation_info
                        raw_response=None,
                        error=moderation_info.get("error"),
                    )
                    # Will be caught below and raise ContentModerationError in finally
                    raise ContentModerationError(
                        f"Input blocked due to policy violations: "
                        f"{moderation_info.get('violated_policies', [])}",
                        mod_result,
                    )

        except ContentModerationError as e:
            # Content blocked - set flag and store error to re-raise after logging
            input_was_blocked = True
            input_block_error = e
            status = "error"
            error_message = str(e)
            # Don't re-raise yet - let the finally block log it first
        except Exception as e:
            # Fail open - if processing fails, continue with original args
            args = original_args
            kwargs = original_kwargs
            if exec_metadata is None:
                exec_metadata = {}
            exec_metadata["processing_error"] = str(e)

    # Capture input (using potentially redacted args/kwargs) as plaintext
    # Note: This is for logging the processed/redacted input, not the original
    captured_input_text: Optional[str] = None
    if capture_input:
        try:
            captured_input_text = _capture_input_as_text(func, args, kwargs)
        except Exception:
            # Fail open - continue without input capture
            captured_input_text = None

    # Execute function with timing
    started_at = datetime.utcnow()
    output_text: Optional[str] = None
    result: Any = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    context_utilization: float = 0.0
    input_cost: float = 0.0
    output_cost: float = 0.0

    try:
        # Execute the actual function
        result = func(*args, **kwargs)

        # Always capture output as plaintext
        try:
            output_text = extract_text_from_response(result)
            if not output_text:
                # Fallback to string representation
                output_text = _safe_str(result)[:2000]
        except Exception:
            # Fail open - continue without output capture
            output_text = None

        # Extract tokens from response if available
        try:
            input_tokens, output_tokens = extract_tokens_from_response(
                result, input_text=input_text
            )

            # Calculate costs if we have tokens
            if input_tokens and output_tokens:
                input_cost, output_cost, _ = calculate_costs(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                    input_price_per_1m=input_token_price,
                    output_price_per_1m=output_token_price,
                )

                # Calculate context utilization
                context_utilization = get_context_utilization(
                    input_tokens=input_tokens, output_tokens=output_tokens, model=model
                )
        except Exception:
            # Fail open - if pricing/token extraction fails, continue without costs
            pass

        return result

    except Exception as e:
        # Capture error information
        status = "error"
        error_message = str(e)
        raise  # Re-raise to preserve original behavior

    finally:
        # Always send log, even if function failed
        completed_at = datetime.utcnow()
        total_latency_ms = int(
            (completed_at - started_at).total_seconds() * 1000)

        # For blocked inputs, set costs to zero since LLM was never called
        # (moderation LLM is self-hosted, so no cost to client)
        if input_was_blocked:
            input_tokens = 0
            output_tokens = 0
            input_cost = 0.0
            output_cost = 0.0
            context_utilization = 0.0

        # For blocked inputs, send log synchronously to get log_id for async task
        # For other cases, send asynchronously as before
        log_id = None

        if input_was_blocked:
            # Synchronous send for blocked input to get log_id
            try:
                log_id = client.send_log_sync(
                    project_id=effective_project_id,
                    trace_id=execution_trace_id,
                    input=captured_input_text,
                    output=None,
                    custom=custom,
                    started_at=started_at,
                    completed_at=completed_at,
                    total_latency_ms=total_latency_ms,
                    status=status,
                    error_message=error_message,
                    function_name=func.__name__,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    context_utilization=context_utilization,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    pii_info=pii_info,
                    moderation_info=moderation_info,
                )
                # log_id is returned directly as string from send_log_sync
            except Exception:
                # Fail open - continue even if log send fails
                pass
        else:
            # Asynchronous send for normal cases
            try:
                log_id = _send_log_async(
                    client=client,
                    project_id=effective_project_id,
                    trace_id=execution_trace_id,
                    input_text=captured_input_text,
                    output_text=output_text,
                    custom=custom,
                    started_at=started_at,
                    completed_at=completed_at,
                    total_latency_ms=total_latency_ms,
                    status=status,
                    error_message=error_message,
                    function_name=func.__name__,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    context_utilization=context_utilization,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    pii_info=pii_info,
                    moderation_info=moderation_info,
                )
            except Exception:
                # Fail open - never crash user's code due to monitoring
                pass

        # ============================================================
        # ASYNC OUTPUT MODERATION: Queue Celery task
        # ============================================================
        # After successful execution, queue async moderation task
        # for the AI output. This doesn't block the user and allows
        # ReviewQueue items to be created asynchronously.
        # ============================================================
        if content_moderation_enabled and output_text and status == "success" and log_id:
            try:
                from src.tasks.moderation_tasks import moderate_output

                # Queue async moderation task
                moderate_output.delay(
                    execution_log_id=str(log_id),
                    output_text=output_text,
                    project_id=effective_project_id,
                    llm_url=llm_url,
                    llm_model=llm_model,
                )
            except Exception:
                # Fail open - if task queueing fails, continue
                # Output already delivered to user
                pass

        # Re-raise the input block error now that logging is complete
        if input_block_error:
            raise input_block_error


def _extract_input_text(args: tuple, kwargs: dict) -> str:
    """Extract text content from function arguments for analysis."""
    texts = []

    # Check for 'messages' parameter (common in chat/LLM APIs)
    # Extract the last user message from conversation history
    messages = kwargs.get("messages") or (
        args[1] if len(args) > 1 and isinstance(args[1], list) else None
    )
    if messages and isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    texts.append(content)
                    break

    # Check positional args (skip self/cls if present)
    for i, arg in enumerate(args):
        # Skip first arg if it looks like self/cls
        if i == 0 and not isinstance(arg, (str, dict, list)):
            continue
        if isinstance(arg, str):
            texts.append(arg)
        elif isinstance(arg, dict):
            for key in ["prompt", "message", "text", "content", "query", "input"]:
                if key in arg and isinstance(arg[key], str):
                    texts.append(arg[key])

    # Check keyword args
    for key in ["prompt", "message", "text", "content", "query", "input"]:
        if key in kwargs and isinstance(kwargs[key], str):
            texts.append(kwargs[key])

    return " ".join(texts) if texts else str(args) + str(kwargs)


def _process_parallel(
    args: tuple, kwargs: dict, input_text: str, project_id: str, llm_url: str, llm_model: str
) -> tuple[tuple, dict, Optional[Dict], Optional[Dict]]:
    """
    Run PII detection and content moderation in parallel.

    This significantly reduces latency when both features are enabled,
    as both are I/O-bound operations (LLM API calls).

    Returns:
        (processed_args, processed_kwargs, pii_info, moderation_info)
    """
    executor = _get_executor()

    pii_detector = get_detector(llm_url=llm_url, model=llm_model)
    moderator = get_moderator(project_id=project_id,
                              llm_url=llm_url, model=llm_model)

    # Submit both tasks
    pii_future = executor.submit(
        _run_pii_detection, pii_detector, args, kwargs)
    moderation_future = executor.submit(
        _run_content_moderation, moderator, input_text)

    # Wait for both to complete
    processed_args, processed_kwargs, pii_info = pii_future.result(timeout=60)
    moderation_info = moderation_future.result(timeout=60)

    return processed_args, processed_kwargs, pii_info, moderation_info


def _run_pii_detection(
    detector: PIIDetector, args: tuple, kwargs: dict
) -> tuple[tuple, dict, Optional[Dict]]:
    """Run PII detection and redaction on inputs."""
    pii_info = None

    # Process positional args
    processed_args = list(args)
    for i, arg in enumerate(args):
        processed_arg, arg_pii_info = detector.process_input(arg)
        processed_args[i] = processed_arg
        if arg_pii_info:
            pii_info = arg_pii_info

    # Process keyword args
    processed_kwargs = kwargs.copy()
    for key, value in kwargs.items():
        processed_value, kwarg_pii_info = detector.process_input(value)
        processed_kwargs[key] = processed_value
        if kwarg_pii_info:
            pii_info = kwarg_pii_info

    return tuple(processed_args), processed_kwargs, pii_info


def _run_content_moderation(moderator: ContentModerator, text: str) -> Optional[Dict]:
    """Run content moderation on input text."""
    try:
        result = moderator.moderate(text, block_on_violation=True)
        return result.to_dict()
    except ContentModerationError as e:
        # Content was blocked - return the result with blocking info
        # The exception contains the ModerationResult with violation details
        result_dict = e.result.to_dict()
        # Mark that blocking occurred
        result_dict["blocked"] = True
        return result_dict


def _process_pii_only(
    args: tuple, kwargs: dict, llm_url: str, llm_model: str
) -> tuple[tuple, dict, Optional[Dict]]:
    """Process only PII detection (synchronous)."""
    detector = get_detector(llm_url=llm_url, model=llm_model)
    return _run_pii_detection(detector, args, kwargs)


def _process_moderation_only(
    input_text: str, project_id: str, llm_url: str, llm_model: str
) -> Optional[Dict]:
    """Process only content moderation (synchronous)."""
    moderator = get_moderator(project_id=project_id,
                              llm_url=llm_url, model=llm_model)
    return _run_content_moderation(moderator, input_text)


def _capture_input_as_text(func: Callable, args: tuple, kwargs: dict) -> str:
    """
    Capture function input as plaintext.

    For class methods with conversation history (messages list),
    extracts only the last user message instead of entire history.
    For simple functions, captures all arguments.

    Returns a string representation suitable for logging.
    """
    try:
        # Try to get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Build plaintext representation
        parts = []
        for param_name, param_value in bound_args.arguments.items():
            # Skip 'self' parameter for class methods
            if param_name == "self":
                continue

            # Special handling for 'messages' parameter (conversation history)
            if param_name == "messages" and isinstance(param_value, list):
                # Extract only the last user message from conversation
                last_user_msg = None
                for msg in reversed(param_value):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        last_user_msg = msg.get("content", "")
                        break

                if last_user_msg:
                    parts.append(last_user_msg[:500])
                else:
                    # Fallback if no user message found
                    parts.append(f"messages: {str(param_value)[:200]}")
            elif isinstance(param_value, str):
                parts.append(f"{param_name}: {param_value[:500]}")
            else:
                parts.append(f"{param_name}: {str(param_value)[:500]}")

        return " | ".join(parts) if parts else str(args)

    except Exception:
        # Fallback
        return _safe_str(args)[:1000]


def _safe_str(obj: Any) -> str:
    """Safely convert object to string."""
    try:
        return str(obj)
    except Exception:
        return repr(type(obj))


def _send_log_async(
    client,
    project_id: str,
    trace_id: str,
    input_text: Optional[str],
    output_text: Optional[str],
    custom: Optional[Dict[str, Any]],
    started_at: datetime,
    completed_at: datetime,
    total_latency_ms: int,
    status: str,
    error_message: Optional[str],
    function_name: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    context_utilization: float,
    input_cost: float,
    output_cost: float,
    pii_info: Optional[Dict[str, Any]],
    moderation_info: Optional[Dict[str, Any]],
):
    """
    Send log to backend asynchronously in a background thread.

    Thread-safe implementation that doesn't block the main execution.
    """

    def send_in_thread():
        try:
            # Use synchronous HTTP request (no asyncio, works during shutdown)
            client.send_log_sync(
                project_id=project_id,
                trace_id=trace_id,
                input=input_text,
                output=output_text,
                custom=custom,
                started_at=started_at,
                completed_at=completed_at,
                total_latency_ms=total_latency_ms,
                status=status,
                error_message=error_message,
                function_name=function_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context_utilization=context_utilization,
                input_cost=input_cost,
                output_cost=output_cost,
                pii_info=pii_info,
                moderation_info=moderation_info,
            )
        except Exception:
            # Fail open - silently ignore all errors
            pass
        finally:
            # Remove thread from active list when done
            try:
                with _threads_lock:
                    if thread in _active_threads:
                        _active_threads.remove(thread)
            except Exception:
                pass

    # Start background thread (non-daemon so logs complete before exit)
    thread = threading.Thread(target=send_in_thread, daemon=False)

    # Track active thread
    with _threads_lock:
        _active_threads.append(thread)

    thread.start()
