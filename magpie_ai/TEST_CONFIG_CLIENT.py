"""
Examples and tests for Magpie SDK configuration and HTTP client.

Demonstrates configuration loading, client usage, and fail-open behavior.
"""

# =============================================================================
# CONFIGURATION EXAMPLES
# =============================================================================

# Example 1: Configuration via environment variables (recommended)
"""
# Set these in your shell or .env file
export MAGPIE_API_KEY="tr_your_api_key_here"
export MAGPIE_BACKEND_URL="http://localhost:8000"
export MAGPIE_ENABLED="true"  # Optional, defaults to true

# Then in Python:
import magpie_ai

# Config is automatically loaded from environment
# No explicit configuration needed
"""

# Example 2: Programmatic configuration
"""
import magpie_ai

magpie_ai.configure(
    api_key="tr_your_api_key_here",
    backend_url="http://localhost:8000",
    enabled=True,
    timeout=5,
    fail_open=True
)
"""

# Example 3: Get current configuration
"""
from magpie_ai.config import get_config

config = get_config()
print(f"API Key configured: {config.is_configured()}")
print(f"Backend URL: {config.backend_url}")
print(f"Enabled: {config.enabled}")
print(f"Fail-open: {config.fail_open}")
print(f"Timeout: {config.timeout}s")
"""

# Example 4: Disable monitoring temporarily
"""
import magpie_ai

magpie_ai.configure(enabled=False)

# All monitoring calls will be no-ops
# Useful for testing or development
"""

# Example 5: Fail-closed mode (raise exceptions on errors)
"""
import magpie_ai

magpie_ai.configure(
    api_key="tr_key",
    backend_url="http://localhost:8000",
    fail_open=False  # Will raise exceptions instead of failing silently
)

# Use this in environments where you want to know about errors
# Not recommended for production
"""

# =============================================================================
# CLIENT USAGE EXAMPLES
# =============================================================================

# Example 1: Send a log asynchronously
"""
import asyncio
from datetime import datetime
from magpie_ai.client import get_client

async def send_example_log():
    client = get_client()
    
    success = await client.send_log(
        input={"prompt": "Hello, world"},
        output={"response": "Hi there!"},
        metadata={
            "user_id": "user123",
            "environment": "prod"
        },
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        duration_ms=1500,
        status="success",
        function_name="generate_completion",
        trace_id="trace-123"
    )
    
    if success:
        print("✓ Log sent successfully")
    else:
        print("✗ Failed to send log (fail-open)")

# Run it
asyncio.run(send_example_log())
"""

# Example 2: Send a log synchronously
"""
from datetime import datetime
from magpie_ai.client import get_client

client = get_client()

success = client.send_log_sync(
    metadata={"user_id": "user123"},
    status="success",
    function_name="my_function"
)

print(f"Log sent: {success}")
"""

# Example 3: Minimal log (only required fields)
"""
from magpie_ai.client import get_client

client = get_client()

# Only metadata is really useful, everything else is optional
success = client.send_log_sync(
    metadata={"user_id": "user123"},
    status="success"
)
"""

# Example 4: Log with error
"""
from magpie_ai.client import get_client

client = get_client()

success = client.send_log_sync(
    metadata={"user_id": "user123"},
    status="error",
    error_message="API rate limit exceeded"
)
"""

# =============================================================================
# FAIL-OPEN BEHAVIOR EXAMPLES
# =============================================================================

# Example 1: Missing configuration
"""
# If MAGPIE_API_KEY is not set
from magpie_ai.client import get_client

client = get_client()
success = client.send_log_sync(status="success")

# Result: False (with warning message)
# Output: [Triton] Warning: Failed to send log: ...
# Your application continues normally
"""

# Example 2: Backend is down
"""
import magpie_ai

magpie_ai.configure(
    api_key="tr_key",
    backend_url="http://localhost:9999",  # Wrong port
    timeout=2
)

from magpie_ai.client import get_client

client = get_client()
success = client.send_log_sync(status="success")

# Result: False (connection error)
# Output: [Triton] Warning: Failed to send log: ...
# Your application continues normally
"""

# Example 3: Network timeout
"""
import magpie_ai

magpie_ai.configure(
    api_key="tr_key",
    backend_url="http://example.com",  # Slow endpoint
    timeout=0.1  # Very short timeout
)

from magpie_ai.client import get_client

client = get_client()
success = client.send_log_sync(status="success")

# Result: False (timeout)
# Output: [Triton] Warning: Request timeout: ...
# Your application continues normally
"""

# Example 4: Invalid API key
"""
import magpie_ai

magpie_ai.configure(
    api_key="invalid_key",
    backend_url="http://localhost:8000"
)

from magpie_ai.client import get_client

client = get_client()
success = client.send_log_sync(status="success")

# Result: False (401 Unauthorized)
# Output: [Triton] Warning: HTTP error 401: ...
# Your application continues normally
"""

# Example 5: Malformed payload
"""
from magpie_ai.client import get_client
from datetime import datetime

client = get_client()

# Even with bad data, won't crash
success = client.send_log_sync(
    metadata={"key": "value"},
    started_at="not a datetime",  # Invalid type
    status="success"
)

# Result: False or True (depends on serialization)
# Your application continues normally
"""

# =============================================================================
# INTEGRATION PATTERNS
# =============================================================================

# Pattern 1: Check if configured before using
"""
from magpie_ai.config import get_config
from magpie_ai.client import get_client

config = get_config()

if config.is_configured():
    client = get_client()
    client.send_log_sync(status="success")
else:
    print("Magpie SDK not configured, skipping telemetry")
"""

# Pattern 2: Conditional monitoring
"""
import os
from magpie_ai.config import get_config

# Only enable in production
is_production = os.getenv("ENVIRONMENT") == "production"

import magpie_ai
magpie_ai.configure(enabled=is_production)

# Monitoring only happens in production
"""

# Pattern 3: Custom error handling
"""
from magpie_ai.client import get_client
import logging

logger = logging.getLogger(__name__)

client = get_client()
success = client.send_log_sync(
    metadata={"user_id": "123"},
    status="success"
)

if not success:
    logger.warning("Failed to send telemetry to Magpie")
"""

# =============================================================================
# FIELD REFERENCE
# =============================================================================

"""
All fields supported by send_log() and send_log_sync():

Required:
  None - all fields are optional

Optional:
  input: Dict[str, Any]
    - Execution input data
    - Example: {"prompt": "Hello"}
  
  output: Dict[str, Any]
    - Execution output data
    - Example: {"response": "Hi there!"}
  
  metadata: Dict[str, Any]
    - Metadata key-value pairs
    - Validated against project schema
    - Example: {"user_id": "123", "environment": "prod"}
  
  started_at: datetime
    - When execution started
    - Will be serialized to ISO format
  
  completed_at: datetime
    - When execution completed
    - Will be serialized to ISO format
  
  duration_ms: int
    - Execution duration in milliseconds
    - Example: 1500 (1.5 seconds)
  
  status: str
    - Execution status
    - Common values: "success", "error", "timeout"
  
  error_message: str
    - Error message if execution failed
    - Example: "API rate limit exceeded"
  
  function_name: str
    - Name of the monitored function
    - Example: "generate_completion"
  
  trace_id: str
    - Trace ID for grouping related executions
    - Example: "trace-abc-123"

Return value:
  bool - True if log was sent successfully, False otherwise
"""

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

"""
MAGPIE_API_KEY (required)
  Your Magpie API key
  Format: tr_xxxxx...
  Get this from: python -m backend.scripts.generate_api_key

MAGPIE_BACKEND_URL (optional)
  Backend URL
  Default: http://localhost:8000
  Production: https://api.magpie_ai.dev (or your deployed URL)

MAGPIE_ENABLED (optional)
  Whether monitoring is enabled
  Values: "true" or "false"
  Default: true
"""
