"""
A2A (Agent-to-Agent) Protocol handler for OpenClaw Agent Container.

Implements the Google A2A protocol specification for inter-agent communication.
See: https://google.github.io/A2A/

Key components:
  - Agent Card: /.well-known/agent-card.json — describes agent capabilities
  - JSON-RPC 2.0 endpoints for task lifecycle:
      * tasks/send     — submit a new task or update an existing one
      * tasks/get      — retrieve task status and result
      * tasks/cancel   — cancel a running task

Design decisions:
  - Tasks are stored in-memory (suitable for single-instance; use DynamoDB for HA)
  - Each task maps to an OpenClaw chat completion invocation
  - Tenant isolation is maintained via task metadata
  - Streaming (tasks/sendSubscribe) is supported via SSE
"""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A2A Protocol Types
# ---------------------------------------------------------------------------

class TaskState(str, Enum):
    """A2A task lifecycle states (v0.3 lowercase enum values)."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class TaskStatus:
    """Current status of an A2A task."""
    state: TaskState
    message: Optional[str] = None
    timestamp: str = fiel--output truncated--
