"""
AgentCore Memory — optional cloud persistence layer for cross-session memory.

openclaw's native memory (Markdown + SQLite) continues to work within the
container lifecycle.  This module provides the *optional* AgentCore Memory
integration that persists summaries to AWS so they survive container teardown.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import logging
import os
import sys
import time
from typing import Optional

import boto3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

# Memory store ID from environment (requirement 6.1)
MEMORY_STORE_ID = os.environ.get("MEMORY_STORE_ID", "default")


def _memory_client():
    """
    Factory for the bedrock-agentcore-memory boto3 client.

    Using a factory (rather than a module-level singleton) makes the client
    easy to mock in tests — callers can monkeypatch `memory._memory_client`.
    """
    return boto3.client(
        "bedrock-agentcore-memory",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )


def _namespace(tenant_id: str) -> str:
    """Return the Memory namespace for a tenant (requirement 6.1)."""
    return f"tenant_{tenant_id}"


async def load_memory_on_session_start(tenant_id: str) -> Optional[str]:
    """
    Retrieve historical memory summaries for *tenant_id* at session start.

    On ANY exception the function logs a WARNING and returns None so that the
    session continues without memory context (graceful degradation, req 6.6).

    Requirements: 6.2, 6.6
    """
    try:
        client = _memory_client()
        response = client.retrieve_memories(
            memoryId=MEMORY_STORE_ID,
            namespace=_namespace(tenant_id),
            maxResults=10,
        )
        summaries = [m["content"] for m in response.get("memories", [])]
        return "\n".join(summaries) if summaries else None
    except Exception as e:
        logger.warning(
            "AgentCore Memory 读取失败，降级继续 tenant_id=%s error=%s",
            tenant_id,
            e,
        )
        return None  # graceful degradation — session continues without memory


async def save_memory_on_session_end(tenant_id: str, session_summary: str) -> None:
    """
    Persist *session_summary* to the tenant's Memory namespace after a session.

    Runs a memory-poisoning safety check before writing. If the summary contains
    prompt-injection patterns, it is discarded and the failure is logged — the
    response is not affected (requirement 6.6).

    Requirements: 6.3, 6.6
    """
    # Safety check: reject summaries containing injection patterns
    try:
        from safety import check_memory_safety
        check_memory_safety(session_summary, tenant_id)
    except Exception as safety_err:
        logger.error(
            "Memory write blocked — safety violation tenant_id=%s error=%s",
            tenant_id,
            safety_err,
        )
        return  # Do not write poisoned content

    try:
        client = _memory_client()
        client.store_memory(
            memoryId=MEMORY_STORE_ID,
            namespace=_namespace(tenant_id),
            content=session_summary,
            metadata={"tenant_id": tenant_id, "timestamp": time.time()},
        )
        logger.info(
            "AgentCore Memory 写入成功 tenant_id=%s namespace=%s",
            tenant_id,
            _namespace(tenant_id),
        )
    except Exception as e:
        logger.error(
            "AgentCore Memory 写入失败 tenant_id=%s error=%s",
            tenant_id,
            e,
        )


async def clear_tenant_memory(tenant_id: str) -> bool:
    """
    Clear all memory entries for *tenant_id* (supports the ``/memory clear``
    command, requirement 6.7).

    Returns True on success, False on failure (failure is logged at ERROR).

    Requirements: 6.7
    """
    try:
        client = _memory_client()
        client.delete_memories(
            memoryId=MEMORY_STORE_ID,
            namespace=_namespace(tenant_id),
        )
        logger.info(
            "AgentCore Memory 已清除 tenant_id=%s namespace=%s",
            tenant_id,
            _namespace(tenant_id),
        )
        return True
    except Exception as e:
        logger.error(
            "AgentCore Memory 清除失败 tenant_id=%s error=%s",
            tenant_id,
            e,
        )
        return False