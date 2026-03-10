"""
OpenClaw AgentExecutor for the a2a-sdk.

Bridges the a2a-sdk's AgentExecutor interface to the OpenClaw gateway,
so that DefaultRequestHandler + A2AStarletteApplication handle all
A2A protocol details (JSON-RPC dispatch, SSE streaming, task store).
"""

import logging
import os
import sys
import time

import httpx

# Avoid local a2a.py shadowing the a2a SDK package
_local_dir = os.path.dirname(os.path.abspath(__file__))
_removed_paths = []
for _p in [_local_dir, '']:
    while _p in sys.path:
        sys.path.remove(_p)
        _removed_paths.append(_p)
if 'a2a' in sys.modules:
    del sys.modules['a2a']

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_task, new_agent_text_message

for _p in _removed_paths:
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _local_dir not in sys.path:
    sys.path.insert(0, _local_dir)

from permissions import read_permission_profile
from observability import log_agent_invocation
from safety import validate_message

logger = logging.getLogger(__name__)

OPENCLAW_URL = f"http://localhost:{os.environ.get('OPENCLAW_PORT', 18789)}"


def _build_system_prompt(tenant_id: str) -> str:
    try:
        profile = read_permission_profile(tenant_id)
        allowed = profile.get("tools", ["web_search"])
    except Exception:
        allowed = ["web_search"]
    blocked = [t for t in ["shell", "browser", "file", "file_write",
                           "code_execution", "install_skill", "load_extension", "eval"]
               if t not in allowed]
    lines = [f"Allowed tools for this session: {', '.join(allowed)}."]
    if blocked:
        lines.append(
            f"You MUST NOT use these tools: {', '.join(blocked)}. "
            "If the user requests an action that requires a blocked tool, "
            "explain that you don't have permission."
        )
    return " ".join(lines)


class OpenClawExecutor(AgentExecutor):
    """Invokes OpenClaw and pushes status/artifact events to the a2a-sdk EventQueue."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        await updater.start_work(
            new_agent_text_message("Processing your request…", task.context_id, task.id)
        )

        tenant_id = task.context_id or task.id
        try:
            response_text = await self._invoke_openclaw(tenant_id, query)
            await updater.complete(
                new_agent_text_message(response_text, task.context_id, task.id)
            )
        except Exception as e:
            logger.error("OpenClaw invocation failed task=%s: %s", task.id, e)
            await updater.failed(
                new_agent_text_message(str(e), task.context_id, task.id)
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task:
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            await updater.cancel(
                new_agent_text_message("Task canceled.", task.context_id, task.id)
            )

    async def _invoke_openclaw(self, tenant_id: str, message_text: str) -> str:
        system_prompt = _build_system_prompt(tenant_id)
        safe_message = validate_message(message_text)

        headers = {}
        gw_token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
        if gw_token:
            headers["Authorization"] = f"Bearer {gw_token}"

        start_ms = int(time.time() * 1000)
        async with httpx.AsyncClient(
            http2=True, timeout=httpx.Timeout(300.0, connect=10.0), headers=headers
        ) as client:
            resp = await client.post(
                f"{OPENCLAW_URL}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": safe_message},
                    ],
                    "user": f"a2a:{tenant_id}",
                },
            )
        duration_ms = int(time.time() * 1000) - start_ms

        if resp.status_code != 200:
            log_agent_invocation(tenant_id=tenant_id, tools_used=[], duration_ms=duration_ms, status="error")
            raise RuntimeError(f"OpenClaw HTTP {resp.status_code}: {resp.text[:500]}")

        log_agent_invocation(tenant_id=tenant_id, tools_used=[], duration_ms=duration_ms, status="success")
        result = resp.json()
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", str(result))
        return str(result)