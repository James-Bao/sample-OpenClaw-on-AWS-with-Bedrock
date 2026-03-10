"""
Agent Container HTTP/2 server.

Wraps openclaw as a subprocess. Serves all endpoints via an ASGI application
running on hypercorn (HTTP/2 + HTTP/1.1).

Endpoints:
  GET  /ping                      → Health check
  GET  /.well-known/agent-card.json    → A2A Agent Card (discovery)
  POST /invocations               → AgentCore chat completions
  POST /a2a                       → A2A JSON-RPC 2.0 (message/send, GetTask, CancelTask)
  POST /a2a/stream                → A2A SSE streaming (message/stream / SendStreamingMessage)

For each /invocations and /a2a request:
  A. Injects the tenant's allowed tools into the system prompt (soft enforcement).
  E. Audits the response for tool usage patterns (post-execution logging).
"""
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from typing import Optional

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from permissions import read_permission_profile
from observability import log_agent_invocation, log_permission_denied
from safety import validate_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# a2a-sdk native imports (used when A2A_ENABLED)
# NOTE: the local a2a.py shadows the a2a package, so we temporarily
# remove the local dir from sys.path when importing the SDK.
_a2a_sdk_available = False
_local_dir = os.path.dirname(os.path.abspath(__file__))
try:
    # The local a2a.py shadows the a2a SDK package.
    # Remove local dir AND '' (cwd) from sys.path temporarily.
    _removed_paths = []
    for p in [_local_dir, '']:
        while p in sys.path:
            sys.path.remove(p)
            _removed_paths.append(p)
    if 'a2a' in sys.modules:
        del sys.modules['a2a']
    from a2a.types import AgentCard, AgentSkill, AgentCapabilities
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    _a2a_sdk_available = True
except ImportError as e:
    logger.warning("a2a-sdk not available (%s) — A2A will use legacy handler", e)
finally:
    for p in _removed_paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    if _local_dir not in sys.path:
        sys.path.insert(0, _local_dir)

if _a2a_sdk_available:
    from openclaw_executor import OpenClawExecutor

# Legacy fallback
if not _a2a_sdk_available:
    try:
        from a2a import A2AHandler, build_agent_card
    except ImportError:
        pass

OPENCLAW_PORT = 18789
OPENCLAW_URL = f"http://localhost:{OPENCLAW_PORT}"
STARTUP_TIMEOUT = 30

# A2A protocol — enabled by default, set A2A_ENABLED=false to disable
A2A_ENABLED = os.environ.get("A2A_ENABLED", "true").lower() in ("true", "1", "yes")

# Regex to detect tool invocation patterns in openclaw responses.
_TOOL_PATTERN = re.compile(
    r'\b(shell|browser|file_write|code_execution|install_skill|load_extension|eval)\b',
    re.IGNORECASE,
)

# Shared httpx async client with HTTP/2 support (for calling OpenClaw)
_http_client: Optional[httpx.AsyncClient] = None


async def _get_http_client() -> httpx.AsyncClient:
    """Lazy-initialise a shared httpx AsyncClient with HTTP/2 support."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        headers = {}
        gw_token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
        if gw_token:
            headers["Authorization"] = f"Bearer {gw_token}"
        _http_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
            headers=headers,
        )
    return _http_client


# ---------------------------------------------------------------------------
# Permission & audit helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(tenant_id: str) -> str:
    """Plan A: build a system prompt that constrains openclaw to allowed tools."""
    try:
        profile = read_permission_profile(tenant_id)
        allowed = profile.get("tools", ["web_search"])
        blocked = [t for t in ["shell", "browser", "file", "file_write", "code_execution",
                                "install_skill", "load_extension", "eval"]
                   if t not in allowed]
    except Exception:
        allowed = ["web_search"]
        blocked = ["shell", "browser", "file", "file_write", "code_execution",
                   "install_skill", "load_extension", "eval"]

    lines = [
        f"Allowed tools for this session: {', '.join(allowed)}.",
    ]
    if blocked:
        lines.append(
            f"You MUST NOT use these tools: {', '.join(blocked)}. "
            "If the user requests an action that requires a blocked tool, "
            "explain that you don't have permission and they should contact their administrator."
        )
    return " ".join(lines)


def _audit_response(tenant_id: str, response_text: str, allowed_tools: list) -> None:
    """Plan E: scan response for tool usage and log any violations."""
    matches = _TOOL_PATTERN.findall(response_text)
    if not matches:
        return
    for tool in set(t.lower() for t in matches):
        if tool not in allowed_tools:
            log_permission_denied(
                tenant_id=tenant_id,
                tool_name=tool,
                cedar_decision="RESPONSE_AUDIT",
                request_id=None,
            )
            logger.warning(
                "AUDIT: blocked tool '%s' detected in response tenant_id=%s",
                tool, tenant_id,
            )


# ---------------------------------------------------------------------------
# OpenClaw subprocess management
# ---------------------------------------------------------------------------

def start_openclaw() -> subprocess.Popen:
    # Apply config via `openclaw config set` (--config flag removed in v2026.3+)
    config_src = "/app/openclaw.json"
    with open(config_src) as f:
        config = json.loads(f.read()
            .replace("${AWS_REGION}", os.environ.get("AWS_REGION", "us-east-1"))
            .replace("${BEDROCK_MODEL_ID}",
                     os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0")))

    config_map = {
        "gateway.http.endpoints.chatCompletions.enabled": "true",
    }
    for key, val in config_map.items():
        subprocess.run(["openclaw", "config", "set", key, val],
                       capture_output=True, timeout=10)

    env = os.environ.copy()
    env["OPENCLAW_SKIP_ONBOARDING"] = "1"
    proc = subprocess.Popen(
        ["openclaw", "gateway", "--port", str(OPENCLAW_PORT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    threading.Thread(
        target=lambda: [logger.info("[openclaw] %s", l.decode().rstrip()) for l in proc.stdout],
        daemon=True,
    ).start()
    return proc


async def wait_for_openclaw(timeout: int = STARTUP_TIMEOUT) -> None:
    """Wait for the OpenClaw subprocess to become ready, using httpx with HTTP/2."""
    deadline = time.time() + timeout
    client = await _get_http_client()
    while time.time() < deadline:
        try:
            r = await client.post(
                f"{OPENCLAW_URL}/v1/chat/completions",
                json={"model": "probe", "messages": [], "user": "healthcheck"},
                timeout=2.0,
            )
            if r.status_code < 500:
                logger.info("openclaw ready (status=%d, protocol=%s)", r.status_code, r.http_version)
                return
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass
        await asyncio.sleep(1)
    logger.error("openclaw did not become ready within %d seconds", timeout)
    sys.exit(1)


async def _check_openclaw_connectivity() -> bool:
    """Verify that the OpenClaw server is reachable. Used before processing A2A tasks."""
    client = await _get_http_client()
    try:
        r = await client.post(
            f"{OPENCLAW_URL}/v1/chat/completions",
            json={"model": "probe", "messages": [], "user": "connectivity-check"},
            timeout=5.0,
        )
        return r.status_code < 500
    except Exception as e:
        logger.warning("OpenClaw connectivity check failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# A2A → OpenClaw bridge (uses httpx with HTTP/2)
# ---------------------------------------------------------------------------

async def _invoke_for_a2a_async(task_id: str, message_text: str, session_id: str = None) -> str:
    """Invoke OpenClaw on behalf of an A2A task and return the response text.

    Uses httpx AsyncClient with HTTP/2 support to communicate with the
    OpenClaw subprocess, ensuring proper connectivity.
    """
    tenant_id = session_id or task_id
    system_prompt = _build_system_prompt(tenant_id)
    safe_message = validate_message(message_text)

    # Verify OpenClaw is reachable before processing
    if not await _check_openclaw_connectivity():
        raise RuntimeError("OpenClaw server is not reachable — cannot process A2A task")

    client = await _get_http_client()
    start_ms = int(time.time() * 1000)
    try:
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
            timeout=300.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenClaw returned HTTP {resp.status_code}: {resp.text[:500]}"
            )
        result = resp.json()
        duration_ms = int(time.time() * 1000) - start_ms

        # Audit the response
        response_text = json.dumps(result)
        try:
            profile = read_permission_profile(tenant_id)
            allowed = profile.get("tools", ["web_search"])
        except Exception:
            allowed = ["web_search"]
        _audit_response(tenant_id, response_text, allowed)

        log_agent_invocation(
            tenant_id=tenant_id, tools_used=[], duration_ms=duration_ms, status="success"
        )

        # Extract the assistant message text from the chat completions response
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", json.dumps(result))
        return json.dumps(result)

    except httpx.ConnectError as e:
        duration_ms = int(time.time() * 1000) - start_ms
        log_agent_invocation(
            tenant_id=tenant_id, tools_used=[], duration_ms=duration_ms, status="error"
        )
        raise RuntimeError(
            f"Cannot reach OpenClaw server at {OPENCLAW_URL} — "
            f"ensure the openclaw subprocess is running: {e}"
        ) from e
    except Exception as e:
        duration_ms = int(time.time() * 1000) - start_ms
        log_agent_invocation(
            tenant_id=tenant_id, tools_used=[], duration_ms=duration_ms, status="error"
        )
        raise RuntimeError(f"OpenClaw invocation failed: {e}") from e


def _invoke_for_a2a(task_id: str, message_text: str, session_id: str = None) -> str:
    """Synchronous wrapper for the async A2A invocation.

    The A2AHandler calls this synchronously; we bridge to the async httpx client
    by running the coroutine in a new thread with its own event loop.
    """
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_invoke_for_a2a_sync_in_thread, task_id, message_text, session_id)
        return future.result(timeout=310)


def _invoke_for_a2a_sync_in_thread(task_id: str, message_text: str, session_id: str = None) -> str:
    """Run the async invocation in a fresh event loop on a worker thread.

    Creates a dedicated httpx client for this thread's event loop to avoid
    cross-loop issues with the shared _http_client.
    """
    async def _run():
        headers = {}
        gw_token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
        if gw_token:
            headers["Authorization"] = f"Bearer {gw_token}"
        async with httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(300.0, connect=10.0),
            headers=headers,
        ) as client:
            tenant_id = session_id or task_id
            system_prompt = _build_system_prompt(tenant_id)
            safe_message = validate_message(message_text)

            start_ms = int(time.time() * 1000)
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
                timeout=300.0,

            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenClaw returned HTTP {resp.status_code}: {resp.text[:500]}"
                )
            result = resp.json()
            duration_ms = int(time.time() * 1000) - start_ms

            response_text = json.dumps(result)
            try:
                profile = read_permission_profile(tenant_id)
                allowed = profile.get("tools", ["web_search"])
            except Exception:
                allowed = ["web_search"]
            _audit_response(tenant_id, response_text, allowed)

            log_agent_invocation(
                tenant_id=tenant_id, tools_used=[], duration_ms=duration_ms, status="success"
            )

            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", json.dumps(result))
            return json.dumps(result)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# A2A application (a2a-sdk native or legacy fallback)
# ---------------------------------------------------------------------------

_a2a_starlette_app = None  # SDK Starlette ASGI app (lazy init)
_a2a_handler = None         # Legacy handler (fallback)


def _build_a2a_sdk_app():
    """Build the a2a-sdk Starlette ASGI app."""
    global _a2a_starlette_app
    if _a2a_starlette_app is not None:
        return _a2a_starlette_app

    from a2a.server.apps.jsonrpc.starlette_app import AGENT_CARD_WELL_KNOWN_PATH
    import json as _json

    port = int(os.environ.get("PORT", 8080))
    base_url = os.environ.get("A2A_BASE_URL", f"http://localhost:{port}")

    # Load agent card from agent-card.json
    app_dir = os.path.dirname(os.path.abspath(__file__))
    agent_card_path = os.path.join(app_dir, "agent-card.json")

    card_name = "OpenClaw Agent"
    card_description = (
        "A long-running task agent powered by OpenClaw on AWS with Bedrock. "
        "Executes complex, multi-step tasks asynchronously."
    )
    card_version = "1.0.0"
    card_capabilities = AgentCapabilities(streaming=True, pushNotifications=False)
    card_input_modes = ["text"]
    card_output_modes = ["text"]
    skills = []

    # Try loading from agent-card.json first (authoritative source)
    try:
        with open(agent_card_path) as f:
            agent_card_data = _json.load(f)
        card_name = agent_card_data.get("name", card_name)
        card_description = agent_card_data.get("description", card_description)
        card_version = agent_card_data.get("version", card_version)
        card_input_modes = agent_card_data.get("defaultInputModes", card_input_modes)
        card_output_modes = agent_card_data.get("defaultOutputModes", card_output_modes)
        caps = agent_card_data.get("capabilities", {})
        card_capabilities = AgentCapabilities(
            streaming=caps.get("streaming", True),
            pushNotifications=caps.get("pushNotifications", False),
        )
        for s in agent_card_data.get("skills", []):
            skills.append(AgentSkill(
                id=s["id"], name=s["name"],
                description=s.get("description", ""),
                tags=s.get("tags", []),
                examples=s.get("examples", []),
            ))
        logger.info("Loaded agent card from %s (%d skills)", agent_card_path, len(skills))
    except Exception as e:
        logger.warning("Could not load agent-card.json (%s), using default skill", e)
        skills = [AgentSkill(
            id="task-executor", name="Task Executor",
            description="Execute complex tasks via OpenClaw on AWS with Bedrock",
            tags=["long-running", "async"], examples=[],
        )]

    card = AgentCard(
        name=card_name,
        description=card_description,
        url=base_url,
        version=card_version,
        skills=skills,
        capabilities=card_capabilities,
        defaultInputModes=card_input_modes,
        defaultOutputModes=card_output_modes,
    )

    executor = OpenClawExecutor()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    a2a_app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    _a2a_starlette_app = a2a_app.build()
    logger.info("a2a-sdk Starlette app built (streaming + JSON-RPC)")
    return _a2a_starlette_app


def _get_legacy_a2a_handler():
    """Fallback: use our custom A2AHandler if SDK is not available."""
    global _a2a_handler
    if _a2a_handler is None:
        from a2a import A2AHandler, build_agent_card  # noqa: F811
        _a2a_handler = A2AHandler(invoke_fn=_invoke_for_a2a)
    return _a2a_handler


# ---------------------------------------------------------------------------
# ASGI Application (HTTP/2 + HTTP/1.1 via hypercorn)
# ---------------------------------------------------------------------------

async def _read_body(receive) -> bytes:
    """Read the full request body from an ASGI receive channel."""
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break
    return body


async def _send_json(send, status: int, body: dict, headers: list = None) -> None:
    """Send a JSON response via ASGI."""
    data = json.dumps(body).encode()
    response_headers = [
        [b"content-type", b"application/json"],
        [b"content-length", str(len(data)).encode()],
    ]
    if headers:
        response_headers.extend(headers)

    await send({
        "type": "http.response.start",
        "status": status,
        "headers": response_headers,
    })
    await send({
        "type": "http.response.body",
        "body": data,
    })


async def app(scope, receive, send):
    """ASGI application supporting HTTP/2 and HTTP/1.1.

    Routes:
      GET  /ping                         → health check
      *    /.well-known/agent-card.json  → A2A Agent Card (via SDK)
      POST /a2a (message/send, stream)   → A2A JSON-RPC 2.0 (via SDK)
      POST /invocations                  → AgentCore chat completions
    """
    if scope["type"] != "http":
        return

    method = scope["method"]
    path = scope["path"]
    http_version = scope.get("http_version", "unknown")
    logger.debug("Request: %s %s (HTTP/%s)", method, path, http_version)

    # --- Health check (always available) ---
    if method == "GET" and path == "/ping":
        await _send_json(send, 200, {"status": "ok", "http_version": http_version})
        return

    # --- A2A paths → delegate to a2a-sdk Starlette app ---
    if A2A_ENABLED:
        # Paths the SDK handles: /.well-known/agent-card.json and / (JSON-RPC POST)
        # We also accept /a2a prefix and /a2a/.well-known/agent-card.json
        a2a_paths = (
            "/.well-known/agent-card.json", "/.well-known/agent.json",
            "/a2a/.well-known/agent-card.json", "/a2a/.well-known/agent.json",
            "/", "/a2a",
            "/stream", "/a2a/stream",
        )
        if path in a2a_paths:
            if _a2a_sdk_available:
                sdk_app = _build_a2a_sdk_app()
                # Rewrite /a2a → / and /a2a/stream → / so the SDK sees its expected paths
                # The SDK handles both message/send and message/stream on /
                inner_scope = dict(scope)
                if path in ("/a2a", "/a2a/stream", "/stream"):
                    inner_scope["path"] = "/"
                elif path in ("/a2a/.well-known/agent-card.json", "/a2a/.well-known/agent.json"):
                    inner_scope["path"] = "/.well-known/agent-card.json"
                await sdk_app(inner_scope, receive, send)
                return
            else:
                # Legacy fallback — serve agent-card.json directly
                if method == "GET" and "well-known" in path:
                    app_dir = os.path.dirname(os.path.abspath(__file__))
                    agent_card_path = os.path.join(app_dir, "agent-card.json")
                    try:
                        with open(agent_card_path) as f:
                            card = json.load(f)
                        # Ensure the URL reflects the current port
                        port = int(os.environ.get("PORT", 8080))
                        card["url"] = os.environ.get("A2A_BASE_URL", f"http://localhost:{port}")
                    except Exception:
                        # Final fallback if agent-card.json is missing
                        from a2a import build_agent_card
                        port = int(os.environ.get("PORT", 8080))
                        card = build_agent_card(port=port)
                    await _send_json(send, 200, card)
                    return
                elif method == "POST" and path in ("/", "/a2a"):
                    body = await _read_body(receive)
                    handler = _get_legacy_a2a_handler()
                    result = handler.handle_request(body)
                    await _send_json(send, 200, result)
                    return
                elif method == "POST" and path in ("/stream", "/a2a/stream"):
                    body = await _read_body(receive)
                    handler = _get_legacy_a2a_handler()
                    await send({"type": "http.response.start", "status": 200, "headers": [
                        [b"content-type", b"text/event-stream"],
                        [b"cache-control", b"no-cache"],
                    ]})
                    try:
                        for event in handler.handle_streaming_request(body):
                            await send({"type": "http.response.body", "body": event.encode(), "more_body": True})
                    except Exception as e:
                        logger.error("A2A streaming error: %s", e)
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                    return

    # --- POST /invocations → AgentCore ---
    if method == "POST" and path == "/invocations":
        body = await _read_body(receive)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            await _send_json(send, 400, {"error": "invalid json"})
            return

        tenant_id = payload.get("sessionId") or payload.get("tenant_id") or "unknown"
        message = validate_message(payload.get("message", ""))
        session_key = f"agentcore:{tenant_id}"

        system_prompt = _build_system_prompt(tenant_id)

        client = await _get_http_client()
        start_ms = int(time.time() * 1000)
        try:
            resp = await client.post(
                f"{OPENCLAW_URL}/v1/chat/completions",
                json={
                    "model": payload.get("model", "default"),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message},
                    ],
                    "user": session_key,
                },
                timeout=300.0,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenClaw returned HTTP {resp.status_code}: {resp.text[:500]}"
                )
            result = resp.json()
            duration_ms = int(time.time() * 1000) - start_ms

            response_text = json.dumps(result)
            try:
                profile = read_permission_profile(tenant_id)
                allowed = profile.get("tools", ["web_search"])
            except Exception:
                allowed = ["web_search"]
            _audit_response(tenant_id, response_text, allowed)

            log_agent_invocation(
                tenant_id=tenant_id, tools_used=[], duration_ms=duration_ms, status="success"
            )
            await _send_json(send, 200, result)

        except Exception as e:
            duration_ms = int(time.time() * 1000) - start_ms
            log_agent_invocation(
                tenant_id=tenant_id, tools_used=[], duration_ms=duration_ms, status="error"
            )
            logger.error("openclaw invocation failed tenant_id=%s error=%s", tenant_id, e)
            await _send_json(send, 500, {"error": str(e)})
        return

    # --- Fallback ---
    if method == "GET":
        await _send_json(send, 404, {"error": "not found"})
    elif method == "POST":
        await _send_json(send, 404, {"error": "not found"})
    else:
        await _send_json(send, 405, {"error": "method not allowed"})


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def main():
    from hypercorn.config import Config
    from hypercorn.asyncio import serve

    proc = start_openclaw()

    port = int(os.environ.get("PORT", 8080))

    config = Config()
    config.h2_max_concurrent_streams = 128
    config.h2_max_header_list_size = 65536
    config.h11_max_incomplete_size = 16 * 1024

    # TLS for HTTP/2 with ALPN negotiation (h2 + http/1.1)
    cert_dir = os.path.join(os.path.dirname(__file__), "certs")
    certfile = os.path.join(cert_dir, "cert.pem")
    keyfile = os.path.join(cert_dir, "key.pem")
    if os.path.exists(certfile) and os.path.exists(keyfile):
        config.certfile = certfile
        config.keyfile = keyfile
        config.bind = [f"0.0.0.0:{port}"]
        logger.info("TLS enabled — h2 (ALPN) on port %d", port)
    else:
        config.bind = [f"0.0.0.0:{port}"]
        logger.info("No TLS certs found — h2c (cleartext) on port %d", port)

    async def startup():
        """Wait for OpenClaw, then serve the ASGI app."""
        await wait_for_openclaw(STARTUP_TIMEOUT)

        logger.info("HTTP/2 server listening on port %d (h2c + http/1.1)", port)
        if A2A_ENABLED:
            logger.info(
                "A2A protocol ENABLED (HTTP/2) — Agent Card at http://0.0.0.0:%d/.well-known/agent-card.json, "
                "JSON-RPC at /a2a, streaming at /a2a/stream",
                port,
            )
        else:
            logger.info("A2A protocol DISABLED (set A2A_ENABLED=true to enable)")

        await serve(app, config)

    try:
        asyncio.run(startup())
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up httpx client
        if _http_client and not _http_client.is_closed:
            asyncio.run(_http_client.aclose())
        proc.terminate()


if __name__ == "__main__":
    main()
