"""Hermes client for direct OpenCode free-model inference.

The client reproduces OpenCode's free-model request envelope while keeping the
agent loop and every tool execution in Hermes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import secrets
import string
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from hermes_constants import get_hermes_home  # type: ignore[import-not-found]

#: The free tier's path. ``models.opencode.ai/api.json`` publishes
#: ``https://opencode.ai/inference/openai/v1`` for the same provider — that is the AI-SDK base, a
#: different path that also serves a free-tier call today; the pinned one is the path verified end to
#: end against the live relay, so it stays pinned rather than re-derived from a catalogue that
#: describes another client.
DIRECT_URL = "https://opencode.ai/inference/v1/chat/completions"
RESPONSES_URL = "https://opencode.ai/inference/v1/responses"
MODELS_URL = "https://opencode.ai/inference/v1/models"
MODELS_DEV_URL = "https://models.dev/api.json"
LOGICAL_BASE_URL = DIRECT_URL
#: The version this client's User-Agent claims. Pinned only as a FLOOR and a fallback: the value
#: actually sent is derived (``client_version``), because the relay raises the minimum it serves
#: without notice — a pinned number is a latent break, and re-deriving it is the fix when it ages.
OPENCODE_VERSION = "2.0.5"
#: The official client, asked for its own version. It is the client the relay is built to serve and it
#: updates itself, so its number tracks the relay's floor.
OPENCODE_CLI = "opencode"
#: How long a derived version is trusted before the installed client is asked again.
VERSION_CHECK_TTL = 6 * 3600.0
#: The relay accepts only a bare "opencode/<semver>" User-Agent; the official client's own
#: "opencode/latest/<ver>/cli" form is refused with FreeTierError, so we copy its *number*, never its
#: string. This constant is the last-resort fallback when nothing can be derived.
OPENCODE_USER_AGENT = f"opencode/{OPENCODE_VERSION}"
DEFAULT_MODEL = "big-pickle"
FALLBACK_MODELS = (
    DEFAULT_MODEL,
    "cat-jam-free",
    "deepseek-v4.1-flash-free",
    "glm-5.3-flash-free",
    "ll-alpha-free",
    "longcat-2.5-preview-free",
    "mimo-v2.6-flash-free",
    "muse-spark-1.3-contributor-free",
    "nemotron-3.5-lightning-free",
    "nemotron-3.5-super-vl-free",
    "space-bunny-free",
)
RESPONSES_MODELS = frozenset(
    {
        "muse-spark-1.3-contributor-free",
    }
)
COMPAT_TOOL_NAMES = (
    "bash",
    "edit",
    "glob",
    "grep",
    "read",
    "skill",
    "task",
    "todowrite",
    "webfetch",
    "websearch",
    "write",
)
COMPAT_TOOL_TARGETS = {
    "bash": "terminal",
    "edit": "patch",
    "glob": "search_files",
    "grep": "search_files",
    "read": "read_file",
    "skill": "skill_view",
    "task": "delegate_task",
    "todowrite": "todo_list",
    "webfetch": "web_extract",
    "websearch": "web_search",
    "write": "write_file",
}
COMPAT_TOOL_DESCRIPTIONS = {
    "bash": "Execute a shell command in the local workspace.",
    "edit": "Replace exact text in a local file.",
    "glob": "Find local files by glob pattern.",
    "grep": "Search local file contents with a regular expression.",
    "read": "Read a local file.",
    "skill": "Load a Hermes skill by name.",
    "task": "Delegate a task to a Hermes subagent.",
    "todowrite": "Replace the current Hermes task list.",
    "webfetch": "Extract content from a web page.",
    "websearch": "Search the web.",
    "write": "Write a local file.",
}
COMPAT_TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "bash": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "minimum": 1},
            "workdir": {"type": "string"},
        },
        "required": ["command"],
    },
    "edit": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "oldString": {"type": "string"},
            "newString": {"type": "string"},
            "replaceAll": {"type": "boolean"},
        },
        "required": ["filePath", "oldString", "newString"],
    },
    "glob": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["pattern"],
    },
    "grep": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "include": {"type": "string"},
        },
        "required": ["pattern"],
    },
    "read": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["filePath"],
    },
    "skill": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    "task": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "prompt": {"type": "string"},
            "subagent_type": {"type": "string"},
            "task_id": {"type": "string"},
            "command": {"type": "string"},
            "background": {"type": "boolean"},
        },
        "required": ["description", "prompt", "subagent_type"],
    },
    "todowrite": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": [
                                "pending",
                                "in_progress",
                                "completed",
                                "cancelled",
                            ],
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                    "required": ["content", "status", "priority"],
                },
            }
        },
        "required": ["todos"],
    },
    "webfetch": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "format": {"type": "string", "enum": ["text", "markdown", "html"]},
            "timeout": {"type": "integer", "minimum": 1},
        },
        "required": ["url"],
    },
    "websearch": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "numResults": {"type": "integer", "minimum": 1, "maximum": 100},
            "livecrawl": {"type": "string"},
            "type": {"type": "string"},
            "contextMaxCharacters": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
    },
    "write": {
        "type": "object",
        "properties": {
            "filePath": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["filePath", "content"],
    },
}
_BASE62 = string.digits + string.ascii_letters
_SESSION_LOCK = threading.Lock()
_SESSION_COUNTER = 0
_MODEL_SNAPSHOT: tuple[str, ...] | None = None
_MODEL_SNAPSHOT_LOCK = threading.Lock()
#: The version resolved for this process (see ``client_version``); None until first resolved.
_VERSION: str | None = None
#: The lock guarding the handshake file and the version resolved from it (one file, one writer).
_HANDSHAKE_LOCK = threading.Lock()
#: Per-model ``limit`` metadata for the free models (models.dev rows, keyed by model id).
_CATALOG_LIMITS: dict[str, dict[str, int]] = {}


class OpenCodeError(RuntimeError):
    status_code: int | None = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _status_error(code: int, body: str, context: str) -> OpenCodeError:
    exc = OpenCodeError(f"{context} failed with HTTP {code}: {body[:500]}")
    exc.status_code = code
    return exc


#: This provider's own refusal log, beside the model cache. A refusal is decided by bytes a host
#: process built, and a host that has been running since before a fix keeps the module it imported —
#: so the refusal has to stay readable from disk afterwards, with no way to reach into that process.
REFUSAL_LOG = "refusals.jsonl"
#: How many refusals are kept; the file is rewritten with the newest few, so it cannot grow.
REFUSAL_LOG_LINES = 20


def _request_shape(request: urllib.request.Request) -> dict[str, Any]:
    """What was sent, as shape rather than content: the tick in a wire dispute, without the text."""
    raw = getattr(request, "data", None)
    encoded = bytes(raw) if isinstance(raw, (bytes, bytearray)) else b""
    try:
        body = json.loads(encoded.decode("utf-8", "replace") or "{}")
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    raw_tools = body.get("tools")
    tools = raw_tools if isinstance(raw_tools, list) else []
    return {
        "url": request.full_url,
        "headers": dict(request.headers),
        "body": {
            "model": body.get("model"),
            "keys": sorted(body),
            "tools": [
                str((item.get("function") or {}).get("name") or "")
                for item in tools
                if isinstance(item, dict)
            ],
            "messages": len(body.get("messages") or []),
            "max_tokens": body.get("max_tokens"),
            "reasoning_effort": body.get("reasoning_effort"),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        },
    }


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """A cross-process lock around a read-modify-write of this provider's own files.

    Two writers meet on this cache in practice — the host process building requests and a probe
    process replaying them — and an unlocked read-modify-write interleaves: one reader keeps a
    prefix while the other replaces the file, leaving a line cut in half, and the evidence is lost
    exactly when it is wanted. ``fcntl`` is POSIX-only, so where it is absent the write stays
    unlocked rather than the call failing.
    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _connection_facts(response: Any) -> dict[str, Any]:
    """What carried the refusal, dug out of its own response: the peer's view of the connection.

    The one thing a request's shape cannot say is *which* connection was refused, and the gate sees
    the connection we make. best-effort and shape-only; every probe here is guarded because the
    response may already be closed.

    urllib hands an ``HTTPError`` the ``HTTPResponse`` itself, so the socket sits four hops down:
    ``HTTPError.fp`` (the response) -> ``.fp`` (a ``BufferedReader``) -> ``.raw`` (``SocketIO``) ->
    ``._sock`` (the ``SSLSocket``). A chain that does not resolve answers nothing rather than
    raising.
    """
    node: Any = response
    for attribute in ("fp", "fp", "raw", "_sock"):
        node = getattr(node, attribute, None)
        if node is None:
            return {}
    facts: dict[str, Any] = {}
    for name, reader in (("local", "getsockname"), ("peer", "getpeername")):
        with contextlib.suppress(Exception):
            address = getattr(node, reader)()
            facts[name] = f"{address[0]}:{address[1]}"
    with contextlib.suppress(Exception):
        if callable(node.version):
            facts["tls"] = node.version()
    with contextlib.suppress(Exception):
        cipher = node.cipher()
        if cipher:
            facts["cipher"] = str(cipher[0])
    with contextlib.suppress(Exception):
        if callable(node.selected_alpn_protocol):
            facts["alpn"] = node.selected_alpn_protocol()
    for name in ("session_reused", "server_hostname"):
        value = getattr(node, name, None)
        if isinstance(value, (str, bool)):
            facts[name] = value
    return facts


def _refusal_evidence(exc: urllib.error.HTTPError) -> dict[str, Any]:
    """The refusal's own side of the exchange: its headers, and the connection that carried them."""
    evidence: dict[str, Any] = {}
    headers = getattr(exc, "headers", None)
    if headers is not None:
        with contextlib.suppress(Exception):
            evidence["response_headers"] = {str(key): str(value) for key, value in headers.items()}
    facts = _connection_facts(exc)
    if facts:
        evidence["connection"] = facts
    return evidence


def _probe_identical(
    request: urllib.request.Request, timeout: float
) -> tuple[int | None, Any, float]:
    """Re-open the *identical* bytes once, on a connection of its own, after a free-tier refusal.

    The verdict is not on the bytes: a refused turn's own envelope streams ``200`` from a fresh
    process in the same minute (measured live, 2026-10-17). So the bit that separates "another
    attempt fixes it" from "restart the host" is whether this same process, on a brand-new
    connection, is served — and it is recorded either way. A served response *is* the turn's own,
    so the probe is a fix and a diagnostic at once; it is never spent on a request that is not
    already refused, and never on one whose envelope the recovery just rebuilt.

    ``None`` as the status means the connection itself failed.
    """
    started = time.monotonic()
    try:
        response = _urlopen(request, timeout)
    except urllib.error.HTTPError as exc:
        with contextlib.suppress(Exception):
            exc.read()
        return exc.code, None, time.monotonic() - started
    except (urllib.error.URLError, OSError):
        return None, None, time.monotonic() - started
    return 200, response, time.monotonic() - started


def _record_refusal(
    status: int,
    response: str,
    request: urllib.request.Request | None = None,
    evidence: dict[str, Any] | None = None,
    probe: tuple[int | None, Any, float] | None = None,
) -> None:
    """Append a refused call to this provider's own log, newest last, keeping the last few.

    *evidence* is the refusal's own side of the exchange and *probe* the verdict on re-opening the
    identical bytes on a fresh connection; both are shape, never the conversation. The write is
    locked, because a host and a probe process write this file at the same time.

    Never raises: a diagnostic must not be able to turn one refusal into two.
    """
    try:
        path = _model_cache_path().with_name(REFUSAL_LOG)
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "status": status,
            "response": str(response or "")[:2000],
        }
        if request is not None:
            entry.update(_request_shape(request))
        if evidence:
            entry.update(evidence)
        if probe is not None:
            entry["probe"] = {"status": probe[0], "seconds": round(probe[2], 3)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(path):
            previous = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            lines = [*previous[-(REFUSAL_LOG_LINES - 1) :], json.dumps(entry)]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass


def _urlopen(request: urllib.request.Request, timeout: float):
    return _OPENER.open(request, timeout=timeout)


INFERENCE_CONTEXT = "OpenCode free-model inference"

#: The relay's free-tier refusal, as the ``type`` of a 403 body. The gate decides whether a request
#: looks like the official client's; one this client already builds inside the documented
#: vocabulary (bare ``opencode/<semver>`` agent, ``ses_`` session id, the advertised tool set) is
#: still refused now and then, and the identical bytes are served on a later attempt — verified
#: live by replaying a refused turn's own envelope, and by mutating one field at a time to confirm
#: every field the gate reads is one this client sets. So it is the gate's transient verdict, not
#: the dead credential a generic 403 verdict assumes. Measured 2026-10-17: a refused turn's own
#: bytes stream 200 from a fresh process in the same minute, so the refusal is re-opened once on a
#: connection of its own (``_probe_identical``) and that verdict is recorded either way.
FREE_TIER_REFUSAL = "FreeTierError"


def classify_api_error(
    error: Any = None,
    *,
    status_code: Any = None,
    error_code: Any = None,
    message: Any = None,
    body: Any = None,
    model: Any = None,
    **context: Any,
) -> dict[str, Any] | None:
    """Hermes' ``ProviderProfile.classify_api_error`` hook for this provider's own failures.

    A free-tier refusal becomes a retryable verdict, so one transient refusal no longer ends the
    turn; every other failure returns ``None`` and keeps the generic pipeline's verdict. ``unknown``
    is the taxonomy's retry-with-backoff reason: the refusal names no cause this client can act on,
    and retrying the same envelope is what recovers.
    """
    try:
        code = int(status_code)
    except (TypeError, ValueError):
        return None
    if code != 403:
        return None
    text = str(message or "")
    if isinstance(body, str):
        text = f"{text}\n{body}"
    elif body is not None:
        try:
            text = f"{text}\n{json.dumps(body)}"
        except (TypeError, ValueError):
            pass
    # Case-insensitive: the host hands the classifier a lowercased message (``_Ctx.msg``), so a
    # case-sensitive match here silently declines and the generic 403 verdict stands.
    if FREE_TIER_REFUSAL.lower() not in text.lower():
        return None
    return {"reason": "unknown", "retryable": True, "should_fallback": True}


#: Environment variable naming a file that every inference envelope is appended to. A relay refusal
#: (``403 FreeTierError``) is decided by the exact bytes of the envelope, and the request dump Hermes
#: writes on a failed call records its own ``api_kwargs`` — *before* this client translates them — so
#: it cannot settle a dispute about the wire. Set this to capture what was actually sent.
ENVELOPE_CAPTURE_ENV = "HERMES_OC_DEBUG"

#: Name of the marker file, inside this provider's cache directory, that arms a capture from disk.
#: A long-lived host (the gateway) imports this module once and re-imports nothing, so its
#: environment can never be changed afterwards and the env var alone cannot arm a capture there;
#: a file it can see can. Its content is the destination path, or empty for ``envelopes.jsonl``.
CAPTURE_ARM_FILE = "capture"


def _armed_capture_path() -> str | None:
    """The capture destination named by the cache-dir marker file, or None when not armed."""
    try:
        marker = _model_cache_path().with_name(CAPTURE_ARM_FILE)
        destination = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return destination or str(marker.with_name("envelopes.jsonl"))


def _capture_envelope(request: urllib.request.Request) -> None:
    """Append *request* as one JSON line to the armed capture file.

    Called from the single place that builds a request, so the capture and the wire can never drift.
    Never raises: a diagnostic must not be able to fail an inference call.
    """
    path = os.environ.get(ENVELOPE_CAPTURE_ENV) or _armed_capture_path()
    if not path:
        return
    try:
        raw = getattr(request, "data", None)
        body = bytes(raw).decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else None
        line = json.dumps(
            {"url": request.full_url, "headers": dict(request.headers), "body": body}
        )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except (OSError, ValueError):
        pass


def _inference_request(url: str, body: dict[str, Any], session: str) -> urllib.request.Request:
    """The envelope every free-tier call carries."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer public",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": user_agent(),
            "x-opencode-session": session,
        },
    )
    _capture_envelope(request)
    return request


def _sse_events(
    request: urllib.request.Request,
    timeout: float,
    *,
    owner: "OpenCodeClient | None" = None,
    recover: "Callable[[int, str, urllib.request.Request], urllib.request.Request | None] | None" = None,
) -> Iterator[dict[str, Any]]:
    """Open *request* and yield its ``data:`` payloads until ``[DONE]``.

    With *owner*, the live response is registered with that client so its ``close()`` can abort this
    read from another thread — which is how an interrupt or a restart releases the request.

    With *recover*, a refused call is offered to it once: the relay's checks and its request
    vocabulary move without notice, and a refusal *they* caused is a retry with a re-derived
    envelope rather than the end of the turn. Whatever it returns is sent instead; None (or a second
    refusal) raises, and every refusal is recorded either way.
    """
    retried = False
    while True:
        try:
            response = _urlopen(request, timeout)
        except urllib.error.HTTPError as exc:
            # Before the body is read: reading it closes the response, and the connection facts are
            # only reachable through the response it holds.
            evidence = _refusal_evidence(exc)
            body_text = exc.read().decode("utf-8", "replace")
            replacement = (
                recover(exc.code, body_text, request) if recover is not None and not retried else None
            )
            retried = retried or recover is not None
            free_tier = exc.code == 403 and FREE_TIER_REFUSAL.lower() in body_text.lower()
            probe = (
                _probe_identical(request, timeout)
                if free_tier and replacement is None
                else None
            )
            _record_refusal(exc.code, body_text, request, evidence=evidence, probe=probe)
            if replacement is not None:
                request = replacement
                continue
            if probe is not None and probe[0] == 200 and probe[1] is not None:
                response = probe[1]
            else:
                raise _status_error(exc.code, body_text, INFERENCE_CONTEXT) from exc
        except urllib.error.URLError as exc:
            raise OpenCodeError(f"{INFERENCE_CONTEXT} failed: {exc.reason}") from exc
        tracked = owner._track_inflight(response) if owner is not None else contextlib.nullcontext()
        with response, tracked:
            try:
                for raw_line in response:
                    if owner is not None and owner.is_closed:
                        raise OpenCodeError("OpenCode client was closed mid-request.")
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    if payload:
                        yield json.loads(payload)
            except (OSError, ValueError) as exc:
                # ``close()`` aborts a blocked read from a stranger thread, so the reader sees a closed
                # file: that is an abort, not a provider failure.
                if owner is not None and owner.is_closed:
                    raise OpenCodeError("OpenCode client was closed mid-request.") from exc
                raise
        return


def _retry_with(url: str, body: dict[str, Any]) -> urllib.request.Request:
    """The rebuilt request a recovery sends: the same call, with a session id of its own."""
    return _inference_request(url, body, _session_id())


def _refusal_recovery(
    url: str, body: dict[str, Any], model: str
) -> "Callable[[int, str, urllib.request.Request], urllib.request.Request | None]":
    """The recovery a refused call gets: re-derive whatever the relay moved, then retry it once.

    Three refusals are the relay's side moving rather than ours, and each has one honest answer:

    * a ``426`` names a version floor we are under → adopt it and re-derive the User-Agent;
    * a ``403 FreeTierError`` is its verdict on a request that no longer looks like the official
      client → re-derive the version, and retry only when that produced a different envelope
      (identical bytes get no second shot in-process; a burst of those is the host's retry policy's
      job, and the AGENTS.md gate notes say so);
    * a ``400 [unsupported_value]`` names a dial this model no longer takes → drop it, remember it.

    Anything else returns None and is raised as it stands.
    """

    def recover(
        status: int, response: str, request: urllib.request.Request
    ) -> urllib.request.Request | None:
        if status == 426:
            return _retry_with(url, body) if adopt_version_floor(response) else None
        if status == 403 and FREE_TIER_REFUSAL.lower() in response.lower():
            # A refusal is the one signal that says to stop trusting the cached version: the relay
            # moves its floor without notice, and the installed client is where the new one is.
            before = client_version()
            return _retry_with(url, body) if client_version(refresh=True) != before else None
        if status == 400 and "unsupported_value" in response.lower():
            effort = body.get("reasoning_effort")
            if isinstance(effort, str) and effort:
                _remember_unsupported_effort(model, effort)
                return _retry_with(
                    url, {key: value for key, value in body.items() if key != "reasoning_effort"}
                )
        return None

    return recover


def _json_get(url: str, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": user_agent()},
    )
    with _urlopen(request, timeout) as response:
        return json.load(response)


def _free_live_models(zen: Any, catalog: Any) -> list[str]:
    zen_models = {
        str(item.get("id") or "")
        for item in (zen.get("data") if isinstance(zen, dict) else []) or []
        if isinstance(item, dict) and item.get("id")
    }
    provider = catalog.get("opencode") if isinstance(catalog, dict) else None
    metadata = provider.get("models") if isinstance(provider, dict) else None
    if not isinstance(metadata, dict):
        return []

    free = []
    for model, details in metadata.items():
        if not isinstance(details, dict) or details.get("status") == "deprecated":
            continue
        costs = details.get("cost")
        if (
            model in zen_models
            and details.get("tool_call") is True
            and isinstance(costs, dict)
            and costs
            and all(
                isinstance(cost, (int, float))
                and not isinstance(cost, bool)
                and cost == 0
                for cost in costs.values()
            )
        ):
            free.append(str(model))
    return sorted(free)


def _free_live_limits(models: list[str], catalog: Any) -> dict[str, dict[str, int]]:
    """The ``limit`` row each free model carries in the models.dev catalog.

    The relay's own ``/models`` lists ids only, so this is where a real context window and output
    cap come from — the same rows the official OpenCode routes read.
    """
    provider = catalog.get("opencode") if isinstance(catalog, dict) else None
    metadata = provider.get("models") if isinstance(provider, dict) else None
    if not isinstance(metadata, dict):
        return {}
    return _clean_limits(
        {
            model: (metadata[model] or {}).get("limit")
            for model in models
            if isinstance(metadata.get(model), dict)
        }
    )


def _clean_limits(value: Any) -> dict[str, dict[str, int]]:
    """Keep only usable ``context``/``output`` numbers, so a junk cache cannot leak into a hook."""
    if not isinstance(value, dict):
        return {}
    limits: dict[str, dict[str, int]] = {}
    for model, entry in value.items():
        if not isinstance(model, str) or not isinstance(entry, dict):
            continue
        numbers = {
            key: number
            for key, number in entry.items()
            if key in ("context", "output")
            and isinstance(number, int)
            and not isinstance(number, bool)
            and number > 0
        }
        if numbers:
            limits[model] = numbers
    return limits


def _model_cache_path() -> Path:
    return get_hermes_home() / "cache" / "oc-free-provider" / "models.json"


def _load_model_cache() -> tuple[list[str], dict[str, dict[str, int]]]:
    """The cached discovery snapshot as ``(models, limits)``; a legacy bare list reads as ids only."""
    try:
        cached = json.loads(_model_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], {}
    models = cached.get("models") if isinstance(cached, dict) else cached
    if not isinstance(models, list) or any(
        not isinstance(model, str) or not model for model in models
    ):
        return [], {}
    limits = cached.get("limits") if isinstance(cached, dict) else None
    return sorted(dict.fromkeys(models)), _clean_limits(limits)


def _save_model_cache(models: list[str], limits: dict[str, dict[str, int]]) -> None:
    path = _model_cache_path()
    temporary = path.with_suffix(f".{secrets.token_hex(4)}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps({"models": models, "limits": limits}) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


#: Any ``N.N.N`` in a version string the relay or the official client hands us.
_SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _semver(value: Any) -> str | None:
    """The first ``N.N.N`` in *value* as a normalized string, or None when there is none."""
    match = _SEMVER.search(str(value or ""))
    return ".".join(match.groups()) if match else None


def _version_key(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def _handshake_path() -> Path:
    return _model_cache_path().with_name("handshake.json")


def _load_handshake() -> dict[str, Any]:
    """The derived handshake as written by this provider; an unreadable file reads as empty."""
    try:
        state = json.loads(_handshake_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _save_handshake(fields: dict[str, Any]) -> None:
    path = _handshake_path()
    temporary = path.with_suffix(f".{secrets.token_hex(4)}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(fields) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _installed_client_version() -> str | None:
    """The official client's own version, or None when it is absent or will not answer.

    Never raises and never blocks a turn for long: the answer is cached for ``VERSION_CHECK_TTL``,
    and a host without the CLI falls back to the pinned constant.
    """
    try:
        done = subprocess.run(
            [OPENCODE_CLI, "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _semver(done.stdout) if done.returncode == 0 else None


def client_version(refresh: bool = False) -> str:
    """The semver this client claims — derived from what is on the host and from the relay's floor.

    The candidates are the pinned constant, the value resolved earlier (cached beside the model
    cache) and the relay's announced minimum; the highest wins, so the answer never drops below what
    has already been proven or promised. The official client is asked at most once per
    ``VERSION_CHECK_TTL`` (and always on *refresh*, which a refusal triggers) — it is the client the
    relay is built to serve, and it updates itself, so its number is the one that tracks the floor
    the relay moves.
    """
    global _VERSION
    with _HANDSHAKE_LOCK:
        if _VERSION is not None and not refresh:
            return _VERSION
        state = _load_handshake()
        candidates = [
            _semver(OPENCODE_VERSION),
            _semver(state.get("version")),
            _semver(state.get("floor")),
        ]
        checked = state.get("checked")
        fresh = (
            isinstance(checked, (int, float))
            and not isinstance(checked, bool)
            and (time.time() - float(checked)) < VERSION_CHECK_TTL
        )
        if refresh or not fresh:
            candidates.append(_installed_client_version())
        version = max((item for item in candidates if item), key=_version_key)
        _VERSION = version
        _save_handshake({**state, "version": version, "checked": time.time()})
        return version


def user_agent() -> str:
    """The bare ``opencode/<semver>`` the relay accepts, with the current derived version."""
    return f"opencode/{client_version()}"


def adopt_version_floor(text: str) -> str | None:
    """Adopt a minimum version the relay announced, and return it when that raises the floor.

    A 426 names the minimum client version the relay now serves. Raising the floor is the whole fix
    for that refusal, so the value is kept and the version re-resolved on the next call.
    """
    global _VERSION
    floor = _semver(text)
    if floor is None:
        return None
    with _HANDSHAKE_LOCK:
        state = _load_handshake()
        known = _semver(state.get("floor"))
        if known and _version_key(known) >= _version_key(floor):
            return None
        _save_handshake({**state, "floor": floor})
        _VERSION = None
    return floor


def _models_for_process(timeout: float) -> tuple[str, ...]:
    global _MODEL_SNAPSHOT, _CATALOG_LIMITS
    with _MODEL_SNAPSHOT_LOCK:
        if _MODEL_SNAPSHOT is None:
            limits: dict[str, dict[str, int]] = {}
            try:
                zen = _json_get(MODELS_URL, timeout)
                catalog = _json_get(MODELS_DEV_URL, timeout)
                models = _free_live_models(zen, catalog)
                limits = _free_live_limits(models, catalog)
            except (OSError, ValueError, urllib.error.URLError):
                models = []
            if models:
                _save_model_cache(models, limits)
            else:
                models, limits = _load_model_cache()
                models = models or list(FALLBACK_MODELS)
            _MODEL_SNAPSHOT = tuple(models)
            _CATALOG_LIMITS = limits
        return _MODEL_SNAPSHOT


def catalog_limit(model: str | None, key: str) -> int | None:
    """A free model's own catalog ``limit`` (``context``/``output``), or None while cold.

    Cache-only by design: the provider hooks that read it run on the session's hot path, so the
    worst case is one small file read, never a request.
    """
    global _CATALOG_LIMITS
    if not model:
        return None
    with _MODEL_SNAPSHOT_LOCK:
        if not _CATALOG_LIMITS:
            _CATALOG_LIMITS = _load_model_cache()[1]
        entry = _CATALOG_LIMITS.get(str(model)) or {}
        value = entry.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _session_id() -> str:
    """OpenCode session id: ``ses_`` + 6 inverted-timestamp bytes + 14 base62 chars.

    The official client encodes ``~(ms * 4096 + counter)`` in the low 48 bits, so
    ids sort newest-first; the relay rejects any id that is not encoded this way.
    """
    global _SESSION_COUNTER
    with _SESSION_LOCK:
        _SESSION_COUNTER = (_SESSION_COUNTER + 1) & 0xFFF
        counter = _SESSION_COUNTER
        stamp = int(time.time() * 1000)
    value = ~((stamp << 12) + counter) & ((1 << 48) - 1)
    return (
        "ses_"
        + value.to_bytes(6, "big").hex()
        + "".join(secrets.choice(_BASE62) for _ in range(14))
    )


def _compat_tool(name: str) -> dict[str, Any]:
    """OpenCode-named alias for a Hermes tool this session did not supply.

    The relay fingerprints the advertised tool schemas and rejects a ``read``
    with an empty parameter object, so the marker keeps OpenCode's real schema
    even though the tool cannot be dispatched; the description says so.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Compatibility marker; unavailable. Never call.",
            "parameters": COMPAT_TOOL_PARAMETERS[name],
        },
    }


def _mapped_tool(name: str, target: dict[str, Any]) -> dict[str, Any]:
    function = target.get("function") or {}
    target_name = str(function.get("name") or COMPAT_TOOL_TARGETS[name])
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                f"{COMPAT_TOOL_DESCRIPTIONS[name]} "
                f"OpenCode-compatible alias for Hermes {target_name}."
            ),
            "parameters": COMPAT_TOOL_PARAMETERS[name],
        },
    }


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
        return ""
    return str(tool["function"].get("name") or "").strip()


def _wire_tools(
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_name = {_tool_name(tool): tool for tool in tools or [] if _tool_name(tool)}
    mapped: dict[str, str] = {}
    wire = []
    for name in COMPAT_TOOL_NAMES:
        target_name = COMPAT_TOOL_TARGETS[name]
        if target := by_name.get(target_name):
            wire.append(_mapped_tool(name, target))
            mapped[name] = target_name
        else:
            wire.append(_compat_tool(name))
    mapped_targets = set(mapped.values())
    wire.extend(
        tool
        for tool in tools or []
        if _tool_name(tool) not in COMPAT_TOOL_NAMES
        and _tool_name(tool) not in mapped_targets
    )
    return wire, mapped


def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        function = tool.get("function") or {}
        result.append(
            {
                "type": "function",
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "parameters": function.get("parameters") or {"type": "object"},
                "strict": bool(function.get("strict", False)),
            }
        )
    return result


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("text") is not None
        )
    return "" if content is None else str(content)


def _responses_content(content: Any, *, output: bool = False) -> list[dict[str, Any]]:
    text_type = "output_text" if output else "input_text"
    if not isinstance(content, list):
        text = _message_text(content)
        return [{"type": text_type, "text": text}] if text else []
    result = []
    for part in content:
        if not isinstance(part, dict):
            result.append({"type": text_type, "text": str(part)})
            continue
        kind = str(part.get("type") or "")
        if kind in {"text", "input_text", "output_text"}:
            result.append({"type": text_type, "text": str(part.get("text") or "")})
        elif not output and kind in {"image_url", "input_image"}:
            image = part.get("image_url")
            if isinstance(image, dict):
                image = image.get("url")
            if image:
                result.append({"type": "input_image", "image_url": str(image)})
    return result


def _responses_input(
    messages: list[dict[str, Any]],
    mapped_tools: dict[str, str],
) -> tuple[list[dict[str, Any]], str]:
    instructions = ["You are opencode"]
    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "system":
            if text := _message_text(message.get("content")):
                instructions.append(text)
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or message.get("call_id") or "")
            if call_id:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": _message_text(message.get("content")),
                    }
                )
            continue
        if role == "assistant":
            for detail in message.get("reasoning_details") or []:
                if isinstance(detail, dict) and detail.get("type") == "reasoning":
                    items.append(dict(detail))
            if content := _responses_content(message.get("content"), output=True):
                items.append({"role": "assistant", "content": content})
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                arguments = function.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, separators=(",", ":"))
                name, arguments = _opencode_tool(
                    str(function.get("name") or ""), arguments, mapped_tools
                )
                call_id = str(call.get("id") or call.get("call_id") or "")
                if call_id:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments,
                        }
                    )
            continue
        content = _responses_content(message.get("content"))
        if content:
            items.append(
                {"role": "user" if role == "user" else role, "content": content}
            )
    return items, "\n\n".join(instructions)


def _responses_tool_choice(value: Any, mapped_tools: dict[str, str]) -> Any:
    value = _wire_tool_choice(value, mapped_tools)
    if not isinstance(value, dict):
        return value
    function = value.get("function")
    if value.get("type") == "function" and isinstance(function, dict):
        return {"type": "function", "name": str(function.get("name") or "")}
    return value


def _effective_timeout(value: Any, default: float = 900.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    values = [
        getattr(value, key, None)
        for key in ("read", "write", "connect", "pool", "timeout")
    ]
    return max(
        (float(item) for item in values if isinstance(item, (int, float))),
        default=default,
    )


def _unsupported_efforts(model: str) -> frozenset[str]:
    """The dial values this model has already answered ``400 [unsupported_value]`` for."""
    state = _load_handshake()
    table = state.get("unsupported_efforts")
    values = table.get(model) if isinstance(table, dict) else None
    return frozenset(str(item) for item in values) if isinstance(values, list) else frozenset()


def _remember_unsupported_effort(model: str, effort: str) -> None:
    """Remember a dial this model refused, so the same session is not charged for it twice."""
    with _HANDSHAKE_LOCK:
        state = _load_handshake()
        table = state.get("unsupported_efforts")
        known = dict(table) if isinstance(table, dict) else {}
        values = {str(item) for item in (known.get(model) or [])}
        values.add(effort)
        known[model] = sorted(values)
        _save_handshake({**state, "unsupported_efforts": known})


def _reasoning_effort(value: Any, model: str) -> str | None:
    """The host's reasoning dial, forwarded as it is — the relay owns this vocabulary, not us.

    Measured against the live relay (2026-10-17): ``none`` (thinking off for real — verified by an
    empty reasoning stream), ``minimal``, ``low``, ``medium``, ``high``, ``max``, ``xhigh`` and even
    ``ultra`` all stream, where an earlier pass had measured five of them and dropped the rest —
    including the host's own thinking-off ``none``, which silently kept thinking. So the value is
    sent rather than filtered, and a ``400 [unsupported_value]`` is what teaches us a model that has
    narrowed: one retry without the dial, and the value is remembered for that model.
    """
    effort = str(value or "").strip().lower()
    if not effort or effort in _unsupported_efforts(model):
        return None
    return effort


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _usage(value: Any = None) -> SimpleNamespace:
    raw = value if isinstance(value, dict) else {}
    prompt = int(raw.get("prompt_tokens", raw.get("input", 0)) or 0)
    completion = int(raw.get("completion_tokens", raw.get("output", 0)) or 0)
    total = int(raw.get("total_tokens", prompt + completion) or prompt + completion)
    raw_details = raw.get("prompt_tokens_details")
    details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
    raw_completion = raw.get("completion_tokens_details")
    completion_details: dict[str, Any] = (
        raw_completion if isinstance(raw_completion, dict) else {}
    )
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=int(details.get("cached_tokens", 0) or 0)
        ),
        # Hermes reads a chat-completions provider's thinking tokens out of
        # ``completion_tokens_details.reasoning_tokens`` (agent/usage_pricing.py:654). The relay
        # sends them on the final chunk, so omitting the key reports every free-model turn as
        # having thought zero tokens.
        completion_tokens_details=SimpleNamespace(
            reasoning_tokens=int(completion_details.get("reasoning_tokens", 0) or 0)
        ),
    )


def _required(arguments: dict[str, Any], name: str, key: str) -> Any:
    if key not in arguments:
        raise OpenCodeError(
            f"OpenCode tool '{name}' omitted required argument '{key}'."
        )
    return arguments[key]


def _mapped_arguments(name: str, encoded: str) -> str:
    try:
        arguments = json.loads(encoded or "{}")
    except json.JSONDecodeError as exc:
        raise OpenCodeError(
            f"OpenCode tool '{name}' returned invalid JSON arguments."
        ) from exc
    if not isinstance(arguments, dict):
        raise OpenCodeError(f"OpenCode tool '{name}' arguments must be a JSON object.")

    if name == "bash":
        mapped = {"command": _required(arguments, name, "command")}
        if "workdir" in arguments:
            mapped["workdir"] = arguments["workdir"]
        if "timeout" in arguments:
            milliseconds = int(arguments["timeout"])
            mapped["timeout"] = max(1, (milliseconds + 999) // 1000)
    elif name == "edit":
        mapped = {
            "mode": "replace",
            "path": _required(arguments, name, "filePath"),
            "old_string": _required(arguments, name, "oldString"),
            "new_string": _required(arguments, name, "newString"),
        }
        if "replaceAll" in arguments:
            mapped["replace_all"] = arguments["replaceAll"]
    elif name == "glob":
        mapped = {
            "target": "files",
            "pattern": _required(arguments, name, "pattern"),
        }
        if "path" in arguments:
            mapped["path"] = arguments["path"]
    elif name == "grep":
        mapped = {
            "target": "content",
            "pattern": _required(arguments, name, "pattern"),
        }
        if "path" in arguments:
            mapped["path"] = arguments["path"]
        if "include" in arguments:
            mapped["file_glob"] = arguments["include"]
    elif name == "read":
        mapped = {"path": _required(arguments, name, "filePath")}
        for key in ("offset", "limit"):
            if key in arguments:
                mapped[key] = arguments[key]
    elif name == "skill":
        mapped = {"name": _required(arguments, name, "name")}
    elif name == "task":
        context = [
            str(_required(arguments, name, "description")),
            f"Requested OpenCode subagent type: {_required(arguments, name, 'subagent_type')}",
        ]
        for key, label in (
            ("task_id", "Previous task ID"),
            ("command", "Requested command"),
            ("background", "Requested background execution"),
        ):
            if key in arguments:
                context.append(f"{label}: {arguments[key]}")
        mapped = {
            "tasks": [
                {
                    "goal": _required(arguments, name, "prompt"),
                    "context": "\n".join(context),
                }
            ]
        }
    elif name == "todowrite":
        todos = _required(arguments, name, "todos")
        if not isinstance(todos, list):
            raise OpenCodeError(
                "OpenCode tool 'todowrite' argument 'todos' must be a list."
            )
        mapped = {
            "todos": [
                {
                    "id": f"oc-{index + 1}",
                    "content": _required(todo, name, "content"),
                    "status": _required(todo, name, "status"),
                }
                for index, todo in enumerate(todos)
                if isinstance(todo, dict)
            ],
            "merge": False,
        }
        if len(mapped["todos"]) != len(todos):
            raise OpenCodeError("OpenCode tool 'todowrite' contains a non-object item.")
    elif name == "webfetch":
        mapped = {"urls": [_required(arguments, name, "url")]}
    elif name == "websearch":
        mapped = {"query": _required(arguments, name, "query")}
        if "numResults" in arguments:
            mapped["limit"] = max(1, min(100, int(arguments["numResults"])))
    elif name == "write":
        mapped = {
            "path": _required(arguments, name, "filePath"),
            "content": _required(arguments, name, "content"),
        }
    else:
        mapped = arguments
    return json.dumps(mapped, separators=(",", ":"))


def _opencode_arguments(name: str, encoded: str) -> str:
    try:
        arguments = json.loads(encoded or "{}")
    except json.JSONDecodeError as exc:
        raise OpenCodeError(
            f"Hermes tool '{name}' has invalid JSON arguments in conversation history."
        ) from exc
    if not isinstance(arguments, dict):
        raise OpenCodeError(
            f"Hermes tool '{name}' arguments in conversation history must be an object."
        )

    if name == "bash":
        mapped = {"command": _required(arguments, name, "command")}
        if "workdir" in arguments:
            mapped["workdir"] = arguments["workdir"]
        if "timeout" in arguments:
            mapped["timeout"] = max(1, int(arguments["timeout"]) * 1000)
    elif name == "edit":
        if arguments.get("mode", "replace") != "replace":
            raise OpenCodeError(
                "A non-replace Hermes patch cannot be replayed as edit."
            )
        mapped = {
            "filePath": _required(arguments, name, "path"),
            "oldString": _required(arguments, name, "old_string"),
            "newString": _required(arguments, name, "new_string"),
        }
        if "replace_all" in arguments:
            mapped["replaceAll"] = arguments["replace_all"]
    elif name in {"glob", "grep"}:
        mapped = {"pattern": _required(arguments, name, "pattern")}
        if "path" in arguments:
            mapped["path"] = arguments["path"]
        if name == "grep" and "file_glob" in arguments:
            mapped["include"] = arguments["file_glob"]
    elif name == "read":
        mapped = {"filePath": _required(arguments, name, "path")}
        for key in ("offset", "limit"):
            if key in arguments:
                mapped[key] = arguments[key]
    elif name == "skill":
        mapped = {"name": _required(arguments, name, "name")}
    elif name == "task":
        tasks = _required(arguments, name, "tasks")
        if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
            raise OpenCodeError("Hermes delegate history cannot be replayed as task.")
        task = tasks[0]
        context = str(task.get("context") or "")
        lines = context.splitlines()
        mapped = {
            "description": lines[0] if lines else "Delegated Hermes task",
            "prompt": _required(task, name, "goal"),
            "subagent_type": "general",
        }
        prefixes = {
            "Requested OpenCode subagent type: ": "subagent_type",
            "Previous task ID: ": "task_id",
            "Requested command: ": "command",
            "Requested background execution: ": "background",
        }
        for line in lines[1:]:
            for prefix, key in prefixes.items():
                if line.startswith(prefix):
                    value: Any = line.removeprefix(prefix)
                    if key == "background":
                        value = str(value).lower() == "true"
                    mapped[key] = value
                    break
    elif name == "todowrite":
        todos = _required(arguments, name, "todos")
        if not isinstance(todos, list) or any(
            not isinstance(todo, dict) for todo in todos
        ):
            raise OpenCodeError("Hermes todo history cannot be replayed as todowrite.")
        mapped = {
            "todos": [
                {
                    "content": _required(todo, name, "content"),
                    "status": _required(todo, name, "status"),
                    "priority": "medium",
                }
                for todo in todos
            ]
        }
    elif name == "webfetch":
        urls = _required(arguments, name, "urls")
        if not isinstance(urls, list) or not urls:
            raise OpenCodeError(
                "Hermes web extract history cannot be replayed as webfetch."
            )
        mapped = {"url": urls[0]}
    elif name == "websearch":
        mapped = {"query": _required(arguments, name, "query")}
        if "limit" in arguments:
            mapped["numResults"] = arguments["limit"]
    elif name == "write":
        mapped = {
            "filePath": _required(arguments, name, "path"),
            "content": _required(arguments, name, "content"),
        }
    else:
        mapped = arguments
    return json.dumps(mapped, separators=(",", ":"))


def _opencode_alias(name: str, arguments: str, mapped_tools: dict[str, str]) -> str:
    aliases = [alias for alias in COMPAT_TOOL_NAMES if mapped_tools.get(alias) == name]
    if not aliases:
        return name
    if name == "search_files":
        try:
            target = json.loads(arguments or "{}").get("target")
        except (AttributeError, json.JSONDecodeError):
            target = None
        return "glob" if target == "files" else "grep"
    return aliases[0]


def _opencode_tool(
    name: str, arguments: str, mapped_tools: dict[str, str]
) -> tuple[str, str]:
    alias = _opencode_alias(name, arguments, mapped_tools)
    if alias == name:
        return name, arguments or "{}"
    return alias, _opencode_arguments(alias, arguments)


def _wire_messages(
    messages: list[dict[str, Any]], mapped_tools: dict[str, str]
) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        item = dict(message)
        if item.get("role") == "assistant" and item.get("tool_calls"):
            calls = []
            for call in item["tool_calls"]:
                if not isinstance(call, dict):
                    calls.append(call)
                    continue
                wired_call = dict(call)
                function = dict(call.get("function") or {})
                arguments = function.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, separators=(",", ":"))
                function["name"], function["arguments"] = _opencode_tool(
                    str(function.get("name") or ""), arguments, mapped_tools
                )
                wired_call["function"] = function
                calls.append(wired_call)
            item["tool_calls"] = calls
        result.append(item)
    return result


def _wire_tool_choice(value: Any, mapped_tools: dict[str, str]) -> Any:
    if not isinstance(value, dict):
        return value
    function = value.get("function")
    if value.get("type") != "function" or not isinstance(function, dict):
        return value
    result = dict(value)
    wired_function = dict(function)
    wired_function["name"] = _opencode_alias(
        str(wired_function.get("name") or ""), "{}", mapped_tools
    )
    result["function"] = wired_function
    return result


def _translate_tool(
    name: str, arguments: str, mapped_tools: dict[str, str]
) -> tuple[str, str]:
    if error := _phantom_tool_error(name, mapped_tools):
        raise error
    target = mapped_tools.get(name, name)
    if target != name:
        arguments = _mapped_arguments(name, arguments)
    return target, arguments or "{}"


def _tool_calls_from_parts(
    parts: dict[int, dict[str, Any]], mapped_tools: dict[str, str]
) -> list[SimpleNamespace]:
    result = []
    for index in sorted(parts):
        call = parts[index]
        name, arguments = _translate_tool(
            str(call.get("name") or ""),
            str(call.get("arguments") or "{}"),
            mapped_tools,
        )
        result.append(
            SimpleNamespace(
                id=call.get("id") or f"call_{index}",
                call_id=call.get("id") or f"call_{index}",
                type=call.get("type") or "function",
                function=SimpleNamespace(name=name, arguments=arguments),
                response_item_id=None,
            )
        )
    return result


def _tool_call_dicts(
    parts: dict[int, dict[str, Any]], mapped_tools: dict[str, str]
) -> list[dict[str, Any]]:
    result = []
    for index in sorted(parts):
        call = parts[index]
        name, arguments = _translate_tool(
            str(call.get("name") or ""),
            str(call.get("arguments") or "{}"),
            mapped_tools,
        )
        result.append(
            {
                "index": index,
                "id": call.get("id") or f"call_{index}",
                "type": call.get("type") or "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return result


def _phantom_tool_error(
    name: str, mapped_tools: dict[str, str]
) -> OpenCodeError | None:
    if name in COMPAT_TOOL_NAMES and name not in mapped_tools:
        return OpenCodeError(
            f"OpenCode attempted compatibility-only tool '{name}'. "
            "No matching Hermes tool was supplied, so it was not executed."
        )
    return None


def _merge_sse(
    events: Iterator[dict[str, Any]],
    *,
    requested_model: str,
    mapped_tools: dict[str, str],
) -> SimpleNamespace:
    content: list[str] = []
    reasoning: list[str] = []
    reasoning_details: list[Any] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason = "stop"
    usage: Any = None
    model = requested_model
    seen = False
    for event in events:
        seen = True
        if event.get("error"):
            raise OpenCodeError(f"OpenCode inference error: {event['error']}")
        model = str(event.get("model") or model)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("content") is not None:
                content.append(str(delta["content"]))
            for key in ("reasoning_content", "reasoning"):
                if delta.get(key) is not None:
                    reasoning.append(str(delta[key]))
                    break
            details = delta.get("reasoning_details")
            if isinstance(details, list):
                reasoning_details.extend(details)
            elif details is not None:
                reasoning_details.append(details)
            for raw_call in delta.get("tool_calls") or []:
                index = int(raw_call.get("index", 0) or 0)
                item = calls.setdefault(
                    index, {"id": "", "type": "function", "name": "", "arguments": ""}
                )
                if raw_call.get("id"):
                    item["id"] = str(raw_call["id"])
                item["type"] = str(raw_call.get("type") or item["type"])
                function = raw_call.get("function") or {}
                if function.get("name"):
                    item["name"] = str(function["name"])
                item["arguments"] += str(function.get("arguments") or "")
                if error := _phantom_tool_error(item["name"], mapped_tools):
                    raise error
    if not seen:
        raise OpenCodeError("OpenCode inference returned no events.")
    if finish_reason == "content_filter":
        raise OpenCodeError("OpenCode inference was blocked by the content filter.")
    tool_calls = _tool_calls_from_parts(calls, mapped_tools)
    text = "".join(content) or None
    thought = "".join(reasoning) or None
    return SimpleNamespace(
        id=None,
        object="chat.completion",
        model=model,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="tool_calls" if tool_calls else finish_reason,
                message=SimpleNamespace(
                    role="assistant",
                    content=text,
                    tool_calls=tool_calls,
                    reasoning=thought,
                    reasoning_content=thought,
                    reasoning_details=reasoning_details or None,
                ),
            )
        ],
        usage=_usage(usage),
    )


def _stream_sse(
    events: Iterator[dict[str, Any]],
    *,
    requested_model: str,
    mapped_tools: dict[str, str],
) -> Iterator[dict[str, Any]]:
    calls: dict[int, dict[int, dict[str, Any]]] = {}
    model = requested_model
    seen = False
    for event in events:
        seen = True
        if event.get("error"):
            raise OpenCodeError(f"OpenCode inference error: {event['error']}")
        model = str(event.get("model") or model)
        event.setdefault("choices", [])
        for choice in event.get("choices") or []:
            if choice.get("finish_reason") == "content_filter":
                raise OpenCodeError(
                    "OpenCode inference was blocked by the content filter."
                )
            choice_index = int(choice.get("index", 0) or 0)
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
                choice["delta"] = delta
            raw_calls = delta.pop("tool_calls", [])
            choice_calls = calls.get(choice_index, {})
            for raw_call in raw_calls:
                choice_calls = calls.setdefault(choice_index, {})
                index = int(raw_call.get("index", 0) or 0)
                item = choice_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "name": "", "arguments": ""},
                )
                if raw_call.get("id"):
                    item["id"] = str(raw_call["id"])
                item["type"] = str(raw_call.get("type") or item["type"])
                function = raw_call.get("function") or {}
                if function.get("name"):
                    item["name"] = str(function["name"])
                item["arguments"] += str(function.get("arguments") or "")
                if error := _phantom_tool_error(item["name"], mapped_tools):
                    raise error
            if choice.get("finish_reason") and choice_calls:
                delta["tool_calls"] = _tool_call_dicts(choice_calls, mapped_tools)
                calls.pop(choice_index)
        yield event
    if not seen:
        raise OpenCodeError("OpenCode inference returned no events.")
    if calls:
        yield {
            "model": model,
            "choices": [
                {
                    "index": choice_index,
                    "delta": {
                        "tool_calls": _tool_call_dicts(choice_calls, mapped_tools)
                    },
                    "finish_reason": "tool_calls",
                }
                for choice_index, choice_calls in sorted(calls.items())
            ],
        }


def _next_or_done(iterator: Iterator[Any]) -> tuple[bool, Any]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


class _ResponseHandle:
    """An already-fetched response that the async contract can still await.

    ``create()`` performs the request eagerly and hands back the parsed response
    wrapped here; the wrapper exists only so ``await client.chat.completions.create(...)``
    works (the async client contract), and delegates every attribute to the response.
    """

    __slots__ = ("_value",)

    def __init__(self, value: SimpleNamespace) -> None:
        self._value = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)

    def __await__(self):
        async def _self() -> "_ResponseHandle":
            return self

        return _self().__await__()


class _LazyStream:
    def __init__(self, factory):
        self._factory = factory
        self._iterator = None

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self._factory())
        return self._iterator

    def close(self) -> None:
        close = getattr(self._iterator, "close", None)
        if callable(close):
            close()

    def __await__(self):
        async def _self():
            return self

        return _self().__await__()

    def __aiter__(self):
        async def _iterate():
            iterator = iter(self)
            while True:
                ok, item = await asyncio.to_thread(_next_or_done, iterator)
                if not ok:
                    return
                yield item

        return _iterate()


class OpenCodeClient:
    """Minimal OpenAI-compatible client used by the Hermes provider profile."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "opencode-public"
        self.base_url = base_url or LOGICAL_BASE_URL
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_chat_completion)
        )
        self.is_closed = False
        self._session = _session_id()
        self._inflight: set[Any] = set()
        self._inflight_lock = threading.Lock()

    def close(self) -> None:
        """Close the client AND abort whatever request it has in flight.

        Hermes' interrupt and stale watchdogs close the request client from a stranger thread and
        expect the socket to die (`agent/client_lifecycle.py`); a flag alone leaves the worker
        blocked in the read until the HTTP timeout expires, so the turn keeps looking active — and
        a gateway restart keeps waiting on it. Closing the live response breaks that read.
        """
        self.is_closed = True
        with self._inflight_lock:
            live = list(self._inflight)
            self._inflight.clear()
        for response in live:
            try:
                response.close()
            except Exception:  # noqa: BLE001 -- an abort must never raise into the closer.
                pass

    @contextlib.contextmanager
    def _track_inflight(self, response: Any) -> Iterator[Any]:
        """Register a live HTTP response for the life of its read (see ``close``)."""
        with self._inflight_lock:
            self._inflight.add(response)
        try:
            yield response
        finally:
            with self._inflight_lock:
                self._inflight.discard(response)

    def list_models(self, *, timeout: float = 15.0) -> list[str]:
        return list(_models_for_process(timeout))

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        timeout: Any = None,
        temperature: Any = None,
        max_tokens: Any = None,
        top_p: Any = None,
        stop: Any = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        seconds = _effective_timeout(timeout)
        selected = str(model or DEFAULT_MODEL)

        def factory() -> SimpleNamespace:
            events, mapped_tools = self._direct_events(
                selected,
                messages or [],
                tools or [],
                tool_choice=tool_choice,
                timeout=seconds,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                extra_body=extra_body,
                extra=kwargs,
            )
            return _merge_sse(
                events, requested_model=selected, mapped_tools=mapped_tools
            )

        if not stream:
            # Resolve the response here, never lazily. Hermes samples this call's latency
            # BEFORE it reads the body: agent/turn_response_check.py:119 computes
            # ``api_duration`` ahead of the transport normalization that first touches
            # ``response.choices``. A lazily-resolved response therefore reports ~0 s and
            # every derived figure (t/s, average latency) is dropped. A plain OpenAI
            # client blocks inside create(); match that.
            return _ResponseHandle(factory())

        def stream_factory() -> Iterator[SimpleNamespace]:
            events, mapped_tools = self._direct_events(
                selected,
                messages or [],
                tools or [],
                tool_choice=tool_choice,
                timeout=seconds,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                extra_body=extra_body,
                extra=kwargs,
            )
            for event in _stream_sse(
                events, requested_model=selected, mapped_tools=mapped_tools
            ):
                yield _namespace(event)

        return _LazyStream(stream_factory)

    def _direct_events(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: Any,
        timeout: float,
        temperature: Any,
        max_tokens: Any,
        top_p: Any,
        stop: Any,
        extra_body: dict[str, Any] | None,
        extra: dict[str, Any],
    ) -> tuple[Iterator[dict[str, Any]], dict[str, str]]:
        if self.is_closed:
            raise OpenCodeError("OpenCode client is closed.")
        if "/" in model:
            provider, model = model.split("/", 1)
            if provider != "opencode":
                raise OpenCodeError(
                    "Direct mode only supports models from the local "
                    "OpenCode 'opencode' provider."
                )
        if model in RESPONSES_MODELS:
            return self._responses_events(
                model,
                messages,
                tools,
                tool_choice=tool_choice,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_body=extra_body,
                extra=extra,
            )
        wire_tools, mapped_tools = _wire_tools(tools)
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are opencode"},
                *_wire_messages(messages, mapped_tools),
            ],
            "tools": wire_tools,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        optional = {
            "tool_choice": _wire_tool_choice(tool_choice, mapped_tools),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop,
            "parallel_tool_calls": extra.get("parallel_tool_calls"),
            "response_format": extra.get("response_format"),
            # The provider profile puts the reasoning dial here (top level, never extra_body.reasoning,
            # whose emission a host allowlist gates). Which values the models take is the relay's to
            # decide and it changes: forwarded as it is, and learned from the relay's own 400 (see
            # ``_reasoning_effort``) rather than filtered against a list that goes stale.
            "reasoning_effort": _reasoning_effort(extra.get("reasoning_effort"), model),
        }
        body.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        if isinstance(extra_body, dict):
            body.update(
                {
                    key: value
                    for key, value in extra_body.items()
                    if key
                    not in {"model", "messages", "tools", "stream", "stream_options"}
                }
            )
        request = _inference_request(DIRECT_URL, body, self._session)
        return (
            _sse_events(
                request,
                timeout,
                owner=self,
                recover=_refusal_recovery(DIRECT_URL, body, model),
            ),
            mapped_tools,
        )

    def _responses_events(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: Any,
        timeout: float,
        temperature: Any,
        max_tokens: Any,
        top_p: Any,
        extra_body: dict[str, Any] | None,
        extra: dict[str, Any],
    ) -> tuple[Iterator[dict[str, Any]], dict[str, str]]:
        wire_tools, mapped_tools = _wire_tools(tools)
        input_items, instructions = _responses_input(messages, mapped_tools)
        body: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "instructions": instructions,
            "tools": _responses_tools(wire_tools),
            "stream": True,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": self._session,
        }
        optional = {
            "tool_choice": _responses_tool_choice(tool_choice, mapped_tools),
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
            "parallel_tool_calls": extra.get("parallel_tool_calls"),
        }
        body.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        if isinstance(extra_body, dict):
            body.update(
                {
                    key: value
                    for key, value in extra_body.items()
                    if key
                    not in {
                        "model",
                        "input",
                        "instructions",
                        "tools",
                        "stream",
                        "store",
                        "include",
                        "prompt_cache_key",
                    }
                }
            )
        request = _inference_request(RESPONSES_URL, body, self._session)

        def events() -> Iterator[dict[str, Any]]:
            seen = False
            saw_tool_call = False
            call_indexes: dict[int, int] = {}
            selected_model = model
            for event in _sse_events(
                request,
                timeout,
                owner=self,
                recover=_refusal_recovery(RESPONSES_URL, body, model),
            ):
                seen = True
                kind = str(event.get("type") or "")
                response = event.get("response") or {}
                selected_model = str(response.get("model") or selected_model)
                if kind == "error" or event.get("error"):
                    raise OpenCodeError(f"OpenCode inference error: {event}")
                if kind == "response.output_text.delta":
                    yield {
                        "model": selected_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": str(event.get("delta") or "")},
                                "finish_reason": None,
                            }
                        ],
                    }
                elif kind == "response.reasoning_summary_text.delta":
                    yield {
                        "model": selected_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "reasoning_content": str(event.get("delta") or "")
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                elif kind == "response.output_item.done":
                    item = event.get("item") or {}
                    if item.get("type") == "reasoning":
                        yield {
                            "model": selected_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_details": [item]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    elif item.get("type") == "function_call":
                        saw_tool_call = True
                        output_index = int(event.get("output_index", 0) or 0)
                        index = call_indexes.setdefault(output_index, len(call_indexes))
                        yield {
                            "model": selected_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": index,
                                                "id": str(item.get("call_id") or ""),
                                                "type": "function",
                                                "function": {
                                                    "name": str(item.get("name") or ""),
                                                    "arguments": str(
                                                        item.get("arguments") or "{}"
                                                    ),
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                elif kind in {"response.failed", "response.incomplete"}:
                    raise OpenCodeError(
                        f"OpenCode inference did not complete: {response}"
                    )
                elif kind == "response.completed":
                    if response.get("status") not in {None, "completed"}:
                        raise OpenCodeError(
                            f"OpenCode inference did not complete: {response}"
                        )
                    usage = response.get("usage") or {}
                    yield {
                        "model": selected_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls"
                                if saw_tool_call
                                else "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                            "prompt_tokens_details": usage.get(
                                "input_tokens_details", {}
                            ),
                        },
                    }
            if not seen:
                raise OpenCodeError("OpenCode inference returned no events.")

        return events(), mapped_tools
