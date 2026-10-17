# pyright: reportArgumentType=false
# ruff: noqa: E402 -- the standalone plugin directory must be added before import.
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hermes_oc_client as client  # type: ignore[import-not-found]


class FakeResponse:
    def __init__(self, lines=(), body=b"{}"):
        self._lines = [
            line.encode() if isinstance(line, str) else line for line in lines
        ]
        self._body = body

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def sse(*events):
    return FakeResponse(
        [*(f"data: {json.dumps(event)}\n" for event in events), "data: [DONE]\n"]
    )


def tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Hermes {name}",
            "parameters": {"type": "object"},
        },
    }


class ClientTests(unittest.TestCase):
    def setUp(self):
        # This provider's own cache directory (the model cache, the refusal log, the handshake file)
        # is kept off the real home: a test must never write into the live profile's cache.
        cache_dir = tempfile.TemporaryDirectory()
        self.addCleanup(cache_dir.cleanup)
        self.cache_path = Path(cache_dir.name) / "models.json"
        home_scope = mock.patch.object(client, "_model_cache_path", return_value=self.cache_path)
        home_scope.start()
        self.addCleanup(home_scope.stop)

    def test_session_id_matches_opencode_shape(self):
        self.assertRegex(client._session_id(), r"^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$")

    def test_session_id_encodes_a_descending_timestamp(self):
        """The relay rejects a ses_ id whose head is not ~(ms * 4096 + counter)."""
        session = client._session_id()
        stamp = (~int(session[4:16], 16) & ((1 << 48) - 1)) >> 12
        # The 48-bit head keeps only the low 36 bits of the millisecond clock.
        window = 1 << 36
        delta = (stamp - int(time.time() * 1000)) % window
        self.assertLess(min(delta, window - delta), 2000)

    def test_wire_endpoints_use_the_inference_base(self):
        for url in (client.DIRECT_URL, client.RESPONSES_URL, client.MODELS_URL):
            self.assertTrue(url.startswith("https://opencode.ai/inference/v1/"), url)

    def test_compat_markers_keep_the_opencode_parameter_schema(self):
        """An empty parameter object on a marker fails the relay's tool gate."""
        wire, mapped = client._wire_tools([])
        self.assertEqual(mapped, {})
        self.assertEqual(
            [entry["function"]["name"] for entry in wire],
            list(client.COMPAT_TOOL_NAMES),
        )
        for entry in wire:
            name = entry["function"]["name"]
            self.assertEqual(entry["function"]["parameters"], client.COMPAT_TOOL_PARAMETERS[name])
            self.assertTrue(entry["function"]["parameters"].get("properties"))

    def test_envelope_capture_records_what_the_single_build_site_sends(self):
        """$HERMES_OC_DEBUG captures the wire envelope; unset it writes nothing and never raises."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "envelope.jsonl"
            with mock.patch.dict("os.environ", {client.ENVELOPE_CAPTURE_ENV: str(path)}):
                request = client._inference_request(
                    client.DIRECT_URL, {"model": "cat-jam-free", "tools": []}, "ses_abc"
                )
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            captured = json.loads(lines[0])
            self.assertEqual(captured["url"], client.DIRECT_URL)
            self.assertEqual(captured["headers"]["Authorization"], "Bearer public")
            self.assertEqual(captured["headers"]["User-agent"], client.user_agent())
            self.assertRegex(captured["headers"]["User-agent"], r"^opencode/\d+\.\d+\.\d+$")
            self.assertEqual(captured["headers"]["X-opencode-session"], "ses_abc")
            # The body is the exact encoded request body, not a re-serialization of it.
            self.assertEqual(
                json.dumps(json.loads(captured["body"]), separators=(",", ":")),
                bytes(request.data or b"").decode(),
            )

    def test_envelope_capture_is_silent_and_safe_when_disabled_or_unwritable(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop(client.ENVELOPE_CAPTURE_ENV, None)
            request = client._inference_request(client.DIRECT_URL, {"model": "m"}, "ses_abc")
            self.assertTrue(request.data)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "not-a-file"
            directory.mkdir()
            with mock.patch.dict("os.environ", {client.ENVELOPE_CAPTURE_ENV: str(directory)}):
                request = client._inference_request(client.DIRECT_URL, {"model": "m"}, "ses_abc")
            self.assertEqual(request.get_header("User-agent"), client.user_agent())

    def test_direct_request_preserves_hermes_tools_and_returns_native_call(self):
        captured = {}

        def fake_open(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return sse(
                {
                    "model": "m",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "hermes_probe",
                                            "arguments": '{"value":"PING"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
                {
                    "model": "m",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    },
                },
            )

        oc = client.OpenCodeClient()
        tool = {
            "type": "function",
            "function": {
                "name": "hermes_probe",
                "description": "probe",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            },
        }
        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
            result = oc.chat.completions.create(
                model="m",
                messages=[{"role": "user", "content": "probe"}],
                tools=[tool],
                timeout=12,
            )
            call = result.choices[0].message.tool_calls[0]

        body = json.loads(captured["request"].data)
        names = [entry["function"]["name"] for entry in body["tools"]]
        self.assertEqual(
            names[: len(client.COMPAT_TOOL_NAMES)], list(client.COMPAT_TOOL_NAMES)
        )
        self.assertIn("hermes_probe", names)
        self.assertEqual(
            body["messages"][0], {"role": "system", "content": "You are opencode"}
        )
        self.assertEqual(call.function.name, "hermes_probe")
        self.assertEqual(json.loads(call.function.arguments), {"value": "PING"})
        self.assertEqual(result.usage.total_tokens, 6)
        self.assertEqual(captured["timeout"], 12)
        self.assertTrue(
            captured["request"].headers["X-opencode-session"].startswith("ses_")
        )

    def test_reasoning_dial_reaches_the_wire_as_the_host_set_it(self):
        """The relay owns this vocabulary and it moves, so the dial is forwarded, not filtered.

        Measured live (2026-10-17): ``none`` (thinking off for real — an empty reasoning stream),
        ``minimal``, ``low``, ``medium``, ``high``, ``max``, ``xhigh`` and even ``ultra`` all stream.
        An earlier pass had measured five of them and dropped the rest, which quietly kept thinking
        for a session that asked for ``none``. A model that narrows now teaches us through its own
        400 (see ``test_refused_dial_is_dropped_retried_and_remembered``).
        """
        captured = {}

        def fake_open(request, timeout):
            captured["request"] = request
            return sse(
                {"model": "m", "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
            )

        for dial in ("none", "minimal", "low", "medium", "high", "max", "xhigh", "ultra"):
            with self.subTest(dial=dial), mock.patch.object(
                client, "_urlopen", side_effect=fake_open
            ):
                oc = client.OpenCodeClient()
                oc.chat.completions.create(
                    model="m",
                    messages=[{"role": "user", "content": "probe"}],
                    tools=[],
                    reasoning_effort=dial,
                )
            body = json.loads(captured["request"].data)
            self.assertEqual(body.get("reasoning_effort"), dial)

    def test_refused_dial_is_dropped_retried_and_remembered(self):
        """A model that narrowed answers ``400 [unsupported_value]``: drop the dial, retry, remember."""
        attempts = []

        def fake_open(request, timeout):
            attempts.append(json.loads(request.data))
            if len(attempts) == 1:
                raise urllib.error.HTTPError(  # type: ignore[arg-type]
                    client.DIRECT_URL,
                    400,
                    "bad request",
                    {},
                    io.BytesIO(
                        json.dumps(
                            {
                                "type": "error",
                                "error": {
                                    "type": "invalid_request_error",
                                    "code": "unsupported_value",
                                    "message": "This model does not support 'ultra' reasoning effort",
                                },
                            }
                        ).encode()
                    ),
                )
            return sse(
                {"model": "m", "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
            )

        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
            response = client.OpenCodeClient().chat.completions.create(
                model="m",
                messages=[{"role": "user", "content": "probe"}],
                tools=[],
                reasoning_effort="ultra",
            )
        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertEqual(attempts[0].get("reasoning_effort"), "ultra")
        self.assertNotIn("reasoning_effort", attempts[1])
        self.assertEqual(client._unsupported_efforts("m"), frozenset({"ultra"}))

        # The learned value stays out of later calls for the same model — one retry, not one per turn.
        attempts.clear()

        def quiet_open(request, timeout):
            attempts.append(json.loads(request.data))
            return sse(
                {"model": "m", "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
            )

        with mock.patch.object(client, "_urlopen", side_effect=quiet_open):
            client.OpenCodeClient().chat.completions.create(
                model="m",
                messages=[{"role": "user", "content": "probe"}],
                tools=[],
                reasoning_effort="ultra",
            )
        self.assertNotIn("reasoning_effort", attempts[0])

    def test_client_version_is_derived_from_the_installed_client(self):
        """The User-Agent's version is derived and cached, never read from the pin alone."""
        with (
            mock.patch.object(client, "_VERSION", None),
            mock.patch.object(
                client, "_installed_client_version", return_value="2.4.1"
            ) as installed,
        ):
            self.assertEqual(client.client_version(), "2.4.1")
            self.assertEqual(client.user_agent(), "opencode/2.4.1")
            # Inside the trust window the installed client is not asked again.
            installed.side_effect = AssertionError("the installed client was asked twice")
            self.assertEqual(client.client_version(), "2.4.1")
        handshake = json.loads(self.cache_path.with_name("handshake.json").read_text())
        self.assertEqual(handshake["version"], "2.4.1")
        self.assertIn("checked", handshake)

    def test_client_version_never_drops_below_the_pin_or_the_relays_floor(self):
        """A stale client, or none at all, must not lower what we claim."""
        self.cache_path.with_name("handshake.json").write_text(
            json.dumps({"version": "1.0.0", "floor": "9.9.9", "checked": time.time()})
        )
        with (
            mock.patch.object(client, "_VERSION", None),
            mock.patch.object(client, "_installed_client_version", return_value=None),
        ):
            self.assertEqual(client.client_version(), "9.9.9")
        self.cache_path.with_name("handshake.json").unlink()
        with (
            mock.patch.object(client, "_VERSION", None),
            mock.patch.object(client, "_installed_client_version", return_value=None),
        ):
            self.assertEqual(client.client_version(), client.OPENCODE_VERSION)

    def test_version_floor_from_the_relay_is_adopted_and_retried_once(self):
        """A 426 names the minimum the relay now serves: adopt it, and the retry carries it."""
        attempts = []

        def fake_open(request, timeout):
            attempts.append(request)
            if len(attempts) == 1:
                raise urllib.error.HTTPError(  # type: ignore[arg-type]
                    client.DIRECT_URL,
                    426,
                    "upgrade required",
                    {},
                    io.BytesIO(
                        b'{"error":{"message":"client 3.1.0 or newer is required"}}'
                    ),
                )
            return sse(
                {"model": "m", "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
            )

        with (
            mock.patch.object(client, "_VERSION", None),
            mock.patch.object(client, "_installed_client_version", return_value=None),
            mock.patch.object(client, "_urlopen", side_effect=fake_open),
        ):
            response = client.OpenCodeClient().chat.completions.create(
                model="m", messages=[{"role": "user", "content": "probe"}], tools=[]
            )
        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(
            attempts[0].get_header("User-agent"), f"opencode/{client.OPENCODE_VERSION}"
        )
        self.assertEqual(attempts[1].get_header("User-agent"), "opencode/3.1.0")
        # Kept for the next process, not just this call.
        handshake = json.loads(self.cache_path.with_name("handshake.json").read_text())
        self.assertEqual(handshake["floor"], "3.1.0")

    def test_free_tier_refusal_with_an_unchanged_envelope_is_recorded_not_retried(self):
        """Identical bytes get no in-process retry; the host's policy owns that, and we leave evidence.

        The refusal is the relay's verdict on bytes this client built, and the host that built them
        (the gateway) may carry a module nobody can reach into — so the record on disk is what a
        later reader gets, and it carries the request's shape rather than its conversation.
        """
        refusal = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "FreeTierError",
                    "message": "OpenCode's free tier can only be used from within OpenCode.",
                },
            }
        )
        attempts = []

        def fake_open(request, timeout):
            attempts.append(request)
            raise urllib.error.HTTPError(  # type: ignore[arg-type]
                client.DIRECT_URL, 403, "forbidden", {}, io.BytesIO(refusal.encode())
            )

        with (
            mock.patch.object(client, "_VERSION", None),
            mock.patch.object(client, "_installed_client_version", return_value=None),
            mock.patch.object(client, "_urlopen", side_effect=fake_open),
        ):
            with self.assertRaises(client.OpenCodeError) as caught:
                client.OpenCodeClient().chat.completions.create(
                    model="deepseek-v4.1-flash-free",
                    messages=[{"role": "user", "content": "a secret prompt"}],
                    tools=[],
                )
        self.assertEqual(caught.exception.status_code, 403)
        self.assertEqual(len(attempts), 1)
        record = json.loads(
            self.cache_path.with_name(client.REFUSAL_LOG).read_text().splitlines()[-1]
        )
        self.assertEqual(record["status"], 403)
        self.assertIn("FreeTierError", record["response"])
        self.assertEqual(record["url"], client.DIRECT_URL)
        self.assertEqual(record["body"]["model"], "deepseek-v4.1-flash-free")
        self.assertEqual(record["body"]["tools"][:3], ["bash", "edit", "glob"])
        self.assertTrue(record["body"]["sha256"])
        self.assertEqual(
            sorted(record["body"]),
            [
                "bytes",
                "keys",
                "max_tokens",
                "messages",
                "model",
                "reasoning_effort",
                "sha256",
                "tools",
            ],
        )
        self.assertNotIn("a secret prompt", json.dumps(record))

    def test_free_tier_refusal_retries_once_when_the_derived_version_moved(self):
        """A refusal is the signal to stop trusting the cached version; a changed envelope retries."""
        attempts = []

        def fake_open(request, timeout):
            attempts.append(request)
            if len(attempts) == 1:
                raise urllib.error.HTTPError(  # type: ignore[arg-type]
                    client.DIRECT_URL, 403, "forbidden", {}, io.BytesIO(b'{"type":"FreeTierError"}')
                )
            return sse(
                {"model": "m", "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
            )

        with (
            mock.patch.object(client, "_VERSION", None),
            mock.patch.object(
                client, "_installed_client_version", side_effect=["2.0.5", "2.9.0"]
            ),
            mock.patch.object(client, "_urlopen", side_effect=fake_open),
        ):
            response = client.OpenCodeClient().chat.completions.create(
                model="m", messages=[{"role": "user", "content": "probe"}], tools=[]
            )
        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertEqual(
            [attempt.get_header("User-agent") for attempt in attempts],
            ["opencode/2.0.5", "opencode/2.9.0"],
        )
        self.assertNotEqual(
            attempts[0].get_header("X-opencode-session"),
            attempts[1].get_header("X-opencode-session"),
        )

    def test_usage_carries_the_reasoning_token_detail_hermes_reads(self):
        """Hermes reads thinking tokens from ``completion_tokens_details`` (usage_pricing.py:654).

        The relay sends that object on the final chunk; without it every free-model turn is
        recorded as having thought zero tokens.
        """
        usage = client._usage(
            {
                "prompt_tokens": 929,
                "completion_tokens": 51,
                "total_tokens": 980,
                "prompt_tokens_details": {"cached_tokens": 896},
                "completion_tokens_details": {"reasoning_tokens": 40},
            }
        )
        self.assertEqual(usage.completion_tokens_details.reasoning_tokens, 40)
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 896)
        self.assertEqual(usage.total_tokens, 980)
        # A payload without the detail reports zero rather than raising.
        self.assertEqual(client._usage({}).completion_tokens_details.reasoning_tokens, 0)
        self.assertEqual(client._usage().completion_tokens_details.reasoning_tokens, 0)

    def test_repeated_tool_identity_and_reasoning_details_are_preserved(self):
        oc = client.OpenCodeClient()
        response = sse(
            {
                "model": "m",
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"type": "reasoning.text", "text": "r"}
                            ],
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "hermes_probe",
                                        "arguments": '{"value":',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
            {
                "model": "m",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "hermes_probe",
                                        "arguments": '"PING"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
        tool = {
            "type": "function",
            "function": {"name": "hermes_probe", "parameters": {"type": "object"}},
        }
        with mock.patch.object(client, "_urlopen", return_value=response):
            result = oc.chat.completions.create(model="m", messages=[], tools=[tool])
            call = result.choices[0].message.tool_calls[0]
        self.assertEqual(call.id, "call_1")
        self.assertEqual(call.function.name, "hermes_probe")
        self.assertEqual(json.loads(call.function.arguments), {"value": "PING"})
        self.assertEqual(
            result.choices[0].message.reasoning_details,
            [{"type": "reasoning.text", "text": "r"}],
        )

    def test_all_opencode_tools_map_to_available_hermes_tools(self):
        targets = list(dict.fromkeys(client.COMPAT_TOOL_TARGETS.values()))
        wire, mapped = client._wire_tools(
            [*(tool(name) for name in targets), tool("hermes_probe")]
        )
        self.assertEqual(mapped, client.COMPAT_TOOL_TARGETS)
        wire_names = [entry["function"]["name"] for entry in wire]
        self.assertEqual(
            wire_names[: len(client.COMPAT_TOOL_NAMES)],
            list(client.COMPAT_TOOL_NAMES),
        )
        self.assertTrue(set(targets).isdisjoint(wire_names))
        self.assertIn("hermes_probe", wire_names)
        self.assertTrue(
            all(
                entry["function"]["parameters"]
                == client.COMPAT_TOOL_PARAMETERS[entry["function"]["name"]]
                for entry in wire[:11]
            )
        )

        cases = {
            "bash": (
                {"command": "pwd", "timeout": 1501, "workdir": "/repo"},
                {"command": "pwd", "timeout": 2, "workdir": "/repo"},
            ),
            "edit": (
                {
                    "filePath": "a.py",
                    "oldString": "old",
                    "newString": "new",
                    "replaceAll": True,
                },
                {
                    "mode": "replace",
                    "path": "a.py",
                    "old_string": "old",
                    "new_string": "new",
                    "replace_all": True,
                },
            ),
            "glob": (
                {"pattern": "*.py", "path": "src"},
                {"target": "files", "pattern": "*.py", "path": "src"},
            ),
            "grep": (
                {"pattern": "TODO", "path": "src", "include": "*.py"},
                {
                    "target": "content",
                    "pattern": "TODO",
                    "path": "src",
                    "file_glob": "*.py",
                },
            ),
            "read": (
                {"filePath": "a.py", "offset": 2, "limit": 4},
                {"path": "a.py", "offset": 2, "limit": 4},
            ),
            "skill": ({"name": "pdf"}, {"name": "pdf"}),
            "task": (
                {
                    "description": "Inspect",
                    "prompt": "Inspect the parser",
                    "subagent_type": "explore",
                    "background": True,
                },
                {
                    "tasks": [
                        {
                            "goal": "Inspect the parser",
                            "context": "Inspect\nRequested OpenCode subagent type: explore\nRequested background execution: True",
                        }
                    ]
                },
            ),
            "todowrite": (
                {
                    "todos": [
                        {"content": "Ship", "status": "in_progress", "priority": "high"}
                    ]
                },
                {
                    "todos": [
                        {"id": "oc-1", "content": "Ship", "status": "in_progress"}
                    ],
                    "merge": False,
                },
            ),
            "webfetch": (
                {"url": "https://example.com", "format": "html", "timeout": 5},
                {"urls": ["https://example.com"]},
            ),
            "websearch": (
                {"query": "Hermes", "numResults": 8, "livecrawl": "always"},
                {"query": "Hermes", "limit": 8},
            ),
            "write": (
                {"filePath": "a.txt", "content": "hello"},
                {"path": "a.txt", "content": "hello"},
            ),
        }
        for alias, (arguments, expected) in cases.items():
            with self.subTest(alias=alias):
                name, encoded = client._translate_tool(
                    alias, json.dumps(arguments), mapped
                )
                self.assertEqual(name, client.COMPAT_TOOL_TARGETS[alias])
                self.assertEqual(json.loads(encoded), expected)
                replay_name, replay_arguments = client._opencode_tool(
                    name, encoded, mapped
                )
                self.assertEqual(replay_name, alias)
                self.assertTrue(
                    set(client.COMPAT_TOOL_PARAMETERS[alias]["required"])
                    <= set(json.loads(replay_arguments))
                )

        wire, mapped = client._wire_tools([tool("read")])
        self.assertNotIn("read", mapped)
        self.assertIn("unavailable", wire[4]["function"]["description"])
        with self.assertRaisesRegex(client.OpenCodeError, "compatibility-only tool"):
            client._translate_tool("read", '{"filePath":"a.py"}', mapped)

    def test_native_tool_history_and_choice_are_rewritten_to_aliases(self):
        captured = {}

        def fake_open(request, _timeout):
            captured["body"] = json.loads(request.data)
            return sse(
                {
                    "model": "m",
                    "choices": [
                        {"delta": {"content": "DONE"}, "finish_reason": "stop"}
                    ],
                }
            )

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command":"pwd","timeout":2}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "/repo",
            },
        ]
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
            result = oc.chat.completions.create(
                model="m",
                messages=messages,
                tools=[tool("terminal"), tool("hermes_probe")],
                tool_choice={"type": "function", "function": {"name": "terminal"}},
            )
            self.assertEqual(result.choices[0].message.content, "DONE")

        body = captured["body"]
        names = [entry["function"]["name"] for entry in body["tools"]]
        self.assertIn("bash", names)
        self.assertIn("hermes_probe", names)
        self.assertNotIn("terminal", names)
        self.assertEqual(body["tool_choice"]["function"]["name"], "bash")
        replay = body["messages"][1]["tool_calls"][0]["function"]
        self.assertEqual(replay["name"], "bash")
        self.assertEqual(
            json.loads(replay["arguments"]), {"command": "pwd", "timeout": 2000}
        )
        self.assertEqual(body["messages"][2], messages[1])

        mapped = {"bash": "terminal"}
        response_input, _ = client._responses_input(messages, mapped)
        self.assertIn(
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "bash",
                "arguments": '{"command":"pwd","timeout":2000}',
            },
            response_input,
        )

    def test_streamed_bash_call_is_buffered_and_mapped_to_terminal(self):
        response = sse(
            {
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": '{"command":"printf',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ' hi","timeout":1500}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_urlopen", return_value=response):
            chunks = list(
                oc.chat.completions.create(
                    model="m", messages=[], tools=[tool("terminal")], stream=True
                )
            )
        calls = [
            call
            for chunk in chunks
            for choice in chunk.choices
            for call in getattr(choice.delta, "tool_calls", [])
        ]
        self.assertEqual(calls[0].function.name, "terminal")
        self.assertEqual(
            json.loads(calls[0].function.arguments),
            {"command": "printf hi", "timeout": 2},
        )

    def test_stream_error_is_explicit_and_session_is_stable(self):
        requests = []

        def fake_open(request, _timeout):
            requests.append(request)
            if len(requests) == 1:
                return sse({"error": {"message": "boom"}})
            return sse(
                {
                    "model": "m",
                    "choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}],
                }
            )

        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
            stream = oc.chat.completions.create(model="m", messages=[], stream=True)
            with self.assertRaisesRegex(client.OpenCodeError, "inference error"):
                list(stream)
            self.assertEqual(
                oc.chat.completions.create(model="m", messages=[])
                .choices[0]
                .message.content,
                "OK",
            )
        self.assertEqual(
            requests[0].headers["X-opencode-session"],
            requests[1].headers["X-opencode-session"],
        )

    def test_tool_result_is_forwarded_unchanged(self):
        captured = {}

        def fake_open(request, _timeout):
            captured["body"] = json.loads(request.data)
            return sse(
                {
                    "model": "m",
                    "choices": [
                        {"delta": {"content": "TOOL_OK"}, "finish_reason": "stop"}
                    ],
                }
            )

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "hermes_probe",
                            "arguments": '{"value":"PING"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "LOCAL_HERMES_RESULT:PONG",
            },
        ]
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
            result = oc.chat.completions.create(model="m", messages=messages, tools=[])
            self.assertEqual(result.choices[0].message.content, "TOOL_OK")
        self.assertEqual(captured["body"]["messages"][2], messages[1])

    def test_responses_model_adapts_native_tool_continuation(self):
        captured = []
        reasoning = {
            "id": "rs_1",
            "type": "reasoning",
            "status": "completed",
            "encrypted_content": "opaque",
            "summary": [],
        }
        first_response = sse(
            {
                "type": "response.created",
                "response": {
                    "model": "muse-spark-1.3-contributor-free",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": reasoning,
            },
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "completed",
                    "name": "hermes_probe",
                    "call_id": "call_1",
                    "arguments": '{"value":"PING"}',
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "model": "muse-spark-1.3-contributor-free",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "total_tokens": 13,
                    },
                },
            },
        )
        second_response = sse(
            {
                "type": "response.output_text.delta",
                "delta": "TOOL_OK",
                "response": {"model": "muse-spark-1.3-contributor-free"},
            },
            {
                "type": "response.completed",
                "response": {
                    "model": "muse-spark-1.3-contributor-free",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 14,
                        "output_tokens": 2,
                        "total_tokens": 16,
                    },
                },
            },
        )

        def fake_open(request, _timeout):
            captured.append((request.full_url, json.loads(request.data)))
            return first_response if len(captured) == 1 else second_response

        tool = {
            "type": "function",
            "function": {
                "name": "hermes_probe",
                "description": "probe",
                "parameters": {"type": "object"},
            },
        }
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
            first = oc.chat.completions.create(
                model="muse-spark-1.3-contributor-free",
                messages=[{"role": "user", "content": "probe"}],
                tools=[tool],
            )
            call = first.choices[0].message.tool_calls[0]
            messages = [
                {"role": "user", "content": "probe"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_details": first.choices[0].message.reasoning_details,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": "LOCAL_HERMES_RESULT:PONG",
                },
            ]
            second = oc.chat.completions.create(
                model="muse-spark-1.3-contributor-free",
                messages=messages,
                tools=[tool],
            )
            self.assertEqual(second.choices[0].message.content, "TOOL_OK")

        self.assertEqual(captured[0][0], client.RESPONSES_URL)
        first_body = captured[0][1]
        self.assertFalse(first_body["store"])
        self.assertEqual(first_body["include"], ["reasoning.encrypted_content"])
        self.assertEqual(first_body["tools"][-1]["name"], "hermes_probe")
        replay = captured[1][1]["input"]
        self.assertIn(reasoning, replay)
        self.assertIn(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "LOCAL_HERMES_RESULT:PONG",
            },
            replay,
        )

    def test_compatibility_only_tool_call_fails_closed(self):
        oc = client.OpenCodeClient()
        response = sse(
            {
                "model": "m",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "x",
                                    "type": "function",
                                    "function": {"name": "bash", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )
        with mock.patch.object(client, "_urlopen", return_value=response):
            with self.assertRaisesRegex(
                client.OpenCodeError, "compatibility-only tool 'bash'"
            ):
                oc.chat.completions.create(model="m", messages=[], tools=[])

    def test_http_error_exposes_status_code(self):
        error = urllib.error.HTTPError(  # type: ignore[arg-type]
            client.DIRECT_URL, 429, "rate", {}, io.BytesIO(b"busy")
        )
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_urlopen", side_effect=error):
            with self.assertRaises(client.OpenCodeError) as caught:
                oc.chat.completions.create(model="m", messages=[], tools=[])
        self.assertEqual(caught.exception.status_code, 429)

    def test_free_tier_refusal_is_classified_retryable(self):
        # The relay's gate refuses a fingerprint-correct request now and then, and serves the same
        # bytes later (replaying a refused turn's own envelope returns 200). The generic 403 verdict
        # ends the turn on it as a dead credential; the hook must make it retryable instead.
        refusal = json.dumps(
            {"type": "error", "error": {"type": "FreeTierError", "message": "can only be used from within OpenCode"}}
        )
        verdict = client.classify_api_error(
            None, status_code=403, message=f"OpenCode free-model inference failed with HTTP 403: {refusal}"
        )
        self.assertIsNotNone(verdict)
        assert verdict is not None  # narrow for the type checker; the assertion above is the test
        self.assertTrue(verdict["retryable"])
        self.assertTrue(verdict["should_fallback"])
        # The host lowercases the message before the hook sees it (``_Ctx.msg``), so that form is
        # the one that actually arrives in production.
        lowered = client.classify_api_error(
            None,
            status_code=403,
            message="opencode free-model inference failed with http 403: "
            '{"type":"error","error":{"type":"freetiererror"}}',
            body={"error": {"type": "FreeTierError"}},
        )
        self.assertIsNotNone(lowered)

    def test_other_failures_keep_the_host_verdict(self):
        # Only the free-tier refusal is transient. A body naming something else — or another status —
        # must return None so the generic pipeline (auth, billing, format) still decides.
        self.assertIsNone(
            client.classify_api_error(None, status_code=403, message="invalid api key")
        )
        self.assertIsNone(
            client.classify_api_error(
                None, status_code=401, message=json.dumps({"error": {"type": "FreeTierError"}})
            )
        )
        self.assertIsNone(client.classify_api_error(None, status_code=None, message="boom"))

    def test_classification_verdict_names_a_host_taxonomy_reason(self):
        # ``reason`` is a FailoverReason *name*, and an unknown one silently drops the verdict.
        try:
            from agent.error_classifier import FailoverReason  # type: ignore[import-not-found]
        except ImportError:  # host tree not on sys.path: the seam itself cannot be checked here
            self.skipTest("host error taxonomy unavailable")
        verdict = client.classify_api_error(None, status_code=403, message="FreeTierError")
        assert verdict is not None
        FailoverReason(verdict["reason"])

    def test_profile_registers_the_hook(self):
        # The hook rides on the dataclass field, so it must be passed to the constructor: a subclass
        # method of the same name would be consumed as that field's default and called unbound.
        try:
            import importlib.util as ilu
            import sys as _sys

            spec = ilu.spec_from_file_location(
                "oc_free_provider_under_test",
                ROOT / "__init__.py",
                submodule_search_locations=[str(ROOT)],
            )
            assert spec is not None and spec.loader is not None
            module = ilu.module_from_spec(spec)
            # `from . import hermes_oc_client` resolves through sys.modules, so a package has to be
            # registered under its own name before its body runs.
            _sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                _sys.modules.pop(spec.name, None)
        except ImportError as exc:  # host provider registry not reachable
            self.skipTest(f"host provider API unavailable: {exc}")
        self.assertIs(module.opencode_free.classify_api_error, module.hermes_oc_client.classify_api_error)
        # Called through the profile it must behave as the hook does — a method of that name would be
        # swallowed as the dataclass field's default and arrive here unbound.
        verdict = module.opencode_free.classify_api_error(None, status_code=403, message="FreeTierError")
        assert verdict is not None
        self.assertTrue(verdict["retryable"])

    def test_empty_and_content_filtered_responses_fail_explicitly(self):
        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_urlopen", return_value=sse()):
            with self.assertRaisesRegex(client.OpenCodeError, "no events"):
                _ = oc.chat.completions.create(model="m", messages=[]).choices
        filtered = sse(
            {
                "model": "m",
                "choices": [{"delta": {}, "finish_reason": "content_filter"}],
            }
        )
        with mock.patch.object(client, "_urlopen", return_value=filtered):
            with self.assertRaisesRegex(client.OpenCodeError, "content filter"):
                _ = oc.chat.completions.create(model="m", messages=[]).choices

    def test_sync_and_async_client_contract(self):
        def fake_open(_request, _timeout):
            return sse(
                {
                    "model": "m",
                    "choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}],
                }
            )

        oc = client.OpenCodeClient()

        async def run():
            completion = await oc.chat.completions.create(model="m", messages=[])
            self.assertEqual(completion.choices[0].message.content, "OK")
            stream = await oc.chat.completions.create(
                model="m", messages=[], stream=True
            )
            chunks = [chunk async for chunk in stream]
            self.assertEqual(chunks[0].choices[0].delta.content, "OK")

        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
            asyncio.run(run())

    def test_redirects_are_disabled(self):
        self.assertIsNone(  # type: ignore[arg-type]
            client._NoRedirect().redirect_request(
                None, None, 302, "", {}, "https://elsewhere"
            )
        )

    def test_closed_client_fails_before_network(self):
        oc = client.OpenCodeClient()
        oc.close()
        with mock.patch.object(client, "_urlopen") as open_request:
            with self.assertRaisesRegex(client.OpenCodeError, "client is closed"):
                oc.chat.completions.create(model="m", messages=[])
        open_request.assert_not_called()

    def test_reasoning_effort_reaches_the_wire_only_when_the_profile_sends_it(self):
        """The profile hands the dial down as a top-level kwarg; the encoder must put it on the wire,
        and leave it off when the profile withholds it (the responses route 400s on it)."""
        captured = {}

        def fake_open(request, timeout):
            captured["request"] = request
            return sse(
                {
                    "model": "m",
                    "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
                }
            )

        oc = client.OpenCodeClient()
        with mock.patch.object(client, "_urlopen", side_effect=fake_open):
            oc.chat.completions.create(
                model="deepseek-v4.1-flash-free",
                messages=[{"role": "user", "content": "hi"}],
                reasoning_effort="high",
            )
            asked = json.loads(captured["request"].data)
            self.assertEqual(asked.get("reasoning_effort"), "high")

            oc.chat.completions.create(
                model="deepseek-v4.1-flash-free",
                messages=[{"role": "user", "content": "hi"}],
            )
            unasked = json.loads(captured["request"].data)
        self.assertNotIn("reasoning_effort", unasked)

    def test_close_aborts_an_in_flight_read(self):
        """``close()`` must break a blocked read, not just set a flag.

        Hermes' interrupt and stale watchdogs close the request client from a stranger thread and
        rely on the socket dying; a flag leaves the worker parked until the HTTP timeout, which is
        what keeps a turn (and a gateway drain) looking active.
        """
        release = threading.Event()
        reading = threading.Event()

        class BlockingRead:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def close(self):
                release.set()

            def __iter__(self):
                reading.set()
                while not release.wait(0.01):
                    pass
                raise ValueError("I/O operation on closed file")

        oc = client.OpenCodeClient()
        outcome: dict = {}

        def run():
            try:
                oc.chat.completions.create(model="m", messages=[])
            except Exception as exc:  # noqa: BLE001 -- asserted below.
                outcome["error"] = exc
            else:
                outcome["error"] = None

        with mock.patch.object(client, "_urlopen", return_value=BlockingRead()):
            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(reading.wait(5), "the request never started reading")
            started = time.monotonic()
            oc.close()
            worker.join(5)
            elapsed = time.monotonic() - started

        self.assertFalse(worker.is_alive(), "close() left the read blocked")
        self.assertLess(elapsed, 2.0, "close() was not what released the read")
        self.assertIsInstance(outcome.get("error"), client.OpenCodeError)
        self.assertIn("closed mid-request", str(outcome["error"]))

    def test_catalog_intersects_live_zero_cost_tool_models(self):
        zen = {"data": [{"id": model} for model in ("free-a", "free-b", "paid")]}
        catalog = {
            "opencode": {
                "models": {
                    "free-a": {"tool_call": True, "cost": {"input": 0, "output": 0}},
                    "free-b": {"tool_call": True, "cost": {"input": 0, "output": 0}},
                    "paid": {"tool_call": True, "cost": {"input": 1, "output": 0}},
                    "retired": {
                        "tool_call": True,
                        "status": "deprecated",
                        "cost": {"input": 0, "output": 0},
                    },
                    "no-tools": {"tool_call": False, "cost": {"input": 0, "output": 0}},
                }
            }
        }
        responses = [
            FakeResponse(body=json.dumps(zen).encode()),
            FakeResponse(body=json.dumps(catalog).encode()),
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models.json"
            with (
                mock.patch.object(client, "_MODEL_SNAPSHOT", None),
                mock.patch.object(client, "_model_cache_path", return_value=cache),
                mock.patch.object(client, "_urlopen", side_effect=responses) as opened,
            ):
                oc = client.OpenCodeClient()
                self.assertEqual(oc.list_models(), ["free-a", "free-b"])
                self.assertEqual(oc.list_models(), ["free-a", "free-b"])
                # The snapshot carries the catalog limits beside the ids; a legacy ids-only file
                # still reads (see test_profile.ModelCacheTests).
                self.assertEqual(
                    json.loads(cache.read_text()), {"models": ["free-a", "free-b"], "limits": {}}
                )
        self.assertEqual(
            [call.args[0].full_url for call in opened.call_args_list],
            [client.MODELS_URL, client.MODELS_DEV_URL],
        )
        self.assertTrue(
            all(
                call.args[0].headers["User-agent"] == client.user_agent()
                for call in opened.call_args_list
            )
        )

    def test_catalog_failure_uses_last_verified_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models.json"
            cache.write_text('["last-free"]')
            with (
                mock.patch.object(client, "_MODEL_SNAPSHOT", None),
                mock.patch.object(client, "_model_cache_path", return_value=cache),
                mock.patch.object(
                    client, "_urlopen", side_effect=urllib.error.URLError("offline")
                ),
            ):
                self.assertEqual(client.OpenCodeClient().list_models(), ["last-free"])


if __name__ == "__main__":
    unittest.main()
