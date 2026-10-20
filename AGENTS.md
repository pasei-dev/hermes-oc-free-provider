# Working on oc-free-provider

## Overview

A Hermes model-provider plugin for the OpenCode free tier: model discovery plus the one inference request
Hermes makes per turn. `README.md` is for the person installing it; this file is for changing it.

## Map

| Path | Why it matters |
|---|---|
| `hermes_oc_client.py` | The whole client — wire envelope, SSE decoding, tool translation, discovery |
| `__init__.py` | Registers the `opencode-free` provider profile with Hermes |
| `plugin.yaml` | Manifest; `version` is the number to bump |
| `tests/test_client.py` | The suite; it mocks `_urlopen`, so it checks the envelope and not the network |

The symbols a change usually touches: `_inference_request` and `_sse_events` (the only place a request is
built and read), `_session_id` (the relay-validated id), `_wire_tools` and `_translate_tool` (the
translation pairs, both directions), the `COMPAT_TOOL_*` alias tables, `_models_for_process` (discovery
with its cache and fallback chain), and the handshake pair `client_version` (what the User-Agent claims,
derived) with `_refusal_recovery` / `_record_refusal` / `_probe_identical` (what a refusal re-derives,
what it leaves behind, and the same-process re-open on a fresh connection that either fixes the turn or
names the restart).

## Commands

From the repo root. Both verified (exit 0, 2026-10-15):

```sh
python -m unittest discover -s tests -v
hermes plugins validate . --json
```

Any Python ≥3.10 works, with the standard library only; `hermes_constants` is supplied by the host when
Hermes loads the plugin, so the suite imports it from there.

The suite never opens a socket, so it cannot tell a working envelope from a stale one. The live check is
the one that can:

```sh
hermes -z 'Reply exactly OK' --provider opencode-free --model deepseek-v4.1-flash-free
```

It must print `OK`. Any model in `FALLBACK_MODELS` works; the `muse-spark-*-contributor-free` ones need
Hermes' own data-training acknowledgement first.

**A code change is not live until the backend restarts.** A gateway imports this plugin once, at process
start, and re-imports nothing afterwards — not even when an install reports reloading it. A stale process
looks exactly like a broken fix: the old failure survives code that is correct on disk. One gateway per
host serves every profile, and `hermes -z` is a *client* of it (`hermes_cli/gateway_chat.py`): the
acceptance check below therefore runs the gateway's in-memory module, never the file you just edited, so
restart the host gateway before concluding anything from it.

The same trap runs the other way, and it is the expensive one: a host that has been up for hours can be
refused while a fresh process streams *byte-identical* requests in the same minute. Before bisecting the
wire, replay the failing turn's own `request_dump_local-*.json` body through the plugin's client in a fresh
process — if that streams `200`, the envelope is right and the host process is what to restart. A refusal
writes itself down for exactly this: `<hermes home>/cache/oc-free-provider/refusals.jsonl` holds the last
20, each with its status, the relay's own words, the refused request's *shape* (URL, headers, model, tool
names, byte count, body hash — never the conversation), the *response's* own headers and the connection
that carried them (local/peer address, TLS version and cipher, ALPN, whether the session was reused), and a
`probe` verdict. `probe.status: 200` says those *same bytes* were served when the client re-opened them on
a fresh connection of its own a moment later — so the envelope is right and the **long-lived host process**
is what to restart; `403` says the gate refused the identical bytes twice, which is the relay's answer that
minute. The re-open is a fix and a diagnostic at once: when it is served, that stream *is* the turn. Set
`HERMES_OC_DEBUG=<file>` for the full bytes, or leave a file at
`<hermes home>/cache/oc-free-provider/capture` to arm the same capture from disk in a host whose environment
cannot be changed after it starts.

## Git

- One change per branch off a freshly fetched `main`; open a pull request rather than pushing `main`.
- Bump `version` in `plugin.yaml` in the same change as the code, last digit only.
- An install pins a commit, so moving one takes `--force --ref <40-char sha>`.
- Treat the repo as if it were public: no local paths, accounts or provenance notes in a tracked file.

## Boundaries

Always

- Prove a wire change against the live relay. The suite mocking `_urlopen` is what lets a stale envelope
  stay green.
- Keep the request in one place. A second place that builds it is how the header set drifts apart.
- Load the client through `__init__._load_client()`. Hermes loads one copy of this plugin per profile
  home into one process (one gateway serves every profile), so a bare `import hermes_oc_client` lets the
  copies drift into each other's namespace — the second copy then imports the first copy's file, its
  registration raises, and the model picker shows the provider with no models.

Ask first

- Adding a dependency. There are none by design, so the plugin runs on any host Hermes runs on.
- Changing `FALLBACK_MODELS` or `RESPONSES_MODELS` — they decide what a session can pick with no discovery.

Never

- Publish to the upstream repository. This repo is a fork, and being upstreamable is not permission.
- Copy the official client's User-Agent. It sends `opencode/latest/<ver>/cli` and the relay refuses it.
- Send anything but the keyless `Bearer public` credential, and never print a credential.
- Give a compatibility marker an empty parameter object — see the tool rule below.

## The relay's checks

Undocumented, enforced per request, and moved without notice. All six were verified against the live relay,
items 1–4 on 2026-10-15, item 5 on 2026-10-17 and item 6 on 2026-10-20:

1. Base URL `https://opencode.ai/inference/v1` for `/chat/completions`, `/responses` and `/models`. The
   older `/zen/v1` answers `FreeTierError` for every shape. `models.opencode.ai/api.json` publishes
   `.../inference/openai/v1` for the same provider — the AI-SDK base, which serves a free-tier call today
   but is not the path verified end to end, so it is deliberately not adopted.
2. `User-Agent: opencode/<semver>`, bare. Extra segments fail. The number is *derived*
   (`client_version`), never just the pinned constant: the highest of the pin, the last resolved value, the
   installed official client's own version, and any floor the relay announced with a `426`. A semver below
   the relay's floor is a separate refusal, and a pinned number is a latent break.
3. `x-opencode-session: ses_` then the low 48 bits of `~(ms * 4096 + counter)` as 6 big-endian bytes in
   hex, then 14 base62 characters. The inversion is what makes ids sort newest-first; without it the relay
   refuses the id. Rebuilt only on a retry after a refusal — an id a client reuses keeps the relay's prompt
   cache warm.
4. Every advertised tool carries a non-empty parameter schema. An alias whose Hermes tool is missing still
   needs OpenCode's real schema, which is why `_compat_tool` uses `COMPAT_TOOL_PARAMETERS`.
5. The tool list is never empty: a body that advertises no tools is refused `403 FreeTierError`, identically
   to a malformed one — the two are the same refusal, which is what makes a tool-less body the cheapest way
   to *measure* the gate without burning a real turn.
6. The window it enforces is the relay's own, not the model's models.dev row: measured 2026-10-20, a
   `deepseek-v4.1-flash-free` prompt of 418,894 tokens is served and a few dozen tokens more is refused
   `400 context_length_exceeded`, where the catalog row publishes 200,000. The ceiling counts the prompt
   alone, so `max_tokens` does not eat into it, and it is what `SERVED_CONTEXT` records (`served_context()`
   walks past the catalog row). A `503 Endpoint is unavailable` is a flaky upstream, not a length verdict:
   the same bytes are served on the next attempt, so retry before believing any measurement from it.

The reasoning dial is the relay's vocabulary too, and it moves in both directions: measured 2026-10-17,
`none` (thinking off for real), `minimal`, `low`, `medium`, `high`, `max`, `xhigh` and `ultra` all stream,
where an earlier pass had measured five of them and dropped the rest — including the host's own
thinking-off `none`, which silently kept thinking. So the dial is forwarded as it is and a
`400 [unsupported_value]` is what teaches us a model that has narrowed: one retry without it, remembered
per model in `handshake.json`. A refusal from any of these causes is re-derived once inside the call
(`_refusal_recovery`); a `403 FreeTierError` instead re-opens the *identical* bytes once on a connection of
its own (`_probe_identical`) and, when that is served, continues the turn on it. What is deliberately not
done is anything past that: a served re-open is the fix, a refused one is the relay's answer, and beyond it
the gate refuses identical bytes in bursts — a retry loop is Hermes' policy, not this client's business.

To re-derive these after a change, capture rather than guess: point the official CLI's provider `baseURL`
at a local catch-proxy through `OPENCODE_CONFIG`, record the request it sends, then replay it against the
live relay mutating one field at a time. `200` passes, `403` is the gate.
