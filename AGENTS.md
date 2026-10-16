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
translation pairs, both directions), the `COMPAT_TOOL_*` alias tables, and `_models_for_process`
(discovery with its cache and fallback chain).

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

Undocumented, enforced per request, and moved without notice. All four were verified against the live relay
on 2026-10-15:

1. Base URL `https://opencode.ai/inference/v1` for `/chat/completions`, `/responses` and `/models`. The
   older `/zen/v1` answers `FreeTierError` for every shape.
2. `User-Agent: opencode/<semver>`, bare. Extra segments fail.
3. `x-opencode-session: ses_` then the low 48 bits of `~(ms * 4096 + counter)` as 6 big-endian bytes in
   hex, then 14 base62 characters. The inversion is what makes ids sort newest-first; without it the relay
   refuses the id.
4. Every advertised tool carries a non-empty parameter schema. An alias whose Hermes tool is missing still
   needs OpenCode's real schema, which is why `_compat_tool` uses `COMPAT_TOOL_PARAMETERS`.

To re-derive these after a change, capture rather than guess: point the official CLI's provider `baseURL`
at a local catch-proxy through `OPENCODE_CONFIG`, record the request it sends, then replay it against the
live relay mutating one field at a time. `200` passes, `403` is the gate.
