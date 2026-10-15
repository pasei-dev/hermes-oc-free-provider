# oc-free-provider

Transport for the OpenCode free models inside Hermes. Hermes keeps the agent loop and runs every tool
locally; the plugin does model discovery and one inference request. It never starts `opencode serve`, an
ACP process, or a tool-execution proxy.

> [!IMPORTANT]
> Unofficial, and not affiliated with or endorsed by OpenCode. This is not permission to use OpenCode Zen:
> use it only where you have the right to, and do not resell or proxy it for anyone else. Free describes
> the price, not the privacy. Some free models train on prompts and completions, so keep confidential data
> out unless the model's own terms allow it. Terms:
> [opencode.ai/legal/terms-of-service](https://opencode.ai/legal/terms-of-service),
> [opencode.ai/docs/zen](https://opencode.ai/docs/zen/).

## Install

```sh
hermes plugins install <owner>/hermes-oc-free-provider --enable
hermes model        # Provider: opencode-free
```

Installs are pinned to a commit. To move an existing install, name the commit explicitly:

```sh
hermes plugins install <owner>/hermes-oc-free-provider --force --ref <40-char sha> --enable
```

Nothing here needs the OpenCode CLI, Node.js or `npx`. Discovery and inference go out over Python's
standard-library HTTPS client, so Linux, macOS and Windows take the same path.

## Which models show up

The plugin intersects OpenCode's live model list with the OpenCode catalog published by
[models.dev](https://models.dev/), and keeps only live, tool-capable models whose published input, output
and cache costs are all zero.

The intersection is cached for the gateway process and written under the active Hermes profile. A new
process queries both endpoints once; if that fails or finds nothing, the plugin uses the last good
intersection, then the models bundled with it. Each model is called over `/chat/completions` or
`/responses` depending on which transport its family uses.

Availability, pricing, retention and data-use rules belong to OpenCode and the model provider, not to this
repository, and they change without notice.

## Tool translation

Hermes stays the runtime: it applies its own policy and approval rules, runs tools locally, and gets each
result back as conversation. The plugin translates both ways, so the model sees OpenCode's tool names
while Hermes dispatches its own.

| OpenCode | Hermes |
|---|---|
| `bash` | `terminal` |
| `edit` | `patch` |
| `glob`, `grep` | `search_files` |
| `read` | `read_file` |
| `skill` | `skill_view` |
| `task` | `delegate_task` |
| `todowrite` | `todo_list` |
| `webfetch` | `web_extract` |
| `websearch` | `web_search` |
| `write` | `write_file` |

An alias is advertised only when Hermes supplied the tool behind it. If the model calls an alias whose
Hermes tool is missing, the request fails closed: nothing runs, and no success is reported. Streamed tool
arguments are buffered until their JSON is complete before translation.

Intent survives the translation; not every OpenCode option does. `bash.timeout` converts from milliseconds
to seconds, todo order is kept but OpenCode's priority label is dropped, `read` directory listings and
image attachments are not emulated, and `webfetch` and `websearch` hints with no Hermes equivalent are
discarded.

## When it stops working

The relay serves the free tier only to requests that look like the official client, and it tightens those
checks without announcing it. The usual symptom: every model 403s with *"free tier can only be used from
within OpenCode"* while the same models answer fine in the OpenCode CLI.

`AGENTS.md` carries the current wire contract (endpoint, User-Agent form, session-id encoding, the tool
schema the relay fingerprints) and the procedure for re-deriving it when a check moves.

## Verify

```sh
python -m unittest discover -s tests -v
hermes plugins validate . --json
```

`hermes plugins doctor .` also checks runtime discovery, import and provider registration.

## License

MIT.
