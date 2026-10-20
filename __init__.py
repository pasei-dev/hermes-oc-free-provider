"""Hermes model-provider registration for OpenCode."""

# ruff: noqa: E402 -- the client module is imported below, before its names are used.

from __future__ import annotations

import sys
from typing import Any

# Import the client *relative to this file*, never as a bare ``hermes_oc_client``. Hermes loads one
# copy of a user plugin per profile home into a single process (one gateway serves every profile),
# so a bare name makes the copies share one ``sys.modules`` entry: the second copy to load silently
# imports the *first* copy's file. Once the copies drift apart that is fatal — the newer copy's
# ``from hermes_oc_client import catalog_limit`` raised ImportError, so the provider never
# registered and the model picker showed it with no models. Hermes imports a plugin directory as a
# package (``submodule_search_locations`` on the plugin dir), which is what lets this resolve to
# this copy's own file.
from . import hermes_oc_client  # noqa: E402

DEFAULT_MODEL = hermes_oc_client.DEFAULT_MODEL
FALLBACK_MODELS = hermes_oc_client.FALLBACK_MODELS
LOGICAL_BASE_URL = hermes_oc_client.LOGICAL_BASE_URL
OpenCodeClient = hermes_oc_client.OpenCodeClient
catalog_limit = hermes_oc_client.catalog_limit
served_context = hermes_oc_client.served_context

from providers import register_provider  # type: ignore[import-not-found]
from providers.base import ProviderProfile  # type: ignore[import-not-found]


#: What the relay accepts as a top-level ``reasoning_effort`` on the chat route. Verified against
#: the live relay (2026-10-15): ``minimal``/``low``/``medium``/``high`` all serve, and ``none``
#: turns thinking off for real. The responses route rejects the parameter outright.
_REASONING_EFFORTS = ("minimal", "low", "medium", "high")


def _takes_reasoning_effort(model: str | None) -> bool:
    """Whether *model* can be sent ``reasoning_effort``.

    The Muse Spark free models ride the responses route, which answers HTTP 400
    ("unknown parameter `reasoning_effort`") for it; they stream their reasoning as
    ``reasoning_details`` without being asked.
    """
    return bool(model) and str(model) not in hermes_oc_client.RESPONSES_MODELS


class OpenCodeFreeProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> OpenCodeClient:
        return OpenCodeClient(**client_kwargs)

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        """``()`` for the models the relay refuses the parameter on, else the accepted levels."""
        return _REASONING_EFFORTS if _takes_reasoning_effort(model) else ()

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Emit the reasoning dial as a top-level ``reasoning_effort`` kwarg.

        Top level and not ``extra_body.reasoning``: that shape's emission is gated by a host
        allowlist a third-party plugin must not edit. ``none`` is the relay's own thinking-off.
        """
        if not _takes_reasoning_effort(model):
            return {}, {}
        config = reasoning_config if isinstance(reasoning_config, dict) else {}
        effort = str(config.get("effort") or "").strip().lower()
        if config.get("enabled") is False or effort == "none":
            return {}, {"reasoning_effort": "none"}
        # Never escalate: an unknown or Hermes-internal level (``ultra``) becomes the default.
        return {}, {"reasoning_effort": effort if effort in _REASONING_EFFORTS else "medium"}

    def get_model_context_length(self, model: str | None) -> int | None:
        """The window the relay actually serves for this free model.

        Hermes cannot resolve this provider name against the catalog, and its family fallback is
        wrong for these ids — it reads the paid sibling's window (1M for `deepseek-v4.1-flash-free`,
        256K default for `big-pickle`). This hook is checked before those guesses, so the meter and
        the compressor agree with what the official OpenCode routes show.

        Where the relay enforces more than the free tier's published row, the measured window wins
        (see ``SERVED_CONTEXT``): `deepseek-v4.1-flash-free` serves a prompt of 418,894 tokens
        although models.dev publishes 200,000, so the published row would compact a turn at less
        than half its real room.
        """
        return served_context(model) or catalog_limit(model, "context")

    def get_max_tokens(self, model: str | None) -> int | None:
        """The catalog's output cap for the model; the relay serves no limit metadata of its own."""
        return catalog_limit(model, "output") or self.default_max_tokens

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        """The free tier bills nothing: report the route as included instead of unknown."""
        try:
            from agent.usage_pricing import CostResult
        except ImportError:  # Older host: leave pricing to Hermes' own fallbacks.
            return None
        from decimal import Decimal

        return CostResult(
            amount_usd=Decimal(0), status="included", source="none", label="included"
        )

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 15.0,
    ) -> list[str] | None:
        client = OpenCodeClient(
            api_key=api_key,
            base_url=base_url or self.base_url,
        )
        try:
            return client.list_models(timeout=timeout) or None
        except Exception:  # noqa: BLE001 -- catalog hooks are required to fail soft.
            return None
        finally:
            client.close()


opencode_free = OpenCodeFreeProfile(
    name="opencode-free",
    aliases=("oc-free",),
    display_name="OpenCode Free",
    description="Unofficial OpenCode free-model compatibility with Hermes-local tools",
    api_mode="chat_completions",
    auth_type="external_process",
    env_vars=(),
    base_url=LOGICAL_BASE_URL,
    # Hermes 0.21 routes credential-free custom clients through this seam. The
    # executable is Hermes' own Python; the provider never spawns it.
    process_command=sys.executable,
    process_command_env_vars=(),
    fallback_models=FALLBACK_MODELS,
    default_aux_model=DEFAULT_MODEL,
    # The relay's free-tier refusal is transient (see ``hermes_oc_client.classify_api_error``), and a
    # generic 403 verdict would end the turn on it. Passed as a constructor argument, not defined as
    # a method: ``classify_api_error`` is a dataclass *field* on ``ProviderProfile``, so a subclass
    # method of that name is consumed as the field's default and the call would land unbound.
    classify_api_error=hermes_oc_client.classify_api_error,
    # ``supports_reasoning`` is what makes Hermes send the dial and render the thinking blocks at
    # all; the per-model vocabulary comes from ``supported_reasoning_efforts`` below.
    model_capabilities={
        model: {"supports_tools": True, "supports_reasoning": True}
        for model in FALLBACK_MODELS
    },
)

register_provider(opencode_free)
