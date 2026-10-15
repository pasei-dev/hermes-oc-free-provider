"""Hermes model-provider registration for OpenCode."""

# ruff: noqa: E402, I001 -- Hermes imports plugin directories without a package name.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PLUGIN_DIR = Path(__file__).parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from hermes_oc_client import (  # type: ignore[import-not-found]
    DEFAULT_MODEL,
    FALLBACK_MODELS,
    LOGICAL_BASE_URL,
    OpenCodeClient,
)
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
    from hermes_oc_client import RESPONSES_MODELS

    return bool(model) and str(model) not in RESPONSES_MODELS


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
    # ``supports_reasoning`` is what makes Hermes send the dial and render the thinking blocks at
    # all; the per-model vocabulary comes from ``supported_reasoning_efforts`` below.
    model_capabilities={
        model: {"supports_tools": True, "supports_reasoning": True}
        for model in FALLBACK_MODELS
    },
)

register_provider(opencode_free)
