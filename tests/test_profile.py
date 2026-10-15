# pyright: reportArgumentType=false
# ruff: noqa: E402 -- the standalone plugin directory must be added before import.
"""The provider profile's reasoning wiring: what the relay is asked for, per model route.

A wrong answer here is a live 400 on every turn of that model, so the split between the two routes
is pinned rather than left to the client's discretion.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location("oc_free_provider_plugin", ROOT / "__init__.py")
assert _SPEC is not None and _SPEC.loader is not None
plugin = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plugin)

CHAT_MODEL = "deepseek-v4.1-flash-free"
RESPONSES_MODEL = "muse-spark-1.3-contributor-free"


class ReasoningWiringTests(unittest.TestCase):
    def test_chat_models_carry_the_dial(self):
        self.assertEqual(
            plugin.opencode_free.build_api_kwargs_extras(
                reasoning_config={"effort": "high"}, model=CHAT_MODEL
            ),
            ({}, {"reasoning_effort": "high"}),
        )
        self.assertEqual(
            plugin.opencode_free.supported_reasoning_efforts(CHAT_MODEL),
            plugin._REASONING_EFFORTS,
        )

    def test_thinking_off_uses_the_relays_own_none(self):
        for config in ({"effort": "none"}, {"enabled": False, "effort": "high"}):
            self.assertEqual(
                plugin.opencode_free.build_api_kwargs_extras(
                    reasoning_config=config, model=CHAT_MODEL
                ),
                ({}, {"reasoning_effort": "none"}),
            )

    def test_an_internal_level_never_escalates(self):
        for effort in ("ultra", "", "  "):
            self.assertEqual(
                plugin.opencode_free.build_api_kwargs_extras(
                    reasoning_config={"effort": effort}, model=CHAT_MODEL
                ),
                ({}, {"reasoning_effort": "medium"}),
            )

    def test_responses_models_are_never_sent_the_parameter(self):
        """Muse Spark answers the parameter with HTTP 400 and streams reasoning_details anyway."""
        for config in (None, {"effort": "high"}, {"effort": "none"}):
            self.assertEqual(
                plugin.opencode_free.build_api_kwargs_extras(
                    reasoning_config=config, model=RESPONSES_MODEL
                ),
                ({}, {}),
            )
        self.assertEqual(plugin.opencode_free.supported_reasoning_efforts(RESPONSES_MODEL), ())

    def test_models_are_declared_reasoning_capable(self):
        """Without the declaration Hermes neither sends the dial nor renders the thinking blocks."""
        declared = plugin.opencode_free.model_capabilities
        self.assertTrue(declared)
        for model, caps in declared.items():
            with self.subTest(model=model):
                self.assertTrue(caps.get("supports_tools"))
                self.assertTrue(caps.get("supports_reasoning"))


if __name__ == "__main__":
    unittest.main()
