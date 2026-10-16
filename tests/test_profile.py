# pyright: reportArgumentType=false
# ruff: noqa: E402 -- the standalone plugin directory must be added before import.
"""The provider profile's reasoning wiring: what the relay is asked for, per model route.

A wrong answer here is a live 400 on every turn of that model, so the split between the two routes
is pinned rather than left to the client's discretion.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_plugin(name: str = "oc_free_provider_plugin", plugin_dir: Path | None = None):
    """Execute a plugin copy's ``__init__.py`` the way Hermes does: as its own package."""
    directory = plugin_dir or ROOT
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


plugin = load_plugin()

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


class CatalogLimitTests(unittest.TestCase):
    """What the context meter, the output cap and the cost line read for a free model."""

    def setUp(self):
        # The module the plugin itself loaded, not the bare `hermes_oc_client` name: the plugin
        # imports its client relative to its own package (see `__init__`).
        client = plugin.hermes_oc_client

        self.client = client
        self._saved = dict(client._CATALOG_LIMITS)
        client._CATALOG_LIMITS.clear()
        client._CATALOG_LIMITS.update(
            {
                "deepseek-v4.1-flash-free": {"context": 200000, "output": 32000},
                "muse-spark-1.3-contributor-free": {"context": 1048576, "output": 131072},
            }
        )

    def tearDown(self):
        self.client._CATALOG_LIMITS.clear()
        self.client._CATALOG_LIMITS.update(self._saved)

    def test_the_paid_siblings_window_never_leaks_in(self):
        # Hermes' family fallback reads the paid `deepseek-v4.1-flash` row (1000000) for this id.
        self.assertEqual(
            plugin.opencode_free.get_model_context_length("deepseek-v4.1-flash-free"), 200000
        )
        self.assertEqual(
            plugin.opencode_free.get_model_context_length("muse-spark-1.3-contributor-free"),
            1048576,
        )
        self.assertIsNone(plugin.opencode_free.get_model_context_length("not-in-the-catalog"))

    def test_the_output_cap_comes_from_the_catalog(self):
        self.assertEqual(
            plugin.opencode_free.get_max_tokens("deepseek-v4.1-flash-free"), 32000
        )
        self.assertIsNone(plugin.opencode_free.get_max_tokens("not-in-the-catalog"))

    def test_free_usage_prices_as_included(self):
        """The host's CostResult is imported lazily, so stand one in: the shape is the contract."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class CostResult:
            amount_usd: object
            status: str
            source: str
            label: str

        host = types.ModuleType("agent")
        host.__path__ = []  # a package stub, so the lazy `from agent.usage_pricing import …` lands
        pricing = types.ModuleType("agent.usage_pricing")
        setattr(pricing, "CostResult", CostResult)
        with mock.patch.dict(sys.modules, {"agent": host, "agent.usage_pricing": pricing}):
            cost = plugin.opencode_free.get_usage_cost("deepseek-v4.1-flash-free", None)
        self.assertEqual((cost.status, cost.label, str(cost.amount_usd)), ("included", "included", "0"))

    def test_cost_degrades_when_the_host_cannot_answer(self):
        """No host module (an older Hermes) must leave pricing to Hermes rather than raise."""
        with mock.patch.dict(sys.modules, {"agent.usage_pricing": None}):
            self.assertIsNone(
                plugin.opencode_free.get_usage_cost("deepseek-v4.1-flash-free", None)
            )

    def test_limits_extraction_keeps_only_real_numbers(self):
        catalog = {
            "opencode": {
                "models": {
                    "free-a": {"limit": {"context": 1000, "output": 100}},
                    "free-b": {"limit": {"context": 0, "output": True}},
                    "free-c": {"limit": "nonsense"},
                    "free-d": {},
                }
            }
        }
        self.assertEqual(
            self.client._free_live_limits(["free-a", "free-b", "free-c", "free-d", "absent"], catalog),
            {"free-a": {"context": 1000, "output": 100}},
        )
        self.assertEqual(self.client._free_live_limits(["free-a"], {"deepinfra": {}}), {})


class ModelCacheTests(unittest.TestCase):
    """The discovery cache carries the limits too, and still reads the older ids-only file."""

    def setUp(self):
        import hermes_oc_client as client

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = client
        patcher = mock.patch.object(
            client, "_model_cache_path", return_value=Path(self.tmp.name) / "models.json"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_limits_round_trip_through_the_cache(self):
        self.client._save_model_cache(["a-free"], {"a-free": {"context": 4096, "output": 512}})
        self.assertEqual(
            self.client._load_model_cache(),
            (["a-free"], {"a-free": {"context": 4096, "output": 512}}),
        )

    def test_a_legacy_ids_only_file_still_reads(self):
        Path(self.tmp.name, "models.json").write_text('["a-free", "a-free", "b-free"]')
        self.assertEqual(self.client._load_model_cache(), (["a-free", "b-free"], {}))


class ClientIsolationTests(unittest.TestCase):
    """One process, several profile homes: each home's copy of this plugin must stand alone.

    The live failure this pins: the gateway loads the plugin once per profile home, and the two
    copies had drifted apart. The second copy's bare ``import hermes_oc_client`` resolved to the
    first copy's already-imported module, so its ``from hermes_oc_client import catalog_limit``
    raised ImportError, the provider never registered, and the model picker listed it with no
    models.
    """

    def _write_a_drifted_copy(self, tmp: str) -> Path:
        """An older home's copy: bare client import, and a client without ``catalog_limit``."""
        stale = Path(tmp) / "oc-free-provider"
        stale.mkdir()
        (stale / "hermes_oc_client.py").write_text(
            "DEFAULT_MODEL = 'big-pickle'\n"
            "FALLBACK_MODELS = (DEFAULT_MODEL,)\n"
            "LOGICAL_BASE_URL = 'https://opencode.ai/inference/v1/chat/completions'\n"
            "class OpenCodeClient:  # no catalog_limit yet\n"
            "    pass\n",
            encoding="utf-8",
        )
        (stale / "__init__.py").write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "from hermes_oc_client import DEFAULT_MODEL, FALLBACK_MODELS, LOGICAL_BASE_URL\n",
            encoding="utf-8",
        )
        return stale

    def test_a_drifted_first_copy_does_not_break_the_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Reproduce the gateway's state: nothing has imported the client under its bare name
            # yet, so the first copy to load is the one that claims it.
            saved = sys.modules.pop("hermes_oc_client", None)

            def _restore():
                if saved is not None:
                    sys.modules["hermes_oc_client"] = saved
                else:
                    sys.modules.pop("hermes_oc_client", None)

            self.addCleanup(_restore)
            stale = self._write_a_drifted_copy(tmp)
            load_plugin("oc_free_home_old", stale)
            self.addCleanup(sys.modules.pop, "oc_free_home_old", None)
            # The drifted copy claimed the bare name, with its own client under it.
            self.assertFalse(hasattr(sys.modules["hermes_oc_client"], "catalog_limit"))

            second = load_plugin("oc_free_home_new", ROOT)
            self.addCleanup(sys.modules.pop, "oc_free_home_new", None)

            # The newer copy must not have resolved its client through that bare name.
            self.assertTrue(callable(second.catalog_limit))
            self.assertIsNot(second.hermes_oc_client, sys.modules["hermes_oc_client"])
            self.assertEqual(second.FALLBACK_MODELS, plugin.FALLBACK_MODELS)

    def test_each_copy_keeps_its_own_client_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "oc-free-provider"
            shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            other = load_plugin("oc_free_copy", copy)
            self.addCleanup(sys.modules.pop, "oc_free_copy", None)
            self.assertIsNot(other.hermes_oc_client, plugin.hermes_oc_client)
            self.assertEqual(
                other.hermes_oc_client.__file__,
                str(copy / "hermes_oc_client.py"),
            )


if __name__ == "__main__":
    unittest.main()
