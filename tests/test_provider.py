"""Offline regressions against real Hermes stable0.23.2; never stub the host ABC.

Run only in an approved development environment:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.image_gen_provider import ImageGenProvider
from agent.secret_scope import (
    reset_multiplex_context, reset_secret_scope, set_multiplex_context, set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
import hermes_9router_image as plugin

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jf1kAAAAASUVORK5CYII="
)
IMAGE = {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}


class Response:
    def __init__(self, status=200, body=None, raw=None):
        self.status_code = status
        self.raw = raw if raw is not None else json.dumps(body).encode()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def iter_content(self, chunk_size):
        yield self.raw


class Session:
    def __init__(self, response):
        self.response = response
        self.trust_env = True
        self.calls = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class ProviderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        home_token = set_hermes_home_override(self.home)
        self.addCleanup(reset_hermes_home_override, home_token)
        multiplex_token = set_multiplex_context(True)
        self.addCleanup(reset_multiplex_context, multiplex_token)
        secret_token = set_secret_scope({}, profile_home=str(self.home))
        self.addCleanup(reset_secret_scope, secret_token)
        self.settings = {"model": "provider/primary"}
        config = patch.object(plugin, "load_config", side_effect=lambda: {
            "image_gen": {"provider": "9router", "9router": self.settings},
        })
        config.start()
        self.addCleanup(config.stop)
        self.provider = plugin.NineRouterImageProvider()

    def generate(self, **kwargs):
        return self.provider.generate("a test image", **kwargs)

    def test_native_contract_and_passive_availability(self):
        self.assertIsInstance(self.provider, ImageGenProvider)
        self.assertEqual(self.provider.capabilities(), {
            "modalities": ["text"], "max_reference_images": 0,
        })
        with patch.object(plugin, "_request_json", side_effect=AssertionError("network")):
            self.assertTrue(self.provider.is_available())

    def test_native_registration_unload_reload(self):
        from agent import image_gen_registry
        from hermes_cli.plugins import PluginManager, PluginManifest

        manager = PluginManager(scope_key=str(self.home))
        manifest = PluginManifest(
            name="9router", version="0.1.1", kind="backend",
            source="user", path=str(ROOT),
        )
        self.addCleanup(manager.unload, "9router")
        for _ in range(2):
            manager._load_plugin(manifest)
            loaded = image_gen_registry.get_provider("9router", scope=manager.scope_key)
            self.assertIsInstance(loaded, ImageGenProvider)
            self.assertEqual(loaded.name, "9router")
            manager.unload("9router")
            self.assertIsNone(image_gen_registry.get_provider("9router", scope=manager.scope_key))

    def test_b64_saved_in_profile_cache_with_matching_extension(self):
        with patch.object(plugin, "_request_json", return_value=(200, IMAGE)) as request:
            result = self.generate(aspect_ratio="square", unknown_future_option=True)
        self.assertTrue(result["success"], result)
        image = Path(result["image"])
        self.assertEqual(image.read_bytes(), PNG)
        self.assertTrue(image.is_relative_to(self.home / "cache/generated/images"))
        self.assertEqual(image.suffix, ".png")
        self.assertEqual(result["modality"], "text")
        payload = request.call_args.kwargs["json_body"]
        self.assertEqual(payload, {
            "model": "provider/primary", "prompt": "a test image", "size": "1024x1024", "n": 1,
        })

    def test_edit_and_invalid_prompt_fail_before_network(self):
        with patch.object(plugin, "_request_json") as request:
            for kwargs in ({"image_url": "https://example.invalid/source.png"},
                           {"reference_image_urls": ["source.png"]}):
                self.assertEqual(self.generate(**kwargs)["error_type"], "modality_unsupported")
            self.assertEqual(self.provider.generate(None)["error_type"], "invalid_input")
        request.assert_not_called()

    def test_only_one_primary_then_explicit_fallbacks(self):
        self.settings["fallback_models"] = ["provider/backup", "provider/backup"]
        with patch.dict(os.environ, {"NINEROUTER_IMAGE_MODEL": "wrong/env"}):
            chain = self.provider._model_chain(
                "provider/override", self.settings, "https://gateway.invalid", "",
            )
        self.assertEqual(chain, ["provider/override", "provider/backup"])

    def test_empty_catalog_does_not_promote_fallback_to_primary(self):
        self.settings = {"fallback_models": ["provider/backup"]}
        with patch.object(plugin, "_request_json", return_value=(200, {"data": []})) as request:
            result = self.generate()
        self.assertFalse(result["success"])
        self.assertTrue(all(call.args[0] == "GET" for call in request.call_args_list))

    def test_429_fallback_and_403_stop(self):
        self.settings["fallback_models"] = ["another-provider/backup"]
        with patch.object(plugin, "_request_json", side_effect=[(429, {}), (200, IMAGE)]) as request:
            result = self.generate(model="provider/override")
        self.assertTrue(result["success"], result)
        self.assertEqual([call.kwargs["json_body"]["model"] for call in request.call_args_list],
                         ["provider/override", "another-provider/backup"])
        secret_body = {"error": {"type": "permission_error", "code": "insufficient_quota",
                                  "message": "secret-token"}}
        with patch.object(plugin, "_request_json", return_value=(403, secret_body)) as request:
            result = self.generate()
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result["error_type"], "permission_error")
        self.assertNotIn("secret-token", json.dumps(result))
        self.assertNotIn("_quota_probe", result)

    def test_structured_quota_fallback_on_402_and_503(self):
        self.settings["fallback_models"] = ["another-provider/backup"]
        quota = {"error": {"code": "insufficient_quota", "message": "secret-quota-detail"}}
        for status in (402, 503):
            with self.subTest(status=status):
                with patch.object(plugin, "_request_json", side_effect=[(status, quota), (200, IMAGE)]):
                    result = self.generate()
                self.assertTrue(result["success"], result)
                self.assertEqual(result["model"], "another-provider/backup")
                self.assertNotIn("secret-quota-detail", json.dumps(result))

    def test_narrow_size_schema_retry_reports_omitted_parameters(self):
        rejected = {"error": {"message": "Additional properties are not allowed ('size', 'n' were unexpected)"}}
        with patch.object(plugin, "_request_json", side_effect=[(400, rejected), (200, IMAGE)]) as request:
            result = self.generate()
        self.assertTrue(result["success"], result)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args.kwargs["json_body"], {
            "model": "provider/primary", "prompt": "a test image",
        })
        self.assertEqual(result["parameters_omitted"], ["size", "n"])
        self.assertTrue(result["aspect_ratio_advisory"])
        self.assertTrue(plugin._size_schema_rejected(
            422, {"error": {"code": "unsupported_parameter", "param": "size"}},
        ))
        for body in (
            {"error": {"message": "model not allowed"}},
            {"error": {"message": "Unknown parameter: prompt"}},
            {"error": {"message": "Additional properties are not allowed ('size', 'prompt' were unexpected)"}},
            {"error": {"type": "permission_error", "code": "unsupported_parameter", "param": "size"}},
        ):
            self.assertFalse(plugin._size_schema_rejected(400, body))
        self.assertFalse(plugin._size_schema_rejected(403, rejected))
        with patch.object(plugin, "_request_json", return_value=(400, rejected)) as request:
            self.assertFalse(self.generate()["success"])
            self.assertEqual(request.call_count, 2)  # At most one bare retry.

    def test_no_schema_retry_and_malformed_data(self):
        for status, body in [(400, {"error": {"message": "size not allowed secret"}}),
                             (500, IMAGE), (200, {"data": {"0": IMAGE["data"][0]}})]:
            with self.subTest(status=status, body=body):
                with patch.object(plugin, "_request_json", return_value=(status, body)) as request:
                    self.assertFalse(self.generate()["success"])
                    self.assertEqual(request.call_count, 1)

    def test_invalid_timeout_and_scoping_failure_are_safe(self):
        self.settings["timeout"] = "secret-invalid-timeout"
        with patch.object(plugin, "_request_json") as request:
            result = self.generate()
        self.assertEqual(result["error_type"], "configuration_error")
        self.assertNotIn("secret-invalid-timeout", json.dumps(result))
        request.assert_not_called()
        self.settings["timeout"] = 180
        with patch.object(plugin, "get_secret", side_effect=RuntimeError("secret-scope-text")):
            with patch.object(plugin, "_request_json") as request:
                self.assertFalse(self.provider.is_available())
                result = self.generate()
        self.assertNotIn("secret-scope-text", json.dumps(result))
        request.assert_not_called()

    def test_profile_env_miss_never_borrows_process_values(self):
        with patch.dict(os.environ, {
            "NINEROUTER_KEY": "wrong-key", "NINEROUTER_URL": "https://wrong.invalid",
            "NINEROUTER_IMAGE_MODEL": "wrong/model",
        }):
            self.assertEqual(plugin._env("NINEROUTER_KEY"), "")
            self.assertEqual(plugin._base_url({}), plugin.DEFAULT_BASE_URL)
            with patch.object(plugin, "_request_json", return_value=(200, IMAGE)) as request:
                self.assertTrue(self.generate()["success"])
            self.assertNotIn("Authorization", request.call_args.kwargs["headers"])

    def test_catalog_cache_separates_credential_url_and_profile(self):
        catalog = {"data": [{"id": "provider/model"}]}
        with patch.object(plugin, "_request_json", return_value=(200, catalog)) as request:
            self.provider._fetch_models("https://one.invalid", "key-a")
            self.provider._fetch_models("https://one.invalid", "key-a")
            self.provider._fetch_models("https://one.invalid", "key-b")
            self.provider._fetch_models("https://two.invalid", "key-b")
            other_token = set_hermes_home_override(self.home / "other")
            try:
                self.provider._fetch_models("https://two.invalid", "key-b")
            finally:
                reset_hermes_home_override(other_token)
        self.assertEqual(request.call_count, 4)

    def test_base64_signatures_extensions_and_atomic_cleanup(self):
        for raw, extension in ((PNG, "png"), (b"\xff\xd8\xfftest", "jpg"),
                               (b"RIFFxxxxWEBPtest", "webp"), (b"GIF89atest", "gif")):
            path, actual = plugin._save_b64_image(base64.b64encode(raw).decode())
            self.assertEqual(actual, extension)
            self.assertEqual(Path(path).suffix, "." + extension)
        for value in ("not base64!", base64.b64encode(b"<html>error</html>").decode()):
            with self.assertRaises(plugin._ProviderError):
                plugin._save_b64_image(value)
        before = set(self.home.rglob("*"))
        with patch.object(plugin.os, "replace", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                plugin._save_b64_image(base64.b64encode(PNG).decode())
        self.assertEqual(set(self.home.rglob("*")), before)

    def test_url_helper_has_no_gateway_auth_and_validates_bytes(self):
        cached = self.home / "cached.png"
        cached.write_bytes(b"\xff\xd8\xfftest")
        with patch.object(plugin, "save_url_image", return_value=cached) as save:
            path, extension = plugin._cache_url_image("https://cdn.invalid/image?secret", 10)
        self.assertEqual(extension, "jpg")
        self.assertEqual(Path(path).suffix, ".jpg")
        self.assertNotIn("headers", save.call_args.kwargs)
        cached.write_bytes(b"<html>failure</html>")
        with patch.object(plugin, "save_url_image", return_value=cached):
            with self.assertRaises(plugin._ProviderError):
                plugin._cache_url_image("https://cdn.invalid/image", 10)
        self.assertFalse(cached.exists())

    def test_partial_url_download_is_cleaned_without_touching_other_calls(self):
        directory = self.home / "cache/generated/images"
        directory.mkdir(parents=True)
        other = directory / "9router_another_call.png"
        other.write_bytes(PNG)

        def interrupted(url, *, prefix, **kwargs):
            (directory / (prefix + "_partial.png")).write_bytes(b"partial")
            raise OSError("secret download URL")

        with patch.object(plugin, "save_url_image", side_effect=interrupted):
            with self.assertRaises(OSError):
                plugin._cache_url_image("https://cdn.invalid/image", 10)
        self.assertEqual(list(directory.iterdir()), [other])

    def test_http_closes_resources_blocks_redirects_and_netrc(self):
        for response in (Response(status=429, raw=b"not json"),
                         Response(status=302, raw=b"redirect"),
                         Response(status=503, body={"error": {"code": "insufficient_quota"}}),
                         Response(body=IMAGE)):
            session = Session(response)
            with patch("requests.Session", return_value=session):
                status, body = plugin._request_json(
                    "POST", "https://gateway.invalid/v1/images/generations",
                    headers={"Authorization": "Bearer fake-key"}, timeout=10,
                )
            self.assertEqual(status, response.status_code)
            if status == 503:
                self.assertEqual(body["error"]["code"], "insufficient_quota")
            self.assertFalse(session.trust_env)
            self.assertFalse(session.calls[0][1]["allow_redirects"])
            self.assertTrue(session.closed and response.closed)
        for raw in (b"[]", b"not json"):
            with patch("requests.Session", return_value=Session(Response(raw=raw))):
                with self.assertRaises(plugin._ProviderError):
                    plugin._request_json("GET", "https://gateway.invalid", headers={}, timeout=10)


if __name__ == "__main__":
    unittest.main()
