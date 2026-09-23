"""Tests for the 9Router image provider. Offline: monkeypatched transport.

Run: python3 -m pytest tests/ -q   (or the stdlib runner below)
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Minimal stubs so the provider module imports outside the Hermes tree.
for mod, attrs in {
    "agent": None,
    "agent.image_gen_provider": {
        "DEFAULT_ASPECT_RATIO": "landscape",
        "ImageGenProvider": object,
        "error_response": lambda **kw: {"success": False, "image": None, **{k: kw.get(k, "") for k in ("error", "error_type", "provider", "model", "prompt", "aspect_ratio")}},
        "resolve_aspect_ratio": lambda v: v if v in ("landscape", "square", "portrait") else "landscape",
        "save_b64_image": None,  # set below
        "save_url_image": None,  # set below
        "success_response": lambda **kw: {"success": True, **kw},
    },
}.items():
    if mod not in sys.modules:
        import types

        m = types.ModuleType(mod)
        if attrs:
            for k, v in attrs.items():
                setattr(m, k, v)
        sys.modules[mod] = m


def _install_save_stub(tmp: Path):
    def save(b64: str, *, prefix: str = "image", extension: str = "png") -> Path:
        raw = base64.b64decode(b64)
        path = tmp / f"{prefix}_test.{extension}"
        path.write_bytes(raw)
        return path

    sys.modules["agent.image_gen_provider"].save_b64_image = save

    def save_url(url: str, *, prefix: str = "image", **_kw) -> Path:
        # Cheap stub: fail fast for invalid URLs (used by the failure test below),
        # otherwise mimic caching by mapping the URL to a local path.
        if url.startswith("bad"):
            raise ValueError("injected cache failure")
        path = tmp / f"{prefix}_cached.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        return path

    setattr(sys.modules["agent.image_gen_provider"], "save_url_image", save_url)


class _Resp:
    def __init__(self, body: dict, status: int = 200):
        self._body = body
        self.status_code = status

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _install_transport(handler):
    import requests as real_requests

    class _FakeRequests:
        @staticmethod
        def request(method, url, headers=None, timeout=None, json=None):
            return handler(method, url, headers or {}, json)

    sys.modules["requests"] = _FakeRequests
    return real_requests


def _load_provider():
    # Fresh import so module-level state resets between checks.
    for name in list(sys.modules):
        if name.startswith("hermes_9router_image"):
            del sys.modules[name]
    import hermes_9router_image as mod

    return mod, mod.NineRouterImageProvider()


def main() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    _install_save_stub(tmp)

    # 1. b64_json success path
    png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()
    calls = []

    def ok_handler(method, url, headers, body):
        calls.append((method, url, body))
        if url.endswith("/v1/models/image"):
            return _Resp({"data": [{"id": "fake/model-a"}, {"id": "fake/model-b"}]})
        return _Resp({"created": 1, "data": [{"b64_json": png_b64}]})

    real = _install_transport(ok_handler)
    mod, provider = _load_provider()
    result = provider.generate("a red apple", "square", model="fake/model-a")
    assert result["success"] is True, result
    assert result["model"] == "fake/model-a"
    assert result["aspect_ratio"] == "square"
    assert result["provider"] == "9router"
    assert Path(result["image"]).exists() and Path(result["image"]).read_bytes().startswith(b"\x89PNG")
    assert result["extra_format"] if "extra_format" in result else result.get("format") in ("png", None)
    post = [c for c in calls if c[0] == "POST"][0]
    assert post[1].endswith("/v1/images/generations")
    assert post[2]["model"] == "fake/model-a"
    assert post[2]["size"] == "1024x1024"

    # 2. URL response path → cached file (not bare URL)
    def url_handler(method, url, headers, body):
        if url.endswith("/v1/models/image"):
            return _Resp({"data": [{"id": "fake/model-a"}]})
        return _Resp({"created": 1, "data": [{"url": "https://example.com/img.png"}]})

    _install_transport(url_handler)
    mod, provider = _load_provider()
    result = provider.generate("a cat", "landscape")
    assert result["success"] is True, result
    assert Path(result["image"]).exists(), result
    assert result["image"].endswith("_cached.png"), result

    # 2b. URL cache failure → explicit error, never return unsafe or expired URL
    def bad_url_handler(method, url, headers, body):
        if url.endswith("/v1/models/image"):
            return _Resp({"data": [{"id": "fake/model-a"}]})
        return _Resp({"created": 1, "data": [{"url": "bad://example.com/img.png"}]})

    _install_transport(bad_url_handler)
    mod, provider = _load_provider()
    result = provider.generate("a cat", "landscape")
    assert result["success"] is False, result
    assert result["image"] is None, result
    assert "Could not cache" in result["error"] and "bad://" not in result["error"], result

    # 3. Provider error dict path (e.g. 429 quota)
    def err_handler(method, url, headers, body):
        if url.endswith("/v1/models/image"):
            return _Resp({"data": [{"id": "fake/model-a"}]})
        return _Resp({"error": {"message": "quota exceeded", "type": "rate_limit"}})

    _install_transport(err_handler)
    mod, provider = _load_provider()
    result = provider.generate("a cat")
    assert result["success"] is False, result
    assert "quota exceeded" in result["error"]
    assert result["error_type"] == "rate_limit"

    # 4. Empty prompt rejection
    result = provider.generate("   ")
    assert result["success"] is False and result["error_type"] == "invalid_input"

    # 5. Aspect ratio clamping
    assert mod.resolve_aspect_ratio("bogus") == "landscape"
    assert mod._ASPECT_TO_SIZE["portrait"] == "1024x1792"

    # 6. Model selection precedence: kwarg > settings > env > catalog-first
    os.environ["NINEROUTER_IMAGE_MODEL"] = "fake/env-model"
    mod, provider = _load_provider()
    assert provider.default_model() == "fake/env-model"
    del os.environ["NINEROUTER_IMAGE_MODEL"]

    # 7. Quota error on primary → falls back to next model; non-quota error → no fallback
    tried = []

    def quota_then_ok(method, url, headers, body):
        tried.append(body.get("model"))
        if url.endswith("/v1/models/image"):
            return _Resp({"data": [{"id": "fake/model-a"}]})
        if body.get("model") == "fake/primary":
            return _Resp({"error": {"message": "429 quota exceeded", "type": "rate_limit"}}, status=429)
        return _Resp({"created": 1, "data": [{"url": "https://example.com/fallback.png"}]})

    _install_transport(quota_then_ok)
    mod, provider = _load_provider()
    chain = provider._model_chain("fake/primary")
    assert chain == ["fake/primary"], chain
    # simulate fallback_models config via a chain of two models
    import hermes_9router_image as m
    orig = provider._model_chain
    provider._model_chain = lambda kw: ["fake/primary", "fake/backup"]
    result = provider.generate("a cat")
    provider._model_chain = orig
    assert result["success"] is True, result
    assert result["model"] == "fake/backup", result
    assert tried == ["fake/primary", "fake/backup"], tried

    def hard_error(method, url, headers, body):
        if url.endswith("/v1/models/image"):
            return _Resp({"data": [{"id": "fake/model-a"}]})
        return _Resp({"error": {"message": "invalid prompt", "type": "invalid_request_error"}})

    _install_transport(hard_error)
    mod, provider = _load_provider()
    provider._model_chain = lambda kw: ["fake/primary", "fake/backup"]
    result = provider.generate("a cat")
    assert result["success"] is False, result
    assert result["model"] == "fake/primary", result  # no fallback on non-quota errors

    # 8. _is_quota_error shapes
    assert m._is_quota_error({"error": {"message": "429 Too Many Requests", "code": 429}})
    assert m._is_quota_error({"error": {"message": "Quota exceeded for this model"}})
    assert m._is_quota_error({"error": {"message": "rate limit reached", "code": "rate_limit_exceeded"}})
    assert not m._is_quota_error({"error": {"message": "invalid prompt"}})
    assert not m._is_quota_error(None)
    assert not m._is_quota_error({})

    sys.modules["requests"] = real
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
