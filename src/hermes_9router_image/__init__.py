"""Native Hermes text-to-image provider for a configured 9Router gateway."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)
from agent.provider_media import cache_dir, cache_path
from agent.secret_scope import get_secret
from hermes_cli.config import load_config
from hermes_constants import get_hermes_home

DEFAULT_BASE_URL = "http://localhost:20128"
DEFAULT_TIMEOUT = 180
_MODELS_CACHE_TTL = 300.0
_MAX_IMAGE_BYTES = 25 * 1024 * 1024
_MAX_JSON_BYTES = 36 * 1024 * 1024
_ASPECT_TO_SIZE = {
    "landscape": "1792x1024",
    "square": "1024x1024",
    "portrait": "1024x1792",
}


class _ProviderError(Exception):
    def __init__(self, message: str, error_type: str = "provider_error") -> None:
        super().__init__(message)
        self.error_type = error_type


def _env(name: str) -> str:
    # A scope miss or scoping error must never borrow the launch profile's env.
    return str(get_secret(name) or "").strip()


def _load_settings() -> Dict[str, Any]:
    cfg = load_config() or {}
    image = cfg.get("image_gen") if isinstance(cfg, dict) else None
    scoped = image.get("9router") if isinstance(image, dict) else None
    if scoped is not None and not isinstance(scoped, dict):
        raise ValueError("Invalid image settings")
    return dict(scoped or {})


def _base_url(settings: Dict[str, Any]) -> str:
    value = settings.get("base_url") or _env("NINEROUTER_URL") or DEFAULT_BASE_URL
    if not isinstance(value, str):
        raise ValueError("Invalid gateway URL")
    parts = urlsplit(value.strip())
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment):
        raise ValueError("Invalid gateway URL")
    parts.port  # Validate the port without including the URL in public errors.
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _timeout(settings: Dict[str, Any]) -> float:
    value = settings.get("timeout", DEFAULT_TIMEOUT)
    if isinstance(value, bool):
        raise ValueError("Invalid timeout")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Invalid timeout")
    return result


def _model(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Invalid model")
    return value.strip()


def _is_quota_error(status: int, body: Dict[str, Any]) -> bool:
    # 9Router also labels some permission failures insufficient_quota: HTTP wins.
    if status == 429:
        return True
    if not 200 <= status < 300:
        return False
    err = body.get("error")
    if not isinstance(err, dict):
        return False
    kind = str(err.get("type") or "").lower()
    if kind in ("permission_error", "authentication_error", "invalid_request_error"):
        return False
    code = str(err.get("code") or "").lower()
    return (code in ("429", "rate_limit_exceeded", "insufficient_quota") or kind in (
        "rate_limit", "rate_limit_error", "quota_exceeded",
    ) or bool(re.search(r"\bquota\b|\brate[\s_-]*limit\b|\btoo many requests\b",
                        str(err.get("message") or ""), re.IGNORECASE)))


def _request_json(
    method: str, url: str, *, headers: Dict[str, str], timeout: float,
    json_body: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, Any]]:
    import requests

    try:
        with requests.Session() as session:
            # Do not substitute netrc credentials or send configured keys via env proxies.
            session.trust_env = False
            with session.request(
                method, url, headers=headers, timeout=timeout, json=json_body,
                allow_redirects=False, stream=True,
            ) as response:
                status = response.status_code
                # No raw gateway error body is needed or exposed.
                if not 200 <= status < 300:
                    return status, {}
                raw = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    raw.extend(chunk)
                    if len(raw) > _MAX_JSON_BYTES:
                        raise _ProviderError("9Router JSON response is too large", "invalid_response")
                try:
                    body = json.loads(raw)
                except (ValueError, UnicodeError):
                    raise _ProviderError("9Router returned invalid JSON", "invalid_response") from None
                if not isinstance(body, dict):
                    raise _ProviderError("9Router returned a non-object response", "invalid_response")
                return status, body
    except _ProviderError:
        raise
    except Exception:
        # requests exceptions may contain credentials, URLs or upstream response text.
        raise _ProviderError("Could not contact the 9Router gateway", "transport_error") from None


def _image_format(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    raise _ProviderError("9Router returned unsupported image bytes", "invalid_response")


def _save_b64_image(value: str) -> Tuple[str, str]:
    if len(value) > ((_MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise _ProviderError("9Router image exceeds the 25 MB limit", "invalid_response")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise _ProviderError("9Router returned invalid image base64", "invalid_response") from None
    if not raw or len(raw) > _MAX_IMAGE_BYTES:
        raise _ProviderError("9Router image is empty or too large", "invalid_response")
    extension = _image_format(raw)
    # Native profile-scoped generated-image cache, with atomic publication.
    path = cache_path("generated/images", prefix="9router", extension=extension)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(raw)
        os.replace(temporary, path)
        return str(path), extension
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cache_url_image(value: str, timeout: float) -> Tuple[str, str]:
    # Hermes validates each URL/redirect and enforces its SSRF/size rules.
    # Never attach the configured gateway Authorization header to this URL.
    prefix = "9router_" + uuid.uuid4().hex
    directory = cache_dir("generated/images")
    path = destination = None
    try:
        path = Path(save_url_image(
            value, prefix=prefix, timeout=timeout, max_bytes=_MAX_IMAGE_BYTES,
        ))
        destination = path
        with path.open("rb") as source:
            raw = source.read(_MAX_IMAGE_BYTES + 1)
        if not raw or len(raw) > _MAX_IMAGE_BYTES:
            raise _ProviderError("9Router image is empty or too large", "invalid_response")
        extension = _image_format(raw)
        destination = path.with_suffix("." + extension)
        if destination != path:
            path.replace(destination)
        return str(destination), extension
    except Exception:
        # The native helper can fail after creating a partial file but before
        # returning its name. A unique per-call prefix confines cleanup safely.
        candidates = set(directory.glob(prefix + "_*"))
        candidates.update(item for item in (path, destination) if item is not None)
        for candidate in candidates:
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
        raise


class NineRouterImageProvider(ImageGenProvider):
    """Additional selectable provider; generation only, one requested image."""

    def __init__(self) -> None:
        self._models_cache: Dict[Tuple[str, str, str], Tuple[List[str], float]] = {}
        self._cache_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "9router"

    @property
    def display_name(self) -> str:
        return "9Router"

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def is_available(self) -> bool:
        # Passive: a local auth-free gateway needs no key and discovery makes no requests.
        try:
            import requests  # noqa: F401
            _base_url(_load_settings())
            _env("NINEROUTER_KEY")
            return True
        except Exception:
            return False

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "9Router",
            "tag": "Text-to-image via a configured 9Router gateway",
            "env_vars": [{
                "key": "NINEROUTER_KEY",
                "prompt": "9Router API key (skip for an auth-free gateway)",
            }],
        }

    def _fetch_models(self, base: str, key: str) -> List[str]:
        scope = (str(get_hermes_home()), base, hashlib.sha256(key.encode()).hexdigest())
        now = time.monotonic()
        with self._cache_lock:
            cached = self._models_cache.get(scope)
            if cached is not None and now - cached[1] < _MODELS_CACHE_TTL:
                return list(cached[0])
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            status, body = _request_json(
                "GET", f"{base}/v1/models/image", headers=headers, timeout=15,
            )
            if not 200 <= status < 300 or body.get("error"):
                return []
            entries = body.get("data")
            if not isinstance(entries, list):
                return []
            models = list(dict.fromkeys(
                item["id"].strip() for item in entries
                if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
            ))
            with self._cache_lock:
                if len(self._models_cache) >= 16:
                    self._models_cache.clear()
                self._models_cache[scope] = (models, time.monotonic())
            return models
        except Exception:
            return []  # Catalog is optional; no raw exception/URL/credential logging.

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            settings = _load_settings()
            models = self._fetch_models(_base_url(settings), _env("NINEROUTER_KEY"))
            return [{"id": model, "display": model, "strengths": "via 9Router"} for model in models]
        except Exception:
            return []

    def default_model(self) -> Optional[str]:
        try:
            settings = _load_settings()
            primary = _model(settings.get("model")) or _env("NINEROUTER_IMAGE_MODEL")
            if primary:
                return primary
            models = self._fetch_models(_base_url(settings), _env("NINEROUTER_KEY"))
            return models[0] if models else None
        except Exception:
            return None

    def _model_chain(
        self, override: Any, settings: Dict[str, Any], base: str, key: str,
    ) -> List[str]:
        primary = _model(override) or _model(settings.get("model")) or _env("NINEROUTER_IMAGE_MODEL")
        if not primary:
            models = self._fetch_models(base, key)
            primary = models[0] if models else ""
        if not primary:
            return []
        fallbacks = settings.get("fallback_models") or []
        if isinstance(fallbacks, str):
            fallbacks = [fallbacks]
        if not isinstance(fallbacks, (list, tuple)):
            raise ValueError("Invalid fallback models")
        return list(dict.fromkeys([primary] + [model for value in fallbacks if (model := _model(value))]))

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        aspect = resolve_aspect_ratio(aspect_ratio)
        text = prompt.strip() if isinstance(prompt, str) else ""
        model_id = ""

        def fail(message: str, kind: str = "provider_error") -> Dict[str, Any]:
            return error_response(
                error=message, error_type=kind, provider=self.name,
                model=model_id, prompt=text, aspect_ratio=aspect,
            )

        if not text:
            return fail("Prompt is required", "invalid_input")
        if (isinstance(image_url, str) and image_url.strip()) or normalize_reference_images(reference_image_urls):
            return fail("9Router image provider supports text-to-image only; omit source/reference images",
                        "modality_unsupported")
        try:
            settings = _load_settings()
            base, key = _base_url(settings), _env("NINEROUTER_KEY")
            timeout = _timeout(settings)
            chain = self._model_chain(kwargs.get("model"), settings, base, key)
        except Exception:
            return fail("Could not resolve 9Router settings or profile credentials", "configuration_error")
        if not chain:
            return fail("No 9Router image model configured or available in the gateway catalog")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        for index, model_id in enumerate(chain):
            try:
                status, body = _request_json(
                    "POST", f"{base}/v1/images/generations", headers=headers, timeout=timeout,
                    json_body={"model": model_id, "prompt": text, "size": _ASPECT_TO_SIZE[aspect], "n": 1},
                )
                if not 200 <= status < 300 or "error" in body:
                    quota = _is_quota_error(status, body)
                    if quota and index + 1 < len(chain):
                        continue
                    kind = ("rate_limit" if quota else "auth_required" if status == 401 else
                            "permission_error" if status == 403 else "provider_error")
                    message = f"9Router returned HTTP {status}" if not 200 <= status < 300 else "9Router reported an image generation error"
                    return fail(message, kind)
                data = body.get("data")
                if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                    return fail("9Router returned no image data", "invalid_response")
                entry = data[0]
                if isinstance(entry.get("b64_json"), str) and entry["b64_json"]:
                    image, extension = _save_b64_image(entry["b64_json"])
                elif isinstance(entry.get("url"), str) and entry["url"]:
                    try:
                        image, extension = _cache_url_image(entry["url"], timeout)
                    except Exception:
                        return fail("Could not cache a valid 9Router image URL", "image_cache_error")
                else:
                    return fail("9Router response has neither url nor b64_json", "invalid_response")
                return success_response(
                    image=image, model=model_id, prompt=text, aspect_ratio=aspect,
                    provider=self.name, extra={"format": extension},
                )
            except _ProviderError as exc:
                return fail(str(exc), exc.error_type)
            except Exception:
                return fail("Could not save the generated 9Router image", "image_cache_error")
        return fail("9Router request failed")


def register(ctx) -> None:
    ctx.register_image_gen_provider(NineRouterImageProvider())
