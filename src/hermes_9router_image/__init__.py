"""9Router image-generation backend for Hermes Agent.

Text-to-image via a 9Router gateway (``POST {base_url}/v1/images/generations``).
Model selection: ``model`` kwarg → ``image_gen.9router.model`` → ``NINEROUTER_IMAGE_MODEL``
→ first model from the live ``/v1/models/image`` catalog (cached, TTL 300s).

Config (``image_gen.9router`` in config.yaml, non-secret):
    base_url   — gateway URL (default: ``NINEROUTER_URL`` env or ``http://localhost:20128``)
    model      — default model id
    timeout    — per-request timeout seconds (default 180)

Secret: ``NINEROUTER_KEY`` (.env). Auth-free local gateways work without it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:20128"
DEFAULT_TIMEOUT = 180
_MODELS_CACHE_TTL = 300.0


class _SchemaRejected(Exception):
    """Upstream rejected the payload schema (e.g. `size`/`n` not allowed for the model)."""


_ASPECT_TO_SIZE = {
    "landscape": "1792x1024",
    "square": "1024x1024",
    "portrait": "1024x1792",
}


def _http():
    # ponytail: requests is part of the Hermes runtime env; is_available() gates on it.
    try:
        import requests  # noqa: F401
        return True
    except ImportError:  # pragma: no cover
        return False


def _load_settings() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
    except ImportError:  # tests / non-Hermes context: env + defaults only
        return {}
    cfg = load_config() or {}
    scoped = cfg.get("image_gen") if isinstance(cfg, dict) else None
    scoped = scoped.get("9router") if isinstance(scoped, dict) else None
    return scoped if isinstance(scoped, dict) else {}


def _base_url(settings: Dict[str, Any]) -> str:
    url = str(
        settings.get("base_url")
        or os.environ.get("NINEROUTER_URL")
        or DEFAULT_BASE_URL
    ).strip()
    return url.rstrip("/")


def _api_key() -> str:
    # ponytail: get_secret is the scoped path; os.environ is the single-profile fallback.
    try:
        from agent.secret_scope import get_secret

        key = get_secret("NINEROUTER_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("NINEROUTER_KEY", "").strip()


class NineRouterImageProvider(ImageGenProvider):
    """Text-to-image through a 9Router gateway (OpenAI-compatible images API)."""

    def __init__(self) -> None:
        self._models_cache: Optional[Tuple[List[str], float]] = None

    @property
    def name(self) -> str:
        return "9router"

    @property
    def display_name(self) -> str:
        return "9Router"

    def is_available(self) -> bool:
        return _http()

    def list_models(self) -> List[Dict[str, Any]]:
        models = self._fetch_models()
        return [
            {
                "id": m,
                "display": m.split("/")[-1],
                "strengths": "via 9Router gateway",
            }
            for m in models
        ]

    def default_model(self) -> Optional[str]:
        settings = _load_settings()
        model = str(settings.get("model") or "").strip()
        if model:
            return model
        env_model = os.environ.get("NINEROUTER_IMAGE_MODEL", "").strip()
        if env_model:
            return env_model
        models = self._fetch_models()
        return models[0] if models else None

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "9Router",
            "tag": "Text-to-image via a local or remote 9Router gateway",
            "env_vars": [
                {
                    "key": "NINEROUTER_KEY",
                    "prompt": "9Router API key (skip for auth-free local gateway)",
                },
            ],
        }

    def _fetch_models(self) -> List[str]:
        if self._models_cache is not None and time.monotonic() - self._models_cache[1] < _MODELS_CACHE_TTL:
            return self._models_cache[0]
        settings = _load_settings()
        base = _base_url(settings)
        headers = {"Authorization": f"Bearer {_api_key()}"} if _api_key() else {}
        try:
            response = _request_json("GET", f"{base}/v1/models/image", headers=headers, timeout=15)
            models = [
                str(entry.get("id"))
                for entry in (response.get("data") or [])
                if isinstance(entry, dict) and entry.get("id")
            ]
            self._models_cache = (models, time.monotonic())
            return models
        except Exception as exc:  # noqa: BLE001 — catalog is optional; generate() reports real errors
            logger.debug("9Router model catalog unavailable: %s", exc)
            return []

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect_ratio = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt is required",
                error_type="invalid_input",
                provider=self.name,
                prompt="",
                aspect_ratio=aspect_ratio,
            )

        settings = _load_settings()
        base = _base_url(settings)
        timeout = int(settings.get("timeout") or DEFAULT_TIMEOUT)
        model_id = (
            str(kwargs.get("model") or "").strip()
            or self.default_model()
            or ""
        )
        if not model_id:
            return error_response(
                error="No image model available on 9Router (catalog empty and no model configured)",
                error_type="provider_error",
                provider=self.name,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        # ponytail: some 9Router models (e.g. flux on Workers AI) reject `size`/`n` —
        # aspect_ratio is then only advisory. Retry without extras on schema rejection.
        size = _ASPECT_TO_SIZE.get(aspect_ratio, "1024x1024")
        payload: Dict[str, Any] = {"model": model_id, "prompt": prompt, "size": size, "n": 1}
        retry_payload: Optional[Dict[str, Any]] = {"model": model_id, "prompt": prompt}
        headers = {"Content-Type": "application/json"}
        api_key = _api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = _request_json(
                "POST", f"{base}/v1/images/generations", headers=headers,
                timeout=timeout, json_body=payload,
            )
        except _SchemaRejected as exc:
            if retry_payload is None:
                return error_response(
                    error=f"9Router request failed: {exc}",
                    error_type=type(exc).__name__,
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )
            logger.debug("9Router rejected size/n for %s; retrying bare payload", model_id)
            try:
                response = _request_json(
                    "POST", f"{base}/v1/images/generations", headers=headers,
                    timeout=timeout, json_body=retry_payload,
                )
            except Exception as exc:  # noqa: BLE001 — surface every transport error to the tool result
                return error_response(
                    error=f"9Router request failed: {exc}",
                    error_type=type(exc).__name__,
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )
        except Exception as exc:  # noqa: BLE001 — surface every transport error to the tool result
            return error_response(
                error=f"9Router request failed: {exc}",
                error_type=type(exc).__name__,
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        err = response.get("error")
        if isinstance(err, dict):
            return error_response(
                error=str(err.get("message") or err),
                error_type=str(err.get("type") or "provider_error"),
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        data = response.get("data") or []
        if not data or not isinstance(data[0], dict):
            return error_response(
                error="9Router returned no image data",
                error_type="provider_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )
        entry = data[0]
        if entry.get("url"):
            return success_response(
                image=str(entry["url"]),
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                provider=self.name,
            )
        b64 = entry.get("b64_json")
        if not b64:
            return error_response(
                error="9Router response has neither url nor b64_json",
                error_type="provider_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )
        try:
            path = save_b64_image(str(b64), prefix="9router", extension="png")
        except Exception as exc:  # noqa: BLE001 — decode/save failure must not raise
            return error_response(
                error=f"Failed to save generated image: {exc}",
                error_type=type(exc).__name__,
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )
        return success_response(
            image=str(path),
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            provider=self.name,
            extra={"format": _sniff_format(path)},
        )


def _sniff_format(path) -> str:
    # ponytail: magic-byte sniff (jpeg/png/webp/gif); "png" is the safe default label.
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
        if head.startswith(b"\xff\xd8"):
            return "jpeg"
        if head.startswith(b"\x89PNG"):
            return "png"
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return "webp"
        if head.startswith(b"GIF8"):
            return "gif"
    except Exception:
        pass
    return "png"


def _request_json(method: str, url: str, *, headers: Dict[str, Any], timeout: int,
                  json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import requests

    response = requests.request(
        method, url, headers=headers, timeout=timeout, json=json_body,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"error": {"message": f"HTTP {response.status_code}: non-JSON response", "type": "provider_error"}}
    if response.status_code == 400 and "not allowed" in json.dumps(body)[:600]:
        raise _SchemaRejected(json.dumps(body)[:600])
    return body


def register(ctx) -> None:  # pragma: no cover — wired by Hermes plugin discovery
    ctx.register_image_gen_provider(NineRouterImageProvider())
