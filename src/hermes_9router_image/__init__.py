"""9Router image-generation backend for Hermes Agent.

Text-to-image via a 9Router gateway (``POST {base_url}/v1/images/generations``).
Model selection: ``model`` kwarg → ``image_gen.9router.model`` → ``NINEROUTER_IMAGE_MODEL``
→ first model from the live ``/v1/models/image`` catalog (cached, TTL 300s).

Config (``image_gen.9router`` in config.yaml, non-secret):
    base_url        — gateway URL (default: ``NINEROUTER_URL`` env or ``http://localhost:20128``)
    model           — default model id
    fallback_models — tried in order, ONLY on 429/quota errors of the previous model
    timeout         — per-request timeout seconds (default 180)

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
    save_url_image,
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


def _is_quota_error(response_body: Any) -> bool:
    """429/quota-style failure worth retrying on another model (user-chosen policy)."""
    if not isinstance(response_body, dict):
        return False
    err = response_body.get("error")
    if isinstance(err, dict):
        message = str(err.get("message") or "").lower()
        code = str(err.get("code") or "").lower()
        status = str(err.get("status") or "").lower()
        return (
            status == "429"
            or "429" in message
            or "quota" in message
            or "rate" in message and "limit" in message
            or code in ("429", "rate_limit_exceeded", "insufficient_quota")
        )
    return response_body.get("status") == 429 or "429" in str(response_body.get("message", "")).lower()


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

    def _model_chain(self, kwarg_model: Optional[str]) -> List[str]:
        """Primary → fallback_models (deduped, order preserved). Fallbacks fire on 429/quota only."""
        settings = _load_settings()
        chain: List[str] = []
        for candidate in (
            kwarg_model,
            str(settings.get("model") or "").strip(),
            os.environ.get("NINEROUTER_IMAGE_MODEL", "").strip(),
        ):
            if candidate and candidate not in chain:
                chain.append(candidate)
        raw = settings.get("fallback_models")
        if isinstance(raw, str):
            raw = [raw]
        for candidate in raw or []:
            candidate = str(candidate).strip()
            if candidate and candidate not in chain:
                chain.append(candidate)
        if not chain:
            models = self._fetch_models()
            if models:
                chain.append(models[0])
        return chain

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
        chain = self._model_chain(str(kwargs.get("model") or "").strip() or None)
        if not chain:
            return error_response(
                error="No image model available on 9Router (catalog empty and no model configured)",
                error_type="provider_error",
                provider=self.name,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )
        headers = {"Content-Type": "application/json"}
        api_key = _api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # ponytail: some 9Router models (e.g. flux on Workers AI) reject `size`/`n` —
        # aspect_ratio is then only advisory. Retry without extras on schema rejection.
        size = _ASPECT_TO_SIZE.get(aspect_ratio, "1024x1024")
        last_error: Optional[Dict[str, Any]] = None
        for model_id in chain:
            result = self._generate_one(
                model_id, prompt, aspect_ratio, size=size,
                base=base, headers=headers, timeout=timeout,
            )
            if result.get("success"):
                return result
            # Fallback policy (user-chosen): move to the next model ONLY on 429/quota.
            if _is_quota_error(result.get("_quota_probe")):
                logger.warning(
                    "9Router model %s hit 429/quota; falling back to next model in chain", model_id
                )
                last_error = result
                continue
            return result
        return last_error or error_response(  # pragma: no cover — chain non-empty guarantees a result
            error="9Router request failed", error_type="provider_error",
            provider=self.name, prompt=prompt, aspect_ratio=aspect_ratio,
        )

    def _generate_one(
        self, model_id: str, prompt: str, aspect_ratio: str, *,
        size: str, base: str, headers: Dict[str, Any], timeout: int,
    ) -> Dict[str, Any]:
        """One attempt against one model. Returns the uniform result dict.

        On a quota-shaped error dict the result carries ``_quota_probe`` (the raw
        error body) so the caller can decide on fallback without re-parsing.
        """
        payload: Dict[str, Any] = {"model": model_id, "prompt": prompt, "size": size, "n": 1}
        retry_payload: Dict[str, Any] = {"model": model_id, "prompt": prompt}

        def _fail(error: str, error_type: str, *, probe: Any = None) -> Dict[str, Any]:
            out = error_response(
                error=error, error_type=error_type, provider=self.name, model=model_id,
                prompt=prompt, aspect_ratio=aspect_ratio,
            )
            if probe is not None:
                out["_quota_probe"] = probe
            return out

        try:
            response = _request_json(
                "POST", f"{base}/v1/images/generations", headers=headers,
                timeout=timeout, json_body=payload,
            )
        except _SchemaRejected:
            logger.debug("9Router rejected size/n for %s; retrying bare payload", model_id)
            try:
                response = _request_json(
                    "POST", f"{base}/v1/images/generations", headers=headers,
                    timeout=timeout, json_body=retry_payload,
                )
            except Exception as exc:  # noqa: BLE001 — surface every transport error to the tool result
                return _fail(f"9Router request failed: {exc}", type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 — surface every transport error to the tool result
            return _fail(f"9Router request failed: {exc}", type(exc).__name__)

        err = response.get("error")
        if isinstance(err, dict):
            return _fail(
                str(err.get("message") or err),
                str(err.get("type") or "provider_error"),
                probe=response if _is_quota_error(response) else None,
            )

        data = response.get("data") or []
        if not data or not isinstance(data[0], dict):
            return _fail("9Router returned no image data", "provider_error")
        entry = data[0]
        if entry.get("url"):
            image_url = str(entry["url"])
            try:
                image = str(save_url_image(image_url, prefix="9router"))
            except Exception as exc:  # noqa: BLE001 — fail closed; URL/errors may contain secrets
                return _fail("Could not cache 9Router image URL", type(exc).__name__)
            return success_response(
                image=image,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                provider=self.name,
            )
        b64 = entry.get("b64_json")
        if not b64:
            return _fail("9Router response has neither url nor b64_json", "provider_error")
        try:
            path = save_b64_image(str(b64), prefix="9router", extension="png")
        except Exception as exc:  # noqa: BLE001 — decode/save failure must not raise
            return _fail(f"Failed to save generated image: {exc}", type(exc).__name__)
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
