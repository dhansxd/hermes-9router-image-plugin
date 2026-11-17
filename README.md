# hermes-9router-image-plugin

An additional selectable [Hermes Agent](https://hermes-agent.nousresearch.com) image provider, routing the native `image_generate` tool through a configured [9Router](https://github.com/decolua/9router) gateway. It uses the official `ImageGenProvider` and `PluginContext.register_image_gen_provider()` APIs. Hermes owns discovery, selection and registration cleanup.

Version **0.1.1** targets Hermes **stable0.23.2** (`ae2a81c`) and the 9Router **0.5.119** image API (`1855aa4`). No Hermes core changes or provider overrides are required.

## Install and lifecycle

Run these in the intended Hermes profile/environment:

```bash
hermes plugins install dhansxd/hermes-9router-image-plugin --enable --yes-deps
hermes plugins list
hermes tools
```

In `hermes tools`, select **Image Generation → 9Router** and a gateway model. The plugin manager ID and native image provider ID are both `9router`. Git installs use the root `__init__.py`; no manual copies, symlinks or pip installation are needed.

```bash
hermes plugins update 9router
hermes plugins disable 9router
hermes plugins enable 9router
hermes plugins remove 9router
```

Disabling/removing is handled by the official manager. Before disabling/removing a selected provider, choose another Image Generation provider with `hermes tools` or change `image_gen.provider`. An explicitly selected but unavailable plugin produces a native Hermes configuration error.

## Configure

Use the active profile's `config.yaml` (default `~/.hermes/config.yaml`):

```yaml
plugins:
  enabled:
    - 9router

image_gen:
  provider: 9router
  model: provider/your-image-model  # native Hermes model picker selection
  9router:
    base_url: http://localhost:20128  # replace with your gateway URL
    fallback_models: []              # optional explicit model IDs, in order
    timeout: 180                     # positive seconds per request
```

Put the gateway key in the active profile's `.env`, never in the YAML:

```dotenv
NINEROUTER_KEY=your-9router-api-key
```

An auth-free gateway may omit the key. The gateway URL may include a deployment path and may end in `/v1`; query strings, fragments and embedded credentials are rejected. The URL resolves from `image_gen.9router.base_url` → profile-scoped `NINEROUTER_URL` → `http://localhost:20128`.

Hermes passes the native `image_gen.model` selection as the provider's `model` argument. Primary model precedence is that argument → `image_gen.9router.model` (legacy default, still supported) → profile-scoped `NINEROUTER_IMAGE_MODEL` → first model from `GET /v1/models/image`. A configured/custom model works without a successful catalog request. Catalog results are cached for 300 seconds, separately by profile, gateway and credential.

## Supported behavior

- Text-to-image only. Source/reference images return `modality_unsupported`; editing, reference conditioning, upscale and creative controls are not advertised.
- Sends `POST /v1/images/generations` with `model`, `prompt`, `n: 1` and an aspect-derived `size` (`1792x1024`, `1024x1024`, `1024x1792`). Some gateway adapters ignore size/count: aspect ratio is a request, not an output-dimension guarantee. Returns the first image if the gateway supplies more than one.
- Supports normalized `data[].b64_json` and `data[].url` responses. PNG/JPEG/WebP/GIF signatures determine the saved extension; unknown bytes, invalid base64 and images over 25 MB are rejected. Signature checks are not a full image decode.
- Saves to the native active-profile `cache/generated/images/` directory. Base64 writes publish atomically. URL outputs use Hermes' download helper and its SSRF, redirect and size checks, then receive signature validation. Invalid downloaded files are removed; a failed download returns an error.
- Gateway keys are never sent to returned image URLs. Gateway API requests do not follow redirects and do not use environment proxies or netrc credentials. JSON responses are capped at 36 MB.
- Fallback tries only explicitly configured `fallback_models`, after HTTP 429 or a structured quota/rate-limit error in a successful JSON response. HTTP 401/403 and other hard errors stop immediately. Timeouts, invalid output and cache failures do not trigger another generation request.
- Gateway errors expose safe status/category messages, without raw upstream bodies, returned URLs or transport exception text. Async upstream polling is handled by 9Router.

## Upgrade from 0.1.0

Keep `9router` and existing `image_gen.9router` settings. Lower-precedence config/env defaults are no longer implicit fallback models; put desired alternatives in `fallback_models`. Schema-rejection retries that silently dropped size/count have been removed; fix the model/gateway request when it returns HTTP 400.

The optional pip entry-point name is now `9router`, matching Git installs. If an earlier pip distribution was enabled as `hermes_9router_image`, disable that old alias and enable `9router` after upgrading. Use one installation source; do not enable duplicate Git and pip copies.

## Verification

The regression source in `tests/test_provider.py` requires a real compatible Hermes runtime. It uses temporary profile data and fake transport; it does not contact a gateway. It covers native registration/teardown and provider behavior without replacing the Hermes ABC with a stub.

In an approved development environment with Hermes and this source importable:

```bash
python -m unittest discover -s tests -v
```

Tests and live generation were **deferred for this audit**. Only static source/contract review, in-memory syntax and metadata checks were performed. No workflow or test environment was added.

## License

MIT
