# hermes-9router-image-plugin

Image-generation backend plugin for [Hermes Agent](https://hermes-agent.nousresearch.com) that routes `image_generate` through a [9Router](https://github.com/decolua/9router) gateway.

## What it does

Replaces the built-in image-generation backend: every `image_generate` call is served by your 9Router instance via its OpenAI-compatible images API (`POST /v1/images/generations`).

- Text-to-image only — `prompt` + `aspect_ratio` (`landscape` | `square` | `portrait`)
- Model from config, or auto-discovered from the gateway's `/v1/models/image` catalog (cached 300s)
- Returns a saved image file (b64 responses are decoded and written to the Hermes image cache) or a URL
- Clear HTTP errors surfaced from the gateway
- Zero extra dependencies — stdlib + `requests` (already in the Hermes runtime)
- Windows / macOS / Linux

## Install

Copy (or symlink) this repo into your Hermes plugins directory as a `9router` backend:

```
~/.hermes/plugins/image_gen/9router/
├── plugin.yaml
└── __init__.py        (this repo's src/hermes_9router_image/__init__.py)
```

Or install the package and point `plugins.enabled` at it.

## Configure

`~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - 9router

image_gen:
  provider: 9router
  9router:
    base_url: http://localhost:20128   # your 9Router gateway
    model: cf/@cf/black-forest-labs/flux-1-schnell
    fallback_models:                   # tried in order, ONLY on 429/quota errors
      - gemini/gemini-3-pro-image-preview
    timeout: 180
```

Secrets go in `~/.hermes/.env` (never in config.yaml):

```
NINEROUTER_KEY=your-9router-api-key
```

Auth-free local gateways work without `NINEROUTER_KEY`.

## Notes

- Some models (e.g. Workers AI flux) reject `size`/`n` parameters — the provider detects the schema rejection and retries with a bare payload; `aspect_ratio` is then advisory.
- Response format is auto-sniffed (jpeg/png/webp/gif) from the decoded bytes.

## Test

```
python3 tests/test_provider.py
```

Offline unit tests with monkeypatched transport — no gateway needed.

## License

MIT
