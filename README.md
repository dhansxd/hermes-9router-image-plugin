# hermes-9router-image-plugin

Image-generation backend plugin for [Hermes Agent](https://hermes-agent.nousresearch.com) that routes `image_generate` through a [9Router](https://github.com/decolua/9router) gateway.

## What it does

Replaces the built-in image-generation backend: every `image_generate` call is served by your 9Router instance via its OpenAI-compatible images API (`POST /v1/images/generations`).

- Text-to-image only — `prompt` + `aspect_ratio` (`landscape` | `square` | `portrait`)
- Model from config, or auto-discovered from the gateway's `/v1/models/image` catalog (cached 300s)
- Returns a saved image file: base64 responses are decoded; URL responses are downloaded into the Hermes image cache with SSRF and size checks. If URL caching fails, returns an error instead of an unsafe or expired URL.
- Clear HTTP errors surfaced from the gateway
- Zero extra dependencies — stdlib + `requests` (already in the Hermes runtime)
- Windows / macOS / Linux

## Install

Install and enable through Hermes:

```bash
hermes plugins install dhansxd/hermes-9router-image-plugin --enable --yes-deps
hermes plugins doctor 9router --ci
```

Git installs include a root `__init__.py` that imports the implementation from
`src/hermes_9router_image`. No manual file copies or symlinks are required.
Alternatively, copy the complete repository into `$HERMES_HOME/plugins/9router/`
and enable `9router`. Do not enable the pip entry-point alias as well.

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
