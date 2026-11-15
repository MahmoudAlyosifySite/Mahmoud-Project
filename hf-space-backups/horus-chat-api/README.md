---
title: Horus OSINT API
emoji: 👁️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: OpenAI-compatible API for the Horus-OSINT model
---

# Horus-OSINT — online API on a **free** Space

Serves `mahmoudalyosify/Horus-OSINT` over an **OpenAI-compatible** HTTP API, so
HORUS SENTINEL can use the fine-tune from any device, including a phone.

## Why this shape

Two constraints forced it:

1. **HF serverless cannot serve this model.** The repo contains one file —
   `llama-3-8b-instruct.Q4_K_M.gguf` — and HF reports
   `inferenceProviderMapping: {}` for it. GGUF is llama.cpp's format; serverless
   inference needs transformers weights (`.safetensors`). **No API token changes
   that.**
2. **Docker Spaces now require a paid plan.** The **Gradio SDK is still free**,
   and a Gradio Space is just a Python process behind a web server — so this runs
   llama.cpp inside it via `llama-cpp-python` and mounts FastAPI routes next to
   the Gradio UI.

Net effect: a real OpenAI-compatible endpoint, on free hardware, serving your
actual fine-tune.

## Deploying it

1. **New Space** → SDK **Gradio** → hardware **CPU basic (free)** → Public.
2. Upload three files to the Space root:
   - `app.py`
   - `requirements.txt`
   - `README.md` (this file — the YAML header above is what configures the Space)
3. **Settings → Variables and secrets → New secret**
   - Name `API_KEY`, value a long random string you generate.

   Without it the endpoint is open to anyone who finds the URL.
4. Wait for the build, then send one request to trigger the ~4.9 GB model
   download. **Give the first call several minutes.**

## Wiring it into HORUS

Command Center → brain chip → **Brain setup**:

| Field | Value |
|---|---|
| Backend | **Remote API (online)** |
| API base URL | `https://<user>-<space>.hf.space/v1` |
| API key | your `API_KEY` secret |
| Model name | `horus-osint` |

## If the Space fails to start

**`No @spaces.GPU function detected during startup`** — the Space is on
**ZeroGPU** hardware, which requires at least one `@spaces.GPU`-decorated
function and kills the container when it finds none. This app has none by
design.

Fix it in **Settings → Hardware → CPU basic (free)**. Do not add the decorator:
ZeroGPU would not help here anyway. `llama-cpp-python`'s CPU wheel does not use
CUDA, and ZeroGPU caps each call at around a minute — long enough for a short
answer, nowhere near enough for a full report.



`[Errno 98] address already in use` means two servers are racing for the port.
`app.py` must **not** call `uvicorn.run()` or name a port: on a Gradio SDK Space,
Gradio builds and owns the FastAPI application, and Hugging Face manages the
process. The OpenAI routes are attached to *Gradio's* app after `launch()`
returns, so there is exactly one server.

## Checking it by hand

```bash
curl https://<user>-<space>.hf.space/health

curl https://<user>-<space>.hf.space/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"horus-osint","messages":[{"role":"user","content":"What is OSINT?"}],"max_tokens":64}'
```

## What to expect on free hardware

2 vCPU, 16 GB RAM, no GPU. An 8B model at Q4 runs at roughly **1–3 tokens per
second**: a short answer takes tens of seconds, a full report several minutes.
Hosting does not make a CPU faster — this is the same arithmetic as running it
on a laptop without a GPU.

Free Spaces also **sleep when idle** and reload the model on wake, so the first
request after a quiet period is slow. HORUS shows that as "the endpoint is
starting up" rather than an error.

If you need genuinely fast responses, the options are paid GPU hardware or a
hosted provider that keeps a model warm. There is no free fast path for an 8B
model.

## Tuning

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | *(unset)* | Bearer token. **Set it.** |
| `MODEL_REPO` | `mahmoudalyosify/Horus-OSINT` | Source repo |
| `MODEL_FILE` | `llama-3-8b-instruct.Q4_K_M.gguf` | GGUF filename |
| `MODEL_ALIAS` | `horus-osint` | Name clients pass as `model` |
| `CTX_SIZE` | `4096` | Context window; larger costs RAM |
| `THREADS` | `2` | Match the free tier's 2 vCPU |
