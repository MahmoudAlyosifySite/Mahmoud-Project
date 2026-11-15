"""Horus-OSINT online API — free Hugging Face Space (Gradio SDK).

WHY THIS SHAPE
--------------
Two constraints forced it:

1. **HF serverless cannot serve this model.** ``mahmoudalyosify/Horus-OSINT``
   ships a single ``.gguf`` file and HF reports ``inferenceProviderMapping: {}``
   for it. GGUF is llama.cpp's format; serverless inference needs transformers
   weights. No API token changes that.
2. **Docker Spaces now require a paid plan.** The Gradio SDK is still free, and
   a Gradio Space is just a Python process behind a web server — so we run
   llama.cpp *inside* it via ``llama-cpp-python`` and mount our own FastAPI
   routes alongside the Gradio UI.

The result is an **OpenAI-compatible** ``/v1/chat/completions`` endpoint on a
free Space, which is exactly what ``horus_brain/remote_provider.py`` already
speaks. Point HORUS at it and the fine-tune runs online, reachable from a phone.

HONEST LIMITS
-------------
Free Spaces are CPU-only (2 vCPU). An 8B model at Q4 runs at roughly 1–3 tokens
per second: a short answer takes tens of seconds, a full report several minutes.
Moving a model to a server does not make a CPU faster. Free Spaces also sleep
when idle and reload on wake, so the first request after a quiet spell is slow —
HORUS surfaces that as "starting up" rather than an error.
"""

from __future__ import annotations

import os
import time
import uuid
from threading import Lock
from typing import Any

import gradio as gr
from fastapi import APIRouter, HTTPException, Request
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Configuration — override in the Space's Variables and Secrets
# --------------------------------------------------------------------------- #
REPO_ID = os.getenv("MODEL_REPO", "mahmoudalyosify/Horus-OSINT")
MODEL_FILE = os.getenv("MODEL_FILE", "llama-3-8b-instruct.Q4_K_M.gguf")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "horus-osint")
# Set this as a SECRET. Without it the endpoint is open to anyone with the URL.
API_KEY = os.getenv("API_KEY", "").strip()
# Free Spaces have limited RAM; a large context window costs memory that the
# weights need. 4096 is comfortable for a briefing prompt.
CTX = int(os.getenv("CTX_SIZE", "4096"))
THREADS = int(os.getenv("THREADS", "2"))

_llm: Any = None
# llama.cpp is not safe for concurrent generation on one context, and a free
# Space has no headroom for a second copy. Serialising is correct here.
_lock = Lock()


def get_llm() -> Any:
    """Load the model on first use.

    Deferred rather than loaded at import so the Space finishes booting and
    starts answering health checks before the ~5 GB download begins.
    """
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
        _llm = Llama(
            model_path=path,
            n_ctx=CTX,
            n_threads=THREADS,
            # The GGUF carries a Llama-3 chat template; llama-cpp-python applies
            # it, so callers send plain messages and never hand-build prompts.
            chat_format="llama-3",
            verbose=False,
        )
    return _llm


# --------------------------------------------------------------------------- #
# OpenAI-compatible API
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ALIAS
    messages: list[Message]
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    stream: bool = False


# Routes live on a router, not on an app of our own. On a Gradio SDK Space,
# Gradio builds and owns the FastAPI application; creating a second one and
# serving it with uvicorn is what produced
#   [Errno 98] address already in use ('0.0.0.0', 7861)
# — two servers racing for the same port. The router is attached to Gradio's app
# after launch instead, so there is exactly one server and Hugging Face manages
# its lifecycle.
api = APIRouter()


def _check_key(request: Request) -> None:
    """Bearer check. Skipped only when no key is configured.

    Raises:
        HTTPException: 401 when a key is set and the header does not match.
    """
    if not API_KEY:
        return
    header = request.headers.get("authorization", "")
    if header.removeprefix("Bearer ").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


@api.get("/health")
def health() -> dict[str, Any]:
    """Liveness. Deliberately does not touch the model, so it answers instantly
    while the weights are still downloading."""
    return {"status": "ok", "model": MODEL_ALIAS, "loaded": _llm is not None}


@api.get("/v1/models")
def list_models(request: Request) -> dict[str, Any]:
    """Model list, for clients that discover before calling."""
    _check_key(request)
    return {"object": "list", "data": [{"id": MODEL_ALIAS, "object": "model"}]}


@api.post("/v1/chat/completions")
def chat_completions(payload: ChatRequest, request: Request) -> dict[str, Any]:
    """OpenAI-compatible completion.

    Streaming is accepted in the request and answered non-streamed: on a CPU
    Space the whole generation is slow enough that partial delivery buys little,
    and a correct SSE implementation here would be more failure surface than it
    is worth.
    """
    _check_key(request)
    llm = get_llm()
    with _lock:
        result = llm.create_chat_completion(
            messages=[m.model_dump() for m in payload.messages],
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
        )

    choice = result["choices"][0]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ALIAS,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": choice["message"]["content"],
                },
                "finish_reason": choice.get("finish_reason", "stop"),
            }
        ],
        "usage": result.get("usage", {}),
    }


# --------------------------------------------------------------------------- #
# Gradio UI — also what keeps the Space alive under the free SDK
# --------------------------------------------------------------------------- #
def _reply(message: str, history: list[Any]) -> str:
    llm = get_llm()
    messages = [{"role": "system", "content": "You are HORUS, an OSINT and geopolitical analyst."}]
    for turn in history or []:
        if isinstance(turn, dict):
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    messages.append({"role": "user", "content": message})
    with _lock:
        out = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.2)
    return out["choices"][0]["message"]["content"]


with gr.Blocks(title="Horus-OSINT API") as demo:
    gr.Markdown(
        f"""
        # Horus-OSINT — online API

        Serving `{REPO_ID}` over an **OpenAI-compatible** endpoint.

        **Endpoint:** `POST /v1/chat/completions` · **Model name:** `{MODEL_ALIAS}`

        Free CPU hardware: expect roughly 1–3 tokens/second, and a slow first
        request after the Space has been idle.
        """
    )
    gr.ChatInterface(_reply, type="messages")

# Launch, then attach. `prevent_thread_lock=True` makes launch() return the
# FastAPI application Gradio built instead of blocking, so the OpenAI routes can
# be registered on the *same* app and served by the *same* process.
#
# No port is named anywhere: Gradio reads GRADIO_SERVER_PORT, which Hugging Face
# sets. Hardcoding one is what collided in the first place.
if __name__ == "__main__":
    fastapi_app, _local_url, _share_url = demo.queue().launch(
        server_name="0.0.0.0",
        prevent_thread_lock=True,
        show_api=False,
        # Gradio 5 defaults to server-side rendering, which spawns a Node
        # process beside Python. This Space is an API with a token UI attached,
        # so SSR buys nothing and only adds a second runtime that can fail —
        # the "Stopping Node.js server..." line in the logs is that process.
        ssr_mode=False,
    )
    # Register, then move to the front. Starlette matches routes in order and
    # Gradio owns a catch-all for its UI, so appended routes would never be
    # reached — /v1/chat/completions would quietly return the Gradio page
    # instead of a completion.
    first_new = len(fastapi_app.router.routes)
    fastapi_app.include_router(api)
    added = fastapi_app.router.routes[first_new:]
    del fastapi_app.router.routes[first_new:]
    fastapi_app.router.routes[0:0] = added

    # launch() returned instead of blocking, so hold the process open. Gradio's
    # server is already running in its own thread.
    demo.block_thread()
