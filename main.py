"""
Project 1: Local LLM API Server
Wraps Ollama with a production-grade FastAPI layer:
- Streaming & non-streaming inference
- Rate limiting, API key auth, request validation
- OpenAPI/Swagger docs auto-generated
"""

import time
import asyncio
import hashlib
from collections import defaultdict
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Local LLM Inference API",
    description="Production FastAPI wrapper around Ollama for local LLM serving.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE_URL = "http://localhost:11434"

# ── Auth ───────────────────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)

# Demo API keys — in production, store hashed keys in PostgreSQL
VALID_API_KEYS = {
    hashlib.sha256("dev-key-123".encode()).hexdigest(),
    hashlib.sha256("prod-key-xyz".encode()).hexdigest(),
}

def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    key_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    if key_hash not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials

# ── Rate Limiting ──────────────────────────────────────────────────────────────
rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 20   # max requests
RATE_LIMIT_WINDOW   = 60   # per 60 seconds

def check_rate_limit(request: Request, api_key: str = Depends(verify_api_key)):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    calls = rate_limit_store[api_key]
    # Purge old entries
    rate_limit_store[api_key] = [t for t in calls if t > window_start]
    if len(rate_limit_store[api_key]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
        )
    rate_limit_store[api_key].append(now)

# ── Schemas ────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    model: str = Field(default="mistral", description="Ollama model name to use")
    prompt: str = Field(..., min_length=1, max_length=8192, description="Input prompt")
    system: Optional[str] = Field(None, description="System prompt")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stream: bool = Field(default=False, description="Enable streaming response")

class GenerateResponse(BaseModel):
    model: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float

class ModelInfo(BaseModel):
    name: str
    size: str
    modified_at: str

# ── Helpers ────────────────────────────────────────────────────────────────────
async def stream_ollama(payload: dict) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/generate", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line:
                    yield f"data: {line}\n\n"
    yield "data: [DONE]\n\n"

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Check server and Ollama connectivity."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()
        return {"status": "ok", "ollama": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {e}")

@app.get("/models", response_model=list[ModelInfo], dependencies=[Depends(verify_api_key)])
async def list_models():
    """List all locally available Ollama models."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        r.raise_for_status()
        models = r.json().get("models", [])
    return [
        ModelInfo(
            name=m["name"],
            size=f"{m.get('size', 0) / 1e9:.1f}GB",
            modified_at=m.get("modified_at", ""),
        )
        for m in models
    ]

@app.post("/generate", dependencies=[Depends(check_rate_limit)])
async def generate(body: GenerateRequest):
    """
    Run inference against a local LLM.
    Supports both streaming (SSE) and blocking responses.
    """
    payload = {
        "model": body.model,
        "prompt": body.prompt,
        "stream": body.stream,
        "options": {
            "temperature": body.temperature,
            "num_predict": body.max_tokens,
        },
    }
    if body.system:
        payload["system"] = body.system

    if body.stream:
        return StreamingResponse(
            stream_ollama(payload),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    # Blocking inference
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e.response.text}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama inference timed out")

    latency_ms = (time.perf_counter() - start) * 1000
    return GenerateResponse(
        model=body.model,
        response=data.get("response", ""),
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
        latency_ms=round(latency_ms, 2),
    )

@app.post("/chat", dependencies=[Depends(check_rate_limit)])
async def chat(request: Request, api_key: str = Depends(verify_api_key)):
    """
    OpenAI-compatible /chat/completions style endpoint.
    Accepts {model, messages[], stream} payload.
    """
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=422, detail="messages array is required")

    payload = {
        "model": body.get("model", "mistral"),
        "messages": messages,
        "stream": body.get("stream", False),
        "options": {
            "temperature": body.get("temperature", 0.7),
            "num_predict": body.get("max_tokens", 512),
        },
    }

    if body.get("stream"):
        async def stream_chat():
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_chat(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
    return r.json()
