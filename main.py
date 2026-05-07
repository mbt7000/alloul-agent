"""
alloul-agent — Arabic-first AI Agent API
Fast, free, OpenAI-compatible using Groq + Llama 3.3 70B
"""
from __future__ import annotations

import os
import time
import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alloul.agent")

# ── Config ──────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")  # Optional auth key

DEFAULT_SYSTEM = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    "أنت مساعد ذكاء اصطناعي احترافي يعمل لصالح منصة علول. "
    "تجيب بالعربية بشكل افتراضي ما لم يطلب المستخدم غير ذلك. "
    "تساعد في تحليل بيانات الشركات والموظفين، وتقديم توصيات ذكية."
)

# ── Pydantic Models ──────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str        # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.3
    stream: bool = False


class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    model: str
    provider: str


# ── Groq Client ──────────────────────────────────────────────────────────────

def get_groq_client() -> AsyncOpenAI:
    if not GROQ_API_KEY:
        raise HTTPException(503, detail="GROQ_API_KEY not configured")
    return AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Optional API key auth — skip if AGENT_API_KEY not set."""
    if AGENT_API_KEY and x_api_key != AGENT_API_KEY:
        raise HTTPException(401, detail="Invalid API key")


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"alloul-agent started — model: {GROQ_MODEL}")
    yield
    logger.info("alloul-agent shutting down")


app = FastAPI(
    title="alloul-agent",
    description="Arabic-first AI Agent for the Alloul platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model=GROQ_MODEL, provider="groq")


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    """Non-streaming chat completion."""
    client = get_groq_client()
    model = req.model or GROQ_MODEL

    messages = []
    if req.system_prompt or DEFAULT_SYSTEM:
        messages.append({"role": "system", "content": req.system_prompt or DEFAULT_SYSTEM})
    messages.extend([{"role": m.role, "content": m.content} for m in req.messages])

    t0 = time.monotonic()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except Exception as e:
        logger.error(f"Groq error: {e}")
        raise HTTPException(502, detail=f"AI provider error: {e}")

    latency = int((time.monotonic() - t0) * 1000)
    content = resp.choices[0].message.content or ""
    return ChatResponse(content=content, model=model, provider="groq", latency_ms=latency)


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(req: ChatRequest):
    """Streaming chat completion — returns text/event-stream."""
    client = get_groq_client()
    model = req.model or GROQ_MODEL

    messages = []
    if req.system_prompt or DEFAULT_SYSTEM:
        messages.append({"role": "system", "content": req.system_prompt or DEFAULT_SYSTEM})
    messages.extend([{"role": m.role, "content": m.content} for m in req.messages])

    async def generate() -> AsyncGenerator[str, None]:
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield f"data: {delta}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/complete", dependencies=[Depends(verify_api_key)])
async def complete(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.3,
):
    """Simple one-shot completion (form params)."""
    req = ChatRequest(
        messages=[Message(role="user", content=user_prompt)],
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return await chat(req)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
